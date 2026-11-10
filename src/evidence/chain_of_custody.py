"""
Chain-of-custody (COC) record management.

Five custody handoff points per shipment:
  1. Inbound baseline  — dock camera captures pallet condition on receipt
  2. Storage custody   — warehouse takes ownership; temp/location logged
  3. Pre-load check    — outbound verification scan before staging
  4. Departure log     — dock door scan; BOL + seal issued; carrier takes custody
  5. Delivery scan     — retailer/DC receiving (if EDI 214 is available)

The COC record is the backbone of every dispute letter: it proves *who* had
the goods *when* and in *what condition*, making warehouse-side liability clear.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CustodyPoint(str, Enum):
    INBOUND_BASELINE = "inbound_baseline"
    STORAGE_CUSTODY = "storage_custody"
    PRE_LOAD_CHECK = "pre_load_check"
    DEPARTURE_LOG = "departure_log"
    DELIVERY_SCAN = "delivery_scan"


@dataclass
class CustodyEvent:
    """A single handoff point in the chain of custody."""

    point: str                    # CustodyPoint value
    timestamp_iso: str
    location: str                 # e.g. "Dock B-7", "Cold Zone P-3", "Bay 12"
    recorded_by: str              # operator / system ID

    # Condition snapshot at this point
    pallet_condition: str         # "intact" | "minor_damage" | "major_damage" | "unknown"
    temperature_c: Optional[float] = None
    seal_number: Optional[str] = None
    bol_number: Optional[str] = None
    carrier_id: Optional[str] = None

    # Evidence references
    image_s3_keys: list[str] = field(default_factory=list)
    sensor_reading_ids: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ChainOfCustodyRecord:
    """Full COC record for one pallet (SSCC)."""

    sscc: str
    sku: str
    po_number: str
    created_at: str
    events: list[CustodyEvent] = field(default_factory=list)

    @property
    def inbound_condition(self) -> Optional[str]:
        for e in self.events:
            if e.point == CustodyPoint.INBOUND_BASELINE.value:
                return e.pallet_condition
        return None

    @property
    def departure_condition(self) -> Optional[str]:
        for e in self.events:
            if e.point == CustodyPoint.DEPARTURE_LOG.value:
                return e.pallet_condition
        return None

    @property
    def damage_occurred_in_warehouse(self) -> Optional[bool]:
        """
        Returns True when the inbound scan showed intact but departure showed damage.
        Returns None when insufficient data is available.
        """
        inbound = self.inbound_condition
        departure = self.departure_condition
        if inbound is None or departure is None:
            return None
        undamaged = {"intact"}
        damaged = {"minor_damage", "major_damage"}
        return inbound in undamaged and departure in damaged

    @property
    def has_complete_custody(self) -> bool:
        """True when all 5 custody points are recorded."""
        recorded = {e.point for e in self.events}
        required = {p.value for p in CustodyPoint}
        return required.issubset(recorded)

    def to_dict(self) -> dict:
        return {
            "sscc": self.sscc,
            "sku": self.sku,
            "po_number": self.po_number,
            "created_at": self.created_at,
            "events": [asdict(e) for e in self.events],
            "damage_in_warehouse": self.damage_occurred_in_warehouse,
            "complete": self.has_complete_custody,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class CustodyRecordStore:
    """
    In-memory + PostgreSQL-backed store for COC records.

    In production: use the SQLAlchemy models in src/api/models/ and flush
    to PostgreSQL after each event. The in-memory dict here is a cache.
    """

    def __init__(self, db_session=None) -> None:
        self._records: dict[str, ChainOfCustodyRecord] = {}
        self._db = db_session

    def create_record(self, sscc: str, sku: str, po_number: str) -> ChainOfCustodyRecord:
        record = ChainOfCustodyRecord(
            sscc=sscc,
            sku=sku,
            po_number=po_number,
            created_at=datetime.utcnow().isoformat() + "Z",
        )
        self._records[sscc] = record
        logger.info("COC record created sscc=%s", sscc)
        return record

    def get_record(self, sscc: str) -> Optional[ChainOfCustodyRecord]:
        return self._records.get(sscc)

    def add_event(self, sscc: str, event: CustodyEvent) -> ChainOfCustodyRecord:
        record = self._records.get(sscc)
        if record is None:
            raise KeyError(f"No COC record found for SSCC {sscc}")
        record.events.append(event)
        logger.info("COC event added sscc=%s point=%s", sscc, event.point)
        return record

    def record_inbound(
        self,
        sscc: str,
        sku: str,
        po_number: str,
        condition: str,
        image_keys: list[str],
        location: str,
        operator: str,
        temperature_c: Optional[float] = None,
    ) -> ChainOfCustodyRecord:
        """Convenience method for the inbound dock camera event."""
        record = self.create_record(sscc, sku, po_number)
        event = CustodyEvent(
            point=CustodyPoint.INBOUND_BASELINE.value,
            timestamp_iso=datetime.utcnow().isoformat() + "Z",
            location=location,
            recorded_by=operator,
            pallet_condition=condition,
            temperature_c=temperature_c,
            image_s3_keys=image_keys,
        )
        record.events.append(event)
        return record

    def record_departure(
        self,
        sscc: str,
        condition: str,
        seal_number: str,
        bol_number: str,
        carrier_id: str,
        image_keys: list[str],
        location: str,
        operator: str,
    ) -> ChainOfCustodyRecord:
        """Convenience method for the departure dock scan."""
        event = CustodyEvent(
            point=CustodyPoint.DEPARTURE_LOG.value,
            timestamp_iso=datetime.utcnow().isoformat() + "Z",
            location=location,
            recorded_by=operator,
            pallet_condition=condition,
            seal_number=seal_number,
            bol_number=bol_number,
            carrier_id=carrier_id,
            image_s3_keys=image_keys,
        )
        return self.add_event(sscc, event)
