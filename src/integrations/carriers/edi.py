"""
Carrier EDI integration.

Supported transaction sets:
  EDI 210  — Motor carrier freight invoice (carrier bills us)
  EDI 214  — Transportation carrier shipment status (delivery confirmation)
  EDI 204  — Motor carrier load tender (we tender a load to carrier)
  EDI 997  — Functional acknowledgement

Transport: AS2 (primary), SFTP (fallback), REST API (modern carriers).

Damage-relevant flow:
  1. EDI 214 arrives → parser extracts delivery exceptions
  2. Exception events are matched against COC departure records
  3. If a delivery exception matches a known damage claim → auto-route to carrier
  4. EDI 210 is parsed to match freight charges against claims amounts
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ── EDI 214 — Shipment Status ─────────────────────────────────────────────────

@dataclass
class ShipmentStatusEvent:
    """One event from an EDI 214 transaction."""

    tracking_number: str
    bol_number: str
    status_code: str         # e.g. "D1" (delivered), "X6" (refused), "X3" (exception)
    status_description: str
    event_timestamp: str
    location_city: str = ""
    location_state: str = ""
    exception_code: Optional[str] = None
    exception_description: Optional[str] = None
    is_damage_exception: bool = False


@dataclass
class EDI214Document:
    """Parsed EDI 214 document."""

    sender_id: str
    receiver_id: str
    interchange_date: str
    events: list[ShipmentStatusEvent] = field(default_factory=list)

    @property
    def damage_exceptions(self) -> list[ShipmentStatusEvent]:
        return [e for e in self.events if e.is_damage_exception]


# Exception codes that indicate damage (carrier-side)
DAMAGE_EXCEPTION_CODES = {
    "X6",   # refused — often due to damage
    "D15",  # delivered with exception
    "X3",   # exception
    "X8",   # damage notation
}

DAMAGE_KEYWORDS = ["damage", "refuse", "shortage", "broken", "crushed", "wet"]


class EDI214Parser:
    """
    Parse EDI 214 (shipment status) flat-file or loop-structured text.
    Minimal parser — production systems use an EDI translator (Gentran, TrueCommerce).
    """

    def parse(self, edi_text: str) -> EDI214Document:
        """Parse EDI 214 flat text into a document with status events."""
        lines = edi_text.strip().split("~")
        sender_id = ""
        receiver_id = ""
        interchange_date = ""
        events: list[ShipmentStatusEvent] = []
        current: dict = {}

        for segment in lines:
            elements = segment.strip().split("*")
            if not elements:
                continue
            seg_id = elements[0].upper()

            if seg_id == "ISA":
                sender_id = elements[6].strip() if len(elements) > 6 else ""
                receiver_id = elements[8].strip() if len(elements) > 8 else ""
                interchange_date = elements[9] if len(elements) > 9 else ""

            elif seg_id == "AT7":
                # Shipment status code
                status_code = elements[1] if len(elements) > 1 else ""
                status_desc = elements[2] if len(elements) > 2 else ""
                ts = elements[6] if len(elements) > 6 else datetime.utcnow().isoformat()
                is_damage = (
                    status_code in DAMAGE_EXCEPTION_CODES
                    or any(kw in status_desc.lower() for kw in DAMAGE_KEYWORDS)
                )
                current["status_code"] = status_code
                current["status_description"] = status_desc
                current["event_timestamp"] = ts
                current["is_damage_exception"] = is_damage

            elif seg_id == "B10":
                current["tracking_number"] = elements[2] if len(elements) > 2 else ""
                current["bol_number"] = elements[1] if len(elements) > 1 else ""

            elif seg_id == "SE":
                # End of transaction set — flush current event
                if current.get("tracking_number"):
                    events.append(ShipmentStatusEvent(**{
                        "tracking_number": current.get("tracking_number", ""),
                        "bol_number": current.get("bol_number", ""),
                        "status_code": current.get("status_code", ""),
                        "status_description": current.get("status_description", ""),
                        "event_timestamp": current.get("event_timestamp", ""),
                        "is_damage_exception": current.get("is_damage_exception", False),
                    }))
                current = {}

        return EDI214Document(
            sender_id=sender_id,
            receiver_id=receiver_id,
            interchange_date=interchange_date,
            events=events,
        )


# ── EDI 210 — Freight Invoice ─────────────────────────────────────────────────

@dataclass
class FreightInvoiceLine:
    bol_number: str
    shipment_date: str
    total_charges: float
    service_code: str
    weight_lbs: float = 0.0
    carrier_scac: str = ""


class EDI210Parser:
    """Parse EDI 210 freight invoice for charge reconciliation."""

    def parse(self, edi_text: str) -> list[FreightInvoiceLine]:
        lines = edi_text.strip().split("~")
        invoices: list[FreightInvoiceLine] = []
        current: dict = {}

        for segment in lines:
            elements = segment.strip().split("*")
            if not elements:
                continue
            seg_id = elements[0].upper()

            if seg_id == "B3":
                current["bol_number"] = elements[2] if len(elements) > 2 else ""
                current["shipment_date"] = elements[7] if len(elements) > 7 else ""
                current["carrier_scac"] = elements[1] if len(elements) > 1 else ""

            elif seg_id == "L9":
                try:
                    current["total_charges"] = float(elements[2]) if len(elements) > 2 else 0.0
                except ValueError:
                    current["total_charges"] = 0.0

            elif seg_id == "L0":
                try:
                    current["weight_lbs"] = float(elements[5]) if len(elements) > 5 else 0.0
                    current["service_code"] = elements[6] if len(elements) > 6 else ""
                except (ValueError, IndexError):
                    pass

            elif seg_id == "SE":
                if current.get("bol_number"):
                    invoices.append(FreightInvoiceLine(
                        bol_number=current.get("bol_number", ""),
                        shipment_date=current.get("shipment_date", ""),
                        total_charges=current.get("total_charges", 0.0),
                        service_code=current.get("service_code", ""),
                        weight_lbs=current.get("weight_lbs", 0.0),
                        carrier_scac=current.get("carrier_scac", ""),
                    ))
                current = {}

        return invoices
