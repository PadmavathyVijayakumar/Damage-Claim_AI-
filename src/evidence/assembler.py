"""
Evidence package assembler.

Collects and packages all evidence for a damage claim:
  - Inbound dock camera images (baseline condition)
  - Cold-chain sensor logs (temperature excursion events)
  - Outbound shipment photos (departure condition)
  - WMS receiving/shipping records
  - BOL scans and seal photos
  - Chain-of-custody timeline

Output: EvidencePackage uploaded to S3, referenced by claim_id.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    """A single piece of evidence in a claim package."""

    item_type: str         # "image" | "sensor_log" | "wms_record" | "bol" | "coc_timeline"
    source: str            # "inbound_camera" | "cold_chain" | "wms" | "outbound_camera"
    s3_key: str            # Location in evidence bucket
    timestamp_iso: str
    description: str
    metadata: dict = field(default_factory=dict)


@dataclass
class EvidencePackage:
    """Complete evidence bundle for one damage claim."""

    claim_id: str
    sscc: str
    sku: str
    po_number: str
    assembled_at: str
    items: list[EvidenceItem] = field(default_factory=list)
    s3_package_key: str = ""   # Key of the uploaded JSON manifest

    @property
    def inbound_images(self) -> list[EvidenceItem]:
        return [i for i in self.items if i.source == "inbound_camera"]

    @property
    def outbound_images(self) -> list[EvidenceItem]:
        return [i for i in self.items if i.source == "outbound_camera"]

    @property
    def temperature_logs(self) -> list[EvidenceItem]:
        return [i for i in self.items if i.source == "cold_chain"]

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "sscc": self.sscc,
            "sku": self.sku,
            "po_number": self.po_number,
            "assembled_at": self.assembled_at,
            "items": [
                {
                    "item_type": i.item_type,
                    "source": i.source,
                    "s3_key": i.s3_key,
                    "timestamp_iso": i.timestamp_iso,
                    "description": i.description,
                    "metadata": i.metadata,
                }
                for i in self.items
            ],
            "s3_package_key": self.s3_package_key,
        }


class EvidenceAssembler:
    """
    Assembles complete evidence packages for damage claims.

    Retrieves evidence from:
      - S3 (dock camera images, outbound photos)
      - InfluxDB (temperature/sensor logs)
      - PostgreSQL (WMS records via SQLAlchemy models)
      - In-memory COC store

    Usage:
        assembler = EvidenceAssembler.from_config(settings)
        package = assembler.assemble(claim_id, sscc, sku, po_number)
        s3_key = assembler.upload(package)
    """

    def __init__(
        self,
        s3_bucket: str,
        s3_client=None,
        influx_client=None,
        db_session=None,
    ) -> None:
        self.s3_bucket = s3_bucket
        self.s3_client = s3_client
        self.influx_client = influx_client
        self.db = db_session

    @classmethod
    def from_config(cls, settings) -> "EvidenceAssembler":
        import boto3

        s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        return cls(s3_bucket=settings.s3_evidence_bucket, s3_client=s3)

    def assemble(
        self,
        claim_id: str,
        sscc: str,
        sku: str,
        po_number: str,
        inbound_image_keys: Optional[list[str]] = None,
        outbound_image_keys: Optional[list[str]] = None,
        sensor_zone_id: Optional[str] = None,
        inbound_timestamp: Optional[str] = None,
        departure_timestamp: Optional[str] = None,
        wms_record: Optional[dict] = None,
    ) -> EvidencePackage:
        """
        Build an EvidencePackage for a claim.

        In production all arguments except claim_id/sscc/sku/po_number are
        looked up from the database; they are accepted here for direct use
        and testing.
        """
        now = datetime.utcnow().isoformat() + "Z"
        package = EvidencePackage(
            claim_id=claim_id,
            sscc=sscc,
            sku=sku,
            po_number=po_number,
            assembled_at=now,
        )

        # ── Inbound camera images ─────────────────────────────────────────
        for key in (inbound_image_keys or []):
            package.items.append(EvidenceItem(
                item_type="image",
                source="inbound_camera",
                s3_key=key,
                timestamp_iso=inbound_timestamp or now,
                description=f"Inbound dock camera — {key.split('/')[-1]}",
                metadata={"sscc": sscc},
            ))

        # ── Outbound photos ───────────────────────────────────────────────
        for key in (outbound_image_keys or []):
            package.items.append(EvidenceItem(
                item_type="image",
                source="outbound_camera",
                s3_key=key,
                timestamp_iso=departure_timestamp or now,
                description=f"Outbound shipment photo — {key.split('/')[-1]}",
                metadata={"sscc": sscc},
            ))

        # ── Cold-chain sensor logs ─────────────────────────────────────────
        if sensor_zone_id and self.influx_client:
            try:
                log_key = self._fetch_sensor_log(
                    sensor_zone_id, inbound_timestamp, departure_timestamp
                )
                if log_key:
                    package.items.append(EvidenceItem(
                        item_type="sensor_log",
                        source="cold_chain",
                        s3_key=log_key,
                        timestamp_iso=now,
                        description=f"Temperature log zone={sensor_zone_id}",
                        metadata={"zone_id": sensor_zone_id},
                    ))
            except Exception as exc:
                logger.warning("Could not fetch sensor log: %s", exc)

        # ── WMS record snapshot ───────────────────────────────────────────
        if wms_record:
            wms_key = f"claims/{claim_id}/wms_record.json"
            package.items.append(EvidenceItem(
                item_type="wms_record",
                source="wms",
                s3_key=wms_key,
                timestamp_iso=now,
                description="WMS receiving / shipping record",
                metadata=wms_record,
            ))

        logger.info(
            "Evidence package assembled claim_id=%s items=%d",
            claim_id, len(package.items),
        )
        return package

    def upload(self, package: EvidencePackage) -> str:
        """Upload evidence manifest JSON to S3; returns the S3 key."""
        if not self.s3_client:
            logger.warning("S3 client not configured — skipping upload")
            return ""

        manifest_key = f"claims/{package.claim_id}/evidence_manifest.json"
        body = json.dumps(package.to_dict(), indent=2).encode()

        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=manifest_key,
            Body=body,
            ContentType="application/json",
        )
        package.s3_package_key = manifest_key
        logger.info("Evidence manifest uploaded to s3://%s/%s", self.s3_bucket, manifest_key)
        return manifest_key

    def _fetch_sensor_log(
        self,
        zone_id: str,
        start: Optional[str],
        stop: Optional[str],
    ) -> Optional[str]:
        """
        Query InfluxDB for sensor readings between start and stop,
        write CSV to S3, return the S3 key.
        """
        # Production: use influxdb_client.QueryApi with Flux query
        # Placeholder returns None for now
        return None
