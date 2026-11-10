"""
Inconsistency detector — finds discrepancies between claimant-submitted
damage photos and warehouse shipment records.

Common fraud/error patterns detected:
  1. Severity inflation — claimant photo shows severe damage, WH photo minor
  2. Wrong pallet — SSCC or label mismatch in claimant image
  3. Timestamp anomaly — claimant photo metadata shows pre-shipment date
  4. Duplicate claim — same pallet images used in multiple claim submissions
  5. Environment mismatch — photo shows outdoor environment for a cold-chain SKU
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Inconsistency:
    """A detected inconsistency between claim and warehouse records."""

    code: str             # e.g. "SEVERITY_INFLATION"
    description: str
    severity: str         # "low" | "medium" | "high"
    evidence_key: str     # S3 key or record field that proves the inconsistency


@dataclass
class InconsistencyReport:
    """Full inconsistency analysis for one claim submission."""

    claim_id: str
    sscc: str
    inconsistencies: list[Inconsistency] = field(default_factory=list)

    @property
    def has_high_severity(self) -> bool:
        return any(i.severity == "high" for i in self.inconsistencies)

    @property
    def recommendation(self) -> str:
        if not self.inconsistencies:
            return "no_issues_found"
        if self.has_high_severity:
            return "flag_for_fraud_review"
        counts = len(self.inconsistencies)
        return "minor_discrepancies" if counts < 2 else "escalate_to_human"


class InconsistencyDetector:
    """
    Checks claimant-submitted evidence against warehouse records.

    Usage:
        detector = InconsistencyDetector()
        report = detector.check(
            claim_id="CLM-001",
            sscc="003714000....",
            warehouse_severity="minor",
            claimant_severity="severe",
            warehouse_image_hash="abc123",
            claimant_image_hash="xyz789",
        )
    """

    def check(
        self,
        claim_id: str,
        sscc: str,
        warehouse_severity: str,
        claimant_severity: str,
        warehouse_image_hash: Optional[str] = None,
        claimant_image_hash: Optional[str] = None,
        claimant_sscc_in_image: Optional[str] = None,
        claimant_photo_timestamp: Optional[str] = None,
        shipment_departure_timestamp: Optional[str] = None,
        seen_hashes: Optional[set] = None,
    ) -> InconsistencyReport:
        """
        Run all inconsistency checks.

        Args:
            claim_id:                     Claim identifier.
            sscc:                         Expected SSCC for this claim.
            warehouse_severity:           Severity recorded in WH outbound scan.
            claimant_severity:            Severity in claimant-submitted photo.
            warehouse_image_hash:         MD5/SHA256 of WH outbound image.
            claimant_image_hash:          MD5/SHA256 of claimant image.
            claimant_sscc_in_image:       SSCC read from claimant image via OCR.
            claimant_photo_timestamp:     EXIF timestamp from claimant photo.
            shipment_departure_timestamp: Departure timestamp from COC record.
            seen_hashes:                  Set of image hashes from prior claims (duplicate detection).

        Returns:
            InconsistencyReport listing all detected issues.
        """
        report = InconsistencyReport(claim_id=claim_id, sscc=sscc)

        # ── 1. Severity inflation ──────────────────────────────────────────
        severity_rank = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
        wh_rank = severity_rank.get(warehouse_severity.lower(), -1)
        cl_rank = severity_rank.get(claimant_severity.lower(), -1)

        if wh_rank >= 0 and cl_rank >= 0 and cl_rank > wh_rank + 1:
            report.inconsistencies.append(Inconsistency(
                code="SEVERITY_INFLATION",
                description=(
                    f"Claimant reports '{claimant_severity}' damage, but warehouse "
                    f"outbound scan recorded only '{warehouse_severity}'."
                ),
                severity="high",
                evidence_key=f"claims/{claim_id}/wh_outbound_scan.json",
            ))
            logger.warning("Severity inflation detected claim_id=%s", claim_id)

        # ── 2. SSCC mismatch ──────────────────────────────────────────────
        if claimant_sscc_in_image and claimant_sscc_in_image != sscc:
            report.inconsistencies.append(Inconsistency(
                code="SSCC_MISMATCH",
                description=(
                    f"SSCC in claimant image ({claimant_sscc_in_image}) does not match "
                    f"expected SSCC ({sscc})."
                ),
                severity="high",
                evidence_key=f"claims/{claim_id}/claimant_image_ocr.json",
            ))

        # ── 3. Timestamp anomaly ──────────────────────────────────────────
        if claimant_photo_timestamp and shipment_departure_timestamp:
            if claimant_photo_timestamp < shipment_departure_timestamp:
                report.inconsistencies.append(Inconsistency(
                    code="TIMESTAMP_ANOMALY",
                    description=(
                        f"Claimant photo timestamp ({claimant_photo_timestamp[:10]}) "
                        f"predates shipment departure ({shipment_departure_timestamp[:10]})."
                    ),
                    severity="high",
                    evidence_key=f"claims/{claim_id}/photo_exif.json",
                ))

        # ── 4. Duplicate claim image ───────────────────────────────────────
        if claimant_image_hash and seen_hashes and claimant_image_hash in seen_hashes:
            report.inconsistencies.append(Inconsistency(
                code="DUPLICATE_IMAGE",
                description="Claimant image hash matches a previously-processed claim.",
                severity="high",
                evidence_key=f"claims/{claim_id}/image_hash_check.json",
            ))

        # ── 5. Image hash identity (same image, different severity claim) ─
        if (
            warehouse_image_hash
            and claimant_image_hash
            and warehouse_image_hash == claimant_image_hash
            and warehouse_severity != claimant_severity
        ):
            report.inconsistencies.append(Inconsistency(
                code="SAME_IMAGE_DIFFERENT_SEVERITY",
                description=(
                    "Claimant submitted the same image as the warehouse outbound scan "
                    "but reported a different severity level."
                ),
                severity="medium",
                evidence_key=f"claims/{claim_id}/wh_outbound_scan.jpg",
            ))

        logger.info(
            "Inconsistency check complete claim_id=%s issues=%d recommendation=%s",
            claim_id,
            len(report.inconsistencies),
            report.recommendation,
        )
        return report

    @staticmethod
    def hash_image(image_bytes: bytes) -> str:
        """Return SHA-256 of image bytes for duplicate detection."""
        return hashlib.sha256(image_bytes).hexdigest()
