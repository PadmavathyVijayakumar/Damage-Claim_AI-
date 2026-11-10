"""
Claims intelligence pipeline — orchestrates the full claims workflow.

Stages:
  1. Evidence assembly (inbound + outbound images, sensor logs, WMS records)
  2. Inbound-vs-outbound comparison (determine damage origin)
  3. Inconsistency detection (flag claimant image anomalies)
  4. Claim classification (validity, liability, priority)
  5. Dispute letter generation (for rejected claims)
  6. Output routing (auto-approve / queue for human review / escalate)

Target: 60–70% of claims auto-processed without human intervention.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .classifier import ClaimClassification, ClaimClassifier
from .inconsistency import InconsistencyReport, InconsistencyDetector
from ..evidence.assembler import EvidencePackage, EvidenceAssembler
from ..evidence.comparator import ComparisonResult, ConditionComparator
from ..evidence.chain_of_custody import ChainOfCustodyRecord
from ..evidence.dispute_generator import DisputeLetter, DisputeLetterGenerator

logger = logging.getLogger(__name__)


@dataclass
class ClaimsPipelineResult:
    """Full output of the claims intelligence pipeline."""

    claim_id: str
    sscc: str
    processed_at: str

    evidence: Optional[EvidencePackage] = None
    comparison: Optional[ComparisonResult] = None
    inconsistencies: Optional[InconsistencyReport] = None
    classification: Optional[ClaimClassification] = None
    dispute_letter: Optional[DisputeLetter] = None

    @property
    def final_action(self) -> str:
        """High-level routing decision."""
        if self.classification is None:
            return "error"
        if self.classification.auto_approve:
            return "auto_approve"
        if self.classification.escalate_to_human:
            return "escalate_to_human"
        if self.dispute_letter and self.dispute_letter.recommended_action == "reject_claim":
            return "reject_claim"
        return "queue_for_review"


class ClaimsPipeline:
    """
    End-to-end claims processing pipeline.

    Usage:
        pipeline = ClaimsPipeline.from_config(settings)
        result = pipeline.process(
            claim_id="CLM-2024-001",
            sscc="003714000123456789",
            sku="SKU-9988",
            po_number="PO-44512",
            claimant_name="RetailCo West",
            claim_amount=3200.0,
        )
        dispatch(result.final_action, result)
    """

    def __init__(
        self,
        assembler: EvidenceAssembler,
        comparator: ConditionComparator,
        inconsistency_detector: InconsistencyDetector,
        classifier: ClaimClassifier,
        dispute_generator: DisputeLetterGenerator,
        seen_image_hashes: Optional[set] = None,
    ) -> None:
        self.assembler = assembler
        self.comparator = comparator
        self.inconsistency_detector = inconsistency_detector
        self.classifier = classifier
        self.dispute_generator = dispute_generator
        self.seen_image_hashes: set = seen_image_hashes or set()

    @classmethod
    def from_config(cls, settings) -> "ClaimsPipeline":
        return cls(
            assembler=EvidenceAssembler.from_config(settings),
            comparator=ConditionComparator(),
            inconsistency_detector=InconsistencyDetector(),
            classifier=ClaimClassifier(),
            dispute_generator=DisputeLetterGenerator(
                company_name=getattr(settings, "company_name", "Mindwave Logistics"),
                contact_email=getattr(settings, "claims_email", "claims@mindwave.ai"),
            ),
        )

    def process(
        self,
        claim_id: str,
        sscc: str,
        sku: str,
        po_number: str,
        claimant_name: str = "Retailer",
        claim_amount: Optional[float] = None,
        # Pre-looked-up evidence (from DB or inline calls)
        inbound_severity: str = "none",
        outbound_severity: str = "none",
        carrier_exception_filed: bool = False,
        cold_chain_excursion: bool = False,
        claimant_severity: Optional[str] = None,
        claimant_sscc_in_image: Optional[str] = None,
        claimant_photo_timestamp: Optional[str] = None,
        shipment_departure_timestamp: Optional[str] = None,
        claimant_image_hash: Optional[str] = None,
        inbound_image_keys: Optional[list[str]] = None,
        outbound_image_keys: Optional[list[str]] = None,
        coc_record: Optional[ChainOfCustodyRecord] = None,
        wms_record: Optional[dict] = None,
    ) -> ClaimsPipelineResult:
        """
        Process a damage claim from start to finish.

        Returns:
            ClaimsPipelineResult with all stage outputs and a routing decision.
        """
        processed_at = datetime.utcnow().isoformat() + "Z"
        result = ClaimsPipelineResult(
            claim_id=claim_id, sscc=sscc, processed_at=processed_at
        )

        # Stage 1 — Assemble evidence
        try:
            result.evidence = self.assembler.assemble(
                claim_id=claim_id,
                sscc=sscc,
                sku=sku,
                po_number=po_number,
                inbound_image_keys=inbound_image_keys,
                outbound_image_keys=outbound_image_keys,
                departure_timestamp=shipment_departure_timestamp,
                wms_record=wms_record,
            )
            self.assembler.upload(result.evidence)
        except Exception as exc:
            logger.error("Evidence assembly failed: %s", exc)

        # Stage 2 — Compare inbound vs outbound
        try:
            result.comparison = self.comparator.compare(
                sscc=sscc,
                inbound_severity=inbound_severity,
                outbound_severity=outbound_severity,
                carrier_reported=carrier_exception_filed,
            )
        except Exception as exc:
            logger.error("Comparison failed: %s", exc)
            return result

        # Stage 3 — Inconsistency detection
        try:
            result.inconsistencies = self.inconsistency_detector.check(
                claim_id=claim_id,
                sscc=sscc,
                warehouse_severity=outbound_severity,
                claimant_severity=claimant_severity or outbound_severity,
                claimant_sscc_in_image=claimant_sscc_in_image,
                claimant_photo_timestamp=claimant_photo_timestamp,
                shipment_departure_timestamp=shipment_departure_timestamp,
                claimant_image_hash=claimant_image_hash,
                seen_hashes=self.seen_image_hashes,
            )
            if claimant_image_hash:
                self.seen_image_hashes.add(claimant_image_hash)
        except Exception as exc:
            logger.error("Inconsistency check failed: %s", exc)

        # Stage 4 — Classify claim
        try:
            images_match = not (
                result.inconsistencies
                and result.inconsistencies.has_high_severity
            )
            result.classification = self.classifier.classify(
                claim_id=claim_id,
                damage_origin=result.comparison.damage_origin,
                inbound_severity=inbound_severity,
                outbound_severity=outbound_severity,
                carrier_exception_filed=carrier_exception_filed,
                cold_chain_excursion=cold_chain_excursion,
                claim_amount=claim_amount,
                claimant_images_match_wh_records=images_match,
            )
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            return result

        # Stage 5 — Generate dispute letter if needed
        if result.comparison.should_reject_claim or result.comparison.carrier_responsible:
            try:
                result.dispute_letter = self.dispute_generator.generate(
                    claim_id=claim_id,
                    comparison=result.comparison,
                    coc_record=coc_record,
                    claimant_name=claimant_name,
                    claim_amount=claim_amount,
                )
            except Exception as exc:
                logger.error("Dispute letter generation failed: %s", exc)

        logger.info(
            "Claims pipeline complete claim_id=%s action=%s",
            claim_id, result.final_action,
        )
        return result
