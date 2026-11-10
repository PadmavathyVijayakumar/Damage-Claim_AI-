"""
Claim classifier — determines claim type and liability.

Three classification axes:
  1. Validity:  valid | invalid | requires_review
  2. Liability: warehouse | carrier | supplier | retailer | shared
  3. Priority:  critical | high | medium | low

Inputs:
  - Vision pipeline result (inbound condition)
  - COC events
  - Comparator output
  - Claimant's submitted damage photos

Rule engine first; ML confidence re-scores borderline cases.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ClaimValidity(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    PARTIAL = "partial"
    REQUIRES_REVIEW = "requires_review"


class LiabilityParty(str, Enum):
    WAREHOUSE = "warehouse"
    CARRIER = "carrier"
    SUPPLIER = "supplier"
    RETAILER = "retailer"
    SHARED = "shared"
    UNKNOWN = "unknown"


class ClaimPriority(str, Enum):
    CRITICAL = "critical"   # > $10,000 or food safety risk
    HIGH = "high"           # $1,000–$10,000
    MEDIUM = "medium"       # $100–$1,000
    LOW = "low"             # < $100


@dataclass
class ClaimClassification:
    """Full classification output for one claim."""

    claim_id: str
    validity: str           # ClaimValidity value
    liability: str          # LiabilityParty value
    priority: str           # ClaimPriority value
    confidence: float
    reasons: list[str] = field(default_factory=list)
    suggested_resolution: str = ""
    auto_approve: bool = False      # True for low-priority valid claims
    escalate_to_human: bool = False


# Dollar thresholds for priority bucketing
PRIORITY_THRESHOLDS = [
    (10_000, ClaimPriority.CRITICAL),
    (1_000,  ClaimPriority.HIGH),
    (100,    ClaimPriority.MEDIUM),
    (0,      ClaimPriority.LOW),
]


def _derive_priority(amount: Optional[float]) -> ClaimPriority:
    if amount is None:
        return ClaimPriority.MEDIUM
    for threshold, priority in PRIORITY_THRESHOLDS:
        if amount >= threshold:
            return priority
    return ClaimPriority.LOW


class ClaimClassifier:
    """
    Rule-engine + ML hybrid claim classifier.

    Rules fire first; confidence < 0.70 escalates to human review.

    Usage:
        clf = ClaimClassifier()
        result = clf.classify(
            claim_id="CLM-001",
            damage_origin="pre_existing",
            inbound_severity="minor",
            outbound_severity="severe",
            carrier_exception_filed=False,
            claim_amount=5500.0,
        )
    """

    def classify(
        self,
        claim_id: str,
        damage_origin: str,          # from comparator: "pre_existing" | "warehouse" | "carrier" | "unknown"
        inbound_severity: str,
        outbound_severity: str,
        carrier_exception_filed: bool = False,
        cold_chain_excursion: bool = False,
        claim_amount: Optional[float] = None,
        claimant_images_match_wh_records: bool = True,
    ) -> ClaimClassification:
        """
        Classify a damage claim.

        Returns:
            ClaimClassification with validity, liability, priority and resolution guidance.
        """
        reasons: list[str] = []
        confidence = 0.80

        # ── Validity determination ─────────────────────────────────────────
        if damage_origin == "pre_existing":
            validity = ClaimValidity.INVALID
            liability = LiabilityParty.SUPPLIER
            reasons.append("Damage confirmed pre-existing at inbound scan.")
            confidence = 0.90

        elif damage_origin == "warehouse":
            validity = ClaimValidity.VALID
            liability = LiabilityParty.WAREHOUSE
            reasons.append("Damage occurred during warehouse custody.")
            confidence = 0.85

        elif damage_origin == "carrier":
            validity = ClaimValidity.VALID
            liability = LiabilityParty.CARRIER
            reasons.append("Goods were intact at departure; carrier filed exception.")
            confidence = 0.75

        elif cold_chain_excursion:
            validity = ClaimValidity.VALID
            liability = LiabilityParty.WAREHOUSE
            reasons.append("Temperature excursion recorded during warehouse storage.")
            confidence = 0.80

        else:
            validity = ClaimValidity.REQUIRES_REVIEW
            liability = LiabilityParty.UNKNOWN
            reasons.append("Insufficient evidence to determine damage origin.")
            confidence = 0.40

        # ── Claimant image inconsistency flag ─────────────────────────────
        if not claimant_images_match_wh_records:
            reasons.append("Claimant-submitted images do not match warehouse outbound photos.")
            if validity == ClaimValidity.VALID:
                validity = ClaimValidity.PARTIAL
                confidence *= 0.85

        # ── Priority ──────────────────────────────────────────────────────
        priority = _derive_priority(claim_amount)

        # ── Auto-approve / escalate logic ─────────────────────────────────
        auto_approve = (
            validity == ClaimValidity.VALID
            and priority == ClaimPriority.LOW
            and confidence >= 0.80
        )
        escalate = (
            confidence < 0.70
            or priority == ClaimPriority.CRITICAL
            or validity == ClaimValidity.REQUIRES_REVIEW
        )

        # ── Resolution suggestion ──────────────────────────────────────────
        resolution = self._suggest_resolution(validity, liability, priority, reasons)

        logger.info(
            "Claim classified claim_id=%s validity=%s liability=%s priority=%s",
            claim_id, validity.value, liability.value, priority.value,
        )
        return ClaimClassification(
            claim_id=claim_id,
            validity=validity.value,
            liability=liability.value,
            priority=priority.value,
            confidence=confidence,
            reasons=reasons,
            suggested_resolution=resolution,
            auto_approve=auto_approve,
            escalate_to_human=escalate,
        )

    @staticmethod
    def _suggest_resolution(
        validity: ClaimValidity,
        liability: LiabilityParty,
        priority: ClaimPriority,
        reasons: list[str],
    ) -> str:
        if validity == ClaimValidity.INVALID:
            return (
                "Issue dispute letter to claimant. Provide inbound scan evidence "
                "showing pre-existing condition. Recommend claimant pursue supplier."
            )
        if validity == ClaimValidity.VALID and liability == LiabilityParty.CARRIER:
            return (
                "File carrier claim with BOL + departure photos + COC record. "
                "Provide evidence package to carrier's claims portal."
            )
        if validity == ClaimValidity.VALID and liability == LiabilityParty.WAREHOUSE:
            if priority in (ClaimPriority.LOW, ClaimPriority.MEDIUM):
                return "Auto-process claim payment per SLA. Notify ops team."
            return "Escalate to claims manager for approval before payment."
        if validity == ClaimValidity.REQUIRES_REVIEW:
            return "Route to human claims analyst. Collect additional evidence."
        return "Review manually."
