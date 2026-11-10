"""
Inbound vs outbound condition comparator.

Compares pallet condition at inbound baseline (dock camera) with
outbound shipment photos to determine:
  - Whether damage was pre-existing (retailer claim should be rejected)
  - Whether damage occurred during warehouse custody (valid claim)
  - Whether damage occurred post-departure (carrier responsibility)

Comparison methods:
  1. AI classification scores (inbound severity vs outbound severity)
  2. Chain-of-custody event diff
  3. Visual embedding similarity (CLIP / ViT-B/32 cosine distance)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class DamageOrigin(str, Enum):
    PRE_EXISTING = "pre_existing"           # damage present on inbound
    WAREHOUSE = "warehouse"                  # intact inbound → damaged departure
    CARRIER = "carrier"                      # intact departure → damaged delivery
    UNKNOWN = "unknown"                      # insufficient evidence


@dataclass
class ComparisonResult:
    """Output of one inbound-vs-outbound comparison."""

    sscc: str
    damage_origin: str           # DamageOrigin value
    inbound_severity: str        # from inbound classification
    outbound_severity: str       # from outbound classification
    similarity_score: float      # 0–1 visual similarity (1 = identical condition)
    confidence: float            # 0–1 confidence in damage_origin verdict
    evidence_summary: str        # human-readable explanation
    dispute_recommended: bool    # True when warehouse should dispute the claim

    @property
    def should_reject_claim(self) -> bool:
        """Return True when the evidence shows damage was pre-existing."""
        return self.damage_origin == DamageOrigin.PRE_EXISTING.value

    @property
    def carrier_responsible(self) -> bool:
        return self.damage_origin == DamageOrigin.CARRIER.value


class ConditionComparator:
    """
    Compares pallet condition across custody events.

    Visual comparison uses CLIP ViT-B/32 embeddings when available;
    falls back to classification-score-only comparison when the model
    is not loaded.
    """

    SEVERITY_RANK = {"none": 0, "minor": 1, "moderate": 2, "severe": 3, "unknown": -1}

    def __init__(self, use_clip: bool = True, device: str = "cpu") -> None:
        self.use_clip = use_clip
        self.device = device
        self._clip_model = None
        self._clip_preprocess = None

    def _load_clip(self):
        if self._clip_model is not None:
            return
        try:
            import clip  # pip install git+https://github.com/openai/CLIP.git
            import torch

            self._clip_model, self._clip_preprocess = clip.load(
                "ViT-B/32", device=self.device
            )
            logger.info("CLIP model loaded on %s", self.device)
        except ImportError:
            logger.warning("CLIP not installed — visual similarity disabled")
            self.use_clip = False

    def _visual_similarity(self, img_a, img_b) -> float:
        """
        Compute cosine similarity between two images using CLIP embeddings.
        Returns 0.0 if CLIP is unavailable.
        """
        if not self.use_clip:
            return 0.0
        try:
            import torch
            from PIL import Image as PILImage

            self._load_clip()
            if self._clip_model is None:
                return 0.0

            def encode(img):
                if hasattr(img, "shape"):  # numpy array
                    from PIL import Image
                    img = Image.fromarray(img)
                tensor = self._clip_preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    return self._clip_model.encode_image(tensor)

            feat_a = encode(img_a)
            feat_b = encode(img_b)
            cos_sim = torch.nn.functional.cosine_similarity(feat_a, feat_b)
            return float(cos_sim.item())
        except Exception as exc:
            logger.warning("CLIP similarity error: %s", exc)
            return 0.0

    def compare(
        self,
        sscc: str,
        inbound_severity: str,
        outbound_severity: str,
        inbound_image=None,
        outbound_image=None,
        carrier_reported: bool = False,
    ) -> ComparisonResult:
        """
        Compare inbound vs outbound condition for one pallet.

        Args:
            sscc:               Pallet SSCC.
            inbound_severity:   Severity at inbound scan ("none"|"minor"|"moderate"|"severe").
            outbound_severity:  Severity at departure scan.
            inbound_image:      Optional image array for CLIP comparison.
            outbound_image:     Optional image array for CLIP comparison.
            carrier_reported:   True when carrier has filed an exception at delivery.

        Returns:
            ComparisonResult with damage origin verdict.
        """
        rank_in = self.SEVERITY_RANK.get(inbound_severity.lower(), -1)
        rank_out = self.SEVERITY_RANK.get(outbound_severity.lower(), -1)

        # Visual similarity
        similarity = 1.0
        if inbound_image is not None and outbound_image is not None:
            similarity = self._visual_similarity(inbound_image, outbound_image)

        # Determine origin
        if rank_in < 0 or rank_out < 0:
            origin = DamageOrigin.UNKNOWN
            confidence = 0.3
            summary = "Insufficient severity data to determine damage origin."
        elif rank_in > 0:
            # Already damaged on inbound
            origin = DamageOrigin.PRE_EXISTING
            confidence = 0.90 if similarity < 0.95 else 0.75
            summary = (
                f"Damage ({inbound_severity}) was present at inbound scan. "
                "Claim should be rejected — damage is pre-existing."
            )
        elif rank_out > rank_in:
            # Intact inbound, damaged at departure
            if carrier_reported:
                origin = DamageOrigin.CARRIER
                confidence = 0.70
                summary = (
                    f"Pallet was intact ({inbound_severity}) at departure but "
                    f"carrier filed an exception. Likely carrier responsibility."
                )
            else:
                origin = DamageOrigin.WAREHOUSE
                confidence = 0.85
                summary = (
                    f"Pallet was intact ({inbound_severity}) at inbound but "
                    f"showed {outbound_severity} damage at departure. "
                    "Damage occurred during warehouse custody."
                )
        else:
            origin = DamageOrigin.UNKNOWN
            confidence = 0.5
            summary = "Condition unchanged between inbound and outbound."

        return ComparisonResult(
            sscc=sscc,
            damage_origin=origin.value,
            inbound_severity=inbound_severity,
            outbound_severity=outbound_severity,
            similarity_score=similarity,
            confidence=confidence,
            evidence_summary=summary,
            dispute_recommended=(origin == DamageOrigin.PRE_EXISTING),
        )
