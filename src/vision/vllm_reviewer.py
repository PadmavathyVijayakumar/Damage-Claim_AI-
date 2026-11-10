"""
VLLM edge-case reviewer — Layer 4 of the 4-layer dock camera AI stack.

Routes only the uncertain 5–10% of YOLO + classifier results to a vision
large language model (GPT-4o Vision or LLaMA 3.2 Vision) for human-labelling
triage and ambiguous-case resolution.

Wayfair's production system ("Tarragon") uses VLLMs specifically for
intelligent sample curation — not for every pallet.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

REVIEW_PROMPT = """You are a warehouse quality-control AI reviewing a cargo image.

Assess the following:
1. Is there visible damage to this pallet or carton? (yes / no / uncertain)
2. If yes, classify the damage type from this list:
   crush | puncture | moisture_stain | tear | seal_failure | open_flap |
   broken_stringer | leaning_stack | wrap_missing | corner_damage |
   top_damage | sidewall_damage | other
3. Rate the severity: none | minor | moderate | severe
4. Would you recommend filing a damage claim? (yes / no)
5. Brief reason (1–2 sentences max).

Respond in JSON only:
{
  "has_damage": true|false,
  "damage_type": "...",
  "severity": "...",
  "recommend_claim": true|false,
  "reason": "..."
}"""


@dataclass
class VLLMReviewResult:
    has_damage: bool
    damage_type: str
    severity: str
    recommend_claim: bool
    reason: str
    model_used: str
    raw_response: str = ""
    error: Optional[str] = None


class GPT4VisionReviewer:
    """
    OpenAI GPT-4o Vision reviewer.
    Uses base64-encoded JPEG crop to avoid S3/URL dependencies.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o",
        max_tokens: int = 256,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

    def review(self, crop: np.ndarray) -> VLLMReviewResult:
        import json

        from openai import OpenAI
        from PIL import Image

        client = OpenAI(api_key=self.api_key)

        # Encode to JPEG bytes → base64
        pil_img = Image.fromarray(crop)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        b64_image = base64.b64encode(buffer.getvalue()).decode()

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": REVIEW_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                    "detail": "low",  # cost control
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw)
            return VLLMReviewResult(
                has_damage=bool(parsed.get("has_damage")),
                damage_type=str(parsed.get("damage_type", "unknown")),
                severity=str(parsed.get("severity", "none")),
                recommend_claim=bool(parsed.get("recommend_claim")),
                reason=str(parsed.get("reason", "")),
                model_used=self.model,
                raw_response=raw,
            )
        except Exception as exc:
            logger.error("GPT-4o Vision review failed: %s", exc)
            return VLLMReviewResult(
                has_damage=False,
                damage_type="unknown",
                severity="none",
                recommend_claim=False,
                reason="Review failed — defaulting to no claim",
                model_used=self.model,
                error=str(exc),
            )


class VLLMReviewRouter:
    """
    Routes uncertain detections to the VLLM reviewer and merges the
    result back into the main classification pipeline.
    """

    def __init__(self, reviewer: Optional[GPT4VisionReviewer] = None) -> None:
        self.reviewer = reviewer

    def should_review(self, classifier_confidence: float) -> bool:
        return classifier_confidence < 0.72

    def review_if_needed(
        self, crop: np.ndarray, classifier_confidence: float
    ) -> Optional[VLLMReviewResult]:
        """
        Returns a VLLMReviewResult only when confidence is below threshold.
        Returns None for high-confidence detections (saves API cost).
        """
        if not self.should_review(classifier_confidence):
            return None
        if self.reviewer is None:
            logger.warning("VLLM reviewer not configured — skipping uncertain case")
            return None
        logger.debug(
            "Routing to VLLM reviewer (classifier confidence=%.3f)", classifier_confidence
        )
        return self.reviewer.review(crop)
