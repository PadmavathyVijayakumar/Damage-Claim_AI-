"""
4-layer dock camera AI pipeline orchestrator.

Layer 1: YOLOv9 / YOLO-World — detection
Layer 2: ResNet-50 / EfficientNet — severity + type classification
Layer 3: PaddleOCR / Textract — label reading
Layer 4: GPT-4o Vision — ambiguous case review (5–10% of detections)

Edge deployment:
  YOLO + classifier run on NVIDIA Jetson Orin or Intel Atom x6000 via
  ONNX/TensorRT. Latency < 100 ms. No cloud dependency for layers 1–2.
  Annotated images + metadata are pushed to S3 for analytics.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .classifier import ClassificationResult, DamageClassifier, DamageSeverity
from .detector import DamageDetector, DetectionResult
from .ocr import LabelData, LabelOCRPipeline
from .vllm_reviewer import VLLMReviewResult, VLLMReviewRouter

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full pipeline output for one camera frame / pallet event."""

    camera_id: str
    zone: str
    frame_id: int
    timestamp_iso: str

    # Layer 1 — detection
    detection: DetectionResult = field(default_factory=DetectionResult)

    # Layer 2 — classification
    classifications: list[ClassificationResult] = field(default_factory=list)

    # Layer 3 — OCR
    label: Optional[LabelData] = None

    # Layer 4 — VLLM review (populated only when confidence is low)
    vllm_review: Optional[VLLMReviewResult] = None

    # Derived outputs
    overall_severity: DamageSeverity = DamageSeverity.NONE
    trigger_claim: bool = False
    escalate_to_human: bool = False

    # Timing
    total_ms: float = 0.0

    def to_wms_payload(self) -> dict:
        """
        Serialises the result into the WMS receiving record attachment format.
        Compatible with Manhattan Active, SAP EWM, Blue Yonder, and Oracle WMS.
        """
        return {
            "pallet_sscc": self.label.sscc if self.label else None,
            "sku": self.label.sku if self.label else None,
            "camera_id": self.camera_id,
            "zone": self.zone,
            "timestamp": self.timestamp_iso,
            "damage_detected": self.detection.has_damage,
            "severity": self.overall_severity.value,
            "trigger_claim": self.trigger_claim,
            "escalate": self.escalate_to_human,
            "frame_id": self.frame_id,
            "inference_ms": self.total_ms,
        }


class DockCameraPipeline:
    """
    Orchestrates the 4-layer AI stack for a single dock zone.

    Typical usage:

        pipeline = DockCameraPipeline.from_config(settings)
        result = pipeline.process_frame(frame_array, camera_id="CAM-B1")
        wms_client.attach_to_receiving_record(result.to_wms_payload())
    """

    def __init__(
        self,
        detector: DamageDetector,
        classifier: DamageClassifier,
        ocr_pipeline: LabelOCRPipeline,
        vllm_router: VLLMReviewRouter,
        zone: str = "B",
    ) -> None:
        self.detector = detector
        self.classifier = classifier
        self.ocr_pipeline = ocr_pipeline
        self.vllm_router = vllm_router
        self.zone = zone
        self._frame_counter = 0

    @classmethod
    def from_config(cls, settings) -> "DockCameraPipeline":
        """Build a pipeline from application settings."""
        from .vllm_reviewer import GPT4VisionReviewer

        detector = DamageDetector(
            model_path=settings.yolo_model_path,
            device="cuda" if settings.environment == "production" else "cpu",
        )
        classifier = DamageClassifier(
            model_path=settings.classifier_model_path,
            confidence_threshold=settings.vllm_confidence_threshold,
        )
        ocr = LabelOCRPipeline()
        reviewer = GPT4VisionReviewer(api_key=settings.openai_api_key)
        vllm_router = VLLMReviewRouter(reviewer=reviewer)
        return cls(detector=detector, classifier=classifier, ocr_pipeline=ocr, vllm_router=vllm_router)

    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "",
        label_crop: Optional[np.ndarray] = None,
    ) -> PipelineResult:
        """
        Run the full 4-layer stack on one camera frame.

        Args:
            frame:       HxWx3 BGR numpy array from OpenCV.
            camera_id:   Source camera identifier.
            label_crop:  Optional separate crop of the label for OCR (Zone C).

        Returns:
            PipelineResult with all layer outputs and derived claim signals.
        """
        import datetime

        t0 = time.perf_counter()
        self._frame_counter += 1
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        result = PipelineResult(
            camera_id=camera_id,
            zone=self.zone,
            frame_id=self._frame_counter,
            timestamp_iso=timestamp,
        )

        # ── Layer 1: Detection ────────────────────────────────────────────
        result.detection = self.detector.detect(
            frame,
            camera_id=camera_id,
            zone=self.zone,
            frame_id=self._frame_counter,
        )

        # ── Layer 2: Classify each damage region ──────────────────────────
        import cv2

        h, w = frame.shape[:2]
        worst_severity = DamageSeverity.NONE
        trigger_claim = False

        for detection in result.detection.damage_detections:
            x1, y1, x2, y2 = detection.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

            clf_result = self.classifier.classify(crop)
            result.classifications.append(clf_result)

            if clf_result.should_trigger_claim:
                trigger_claim = True

            # Track worst severity across all detections
            severity_order = list(DamageSeverity)
            if severity_order.index(clf_result.severity) > severity_order.index(worst_severity):
                worst_severity = clf_result.severity

            # ── Layer 4: VLLM review for uncertain cases ──────────────────
            if clf_result.needs_vllm_review and result.vllm_review is None:
                result.vllm_review = self.vllm_router.review_if_needed(
                    crop, clf_result.confidence
                )
                if result.vllm_review and result.vllm_review.recommend_claim:
                    trigger_claim = True

        result.overall_severity = worst_severity
        result.trigger_claim = trigger_claim

        # ── Layer 3: OCR (label crop or Zone C feed) ──────────────────────
        ocr_source = label_crop if label_crop is not None else frame
        if ocr_source is not None:
            label_img = cv2.cvtColor(ocr_source, cv2.COLOR_BGR2RGB)
            result.label = self.ocr_pipeline.read_label(label_img)
            # Warn if SSCC doesn't match anything on the inbound manifest
            if result.label and not result.label.sscc:
                result.escalate_to_human = True

        # Escalate to human review when VLLM found damage but confidence still low
        if result.vllm_review and result.vllm_review.has_damage and not result.trigger_claim:
            result.escalate_to_human = True

        result.total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Pipeline frame processed",
            extra={
                "camera": camera_id,
                "severity": worst_severity.value,
                "claim": trigger_claim,
                "ms": result.total_ms,
            },
        )
        return result
