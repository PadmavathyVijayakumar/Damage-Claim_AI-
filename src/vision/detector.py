"""
YOLOv9 / YOLO-World damage detector — Layer 1 of the 4-layer dock camera AI stack.

Locates pallets, cartons, strapping, and wrapping in a camera frame and returns
bounding boxes with class labels for downstream classification.

Target throughput: < 0.4 s/pallet at dock speeds.
Edge deployment: NVIDIA Jetson Orin via TensorRT / ONNX.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Damage classes aligned with the OxMaint 16-subtype benchmark
DAMAGE_CLASSES = [
    "crush",
    "puncture",
    "moisture_stain",
    "tear",
    "seal_failure",
    "open_flap",
    "broken_stringer",
    "leaning_stack",
    "wrap_missing",
    "corner_damage",
    "top_damage",
    "sidewall_damage",
    "label_damage",
    "water_damage",
    "overhang",
    "collapsed_pallet",
]

ZONE_CLASSES = ["pallet", "carton", "shrink_wrap", "strapping", "drum", "crate"]


@dataclass
class Detection:
    """Single object detection result from YOLO."""

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    damage_subtype: Optional[str] = None
    zone: str = "unknown"  # Zone A / B / C
    frame_id: int = 0
    timestamp_ms: float = 0.0


@dataclass
class DetectionResult:
    """Aggregated result for one camera frame."""

    detections: list[Detection] = field(default_factory=list)
    frame_id: int = 0
    camera_id: str = ""
    inference_ms: float = 0.0
    image_path: Optional[str] = None

    @property
    def has_damage(self) -> bool:
        return any(d.class_name in DAMAGE_CLASSES for d in self.detections)

    @property
    def damage_detections(self) -> list[Detection]:
        return [d for d in self.detections if d.class_name in DAMAGE_CLASSES]

    @property
    def high_confidence_damage(self, threshold: float = 0.72) -> list[Detection]:
        return [d for d in self.damage_detections if d.confidence >= threshold]


class DamageDetector:
    """
    Wraps an Ultralytics YOLOv9 / YOLO-World model for warehouse damage detection.

    For edge deployment export the model to ONNX or TensorRT:
        model.export(format='onnx')   # for Jetson / OpenVINO
        model.export(format='engine') # TensorRT on Jetson Orin

    Fine-tuning approach (from Mindwave camera guide):
        1. Collect 2,000–5,000 labelled carton images per operator
        2. Use Roboflow for annotation and augmentation
        3. Fine-tune on the warehouse-specific 16 damage classes
        4. Expected accuracy: 85–90% out-of-the-box; 94–97% after 4–8 weeks
    """

    def __init__(
        self,
        model_path: str = "models/weights/yolov9_damage.pt",
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None
        logger.info(
            "DamageDetector initialised",
            extra={"model": model_path, "device": device},
        )

    def _load_model(self):
        """Lazy-load the YOLO model (avoids import cost at startup)."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
            self._model.to(self.device)
            logger.info("YOLO model loaded from %s on %s", self.model_path, self.device)
        except Exception as exc:
            logger.error("Failed to load YOLO model: %s", exc)
            raise

    def detect(
        self,
        image: np.ndarray,
        camera_id: str = "",
        zone: str = "B",
        frame_id: int = 0,
    ) -> DetectionResult:
        """
        Run inference on a single BGR or RGB frame.

        Args:
            image:     HxWxC numpy array (BGR from OpenCV or RGB).
            camera_id: Identifier for the source camera.
            zone:      Dock zone (A=entry, B=floor, C=staging).
            frame_id:  Monotonic frame counter for the camera stream.

        Returns:
            DetectionResult with all YOLO detections.
        """
        import time

        self._load_model()
        t0 = time.perf_counter()

        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_idx = int(box.cls.item())
                class_name = result.names.get(cls_idx, str(cls_idx))
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=float(box.conf.item()),
                        bbox=(x1, y1, x2, y2),
                        zone=zone,
                        frame_id=frame_id,
                        timestamp_ms=time.time() * 1000,
                    )
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "YOLO inference completed",
            extra={"camera": camera_id, "detections": len(detections), "ms": elapsed_ms},
        )
        return DetectionResult(
            detections=detections,
            frame_id=frame_id,
            camera_id=camera_id,
            inference_ms=elapsed_ms,
        )

    def detect_from_file(self, image_path: str | Path, **kwargs) -> DetectionResult:
        """Convenience wrapper — load an image file and run detect()."""
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        result = self.detect(img, **kwargs)
        result.image_path = str(image_path)
        return result

    def export_onnx(self, output_path: str = "models/weights/yolov9_damage.onnx") -> str:
        """Export the loaded model to ONNX for edge deployment."""
        self._load_model()
        path = self._model.export(format="onnx", dynamic=False, simplify=True)
        logger.info("ONNX model exported to %s", path)
        return str(path)
