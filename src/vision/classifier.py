"""
ResNet-50 / EfficientNet-B4 damage severity classifier — Layer 2 of the AI stack.

After YOLO (Layer 1) flags a candidate damage region, this classifier assigns:
  - Severity:  none | minor | moderate | severe
  - Type:      crush | puncture | water_stain | open_flap | seal_failure | ...

Training note:
  Transfer-learn from ImageNet, fine-tune on 2,000–5,000 labelled carton crops.
  Expected accuracy: 85–90% pre-tuning; 94–97% after 4–8 weeks of site data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DamageSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


SEVERITY_CLAIM_THRESHOLD = {
    DamageSeverity.NONE: False,
    DamageSeverity.MINOR: False,
    DamageSeverity.MODERATE: True,
    DamageSeverity.SEVERE: True,
}


@dataclass
class ClassificationResult:
    severity: DamageSeverity
    damage_type: str
    confidence: float
    class_probabilities: dict[str, float]
    should_trigger_claim: bool
    crop_path: Optional[str] = None

    @property
    def needs_vllm_review(self) -> bool:
        """Route to Layer 4 (VLLM) when confidence is below threshold."""
        return self.confidence < 0.72


class DamageClassifier:
    """
    Two-stage severity & type classifier.

    Architecture options:
      - EfficientNet-B4  (default, best accuracy/speed tradeoff)
      - ResNet-50        (simpler, easier to deploy on constrained edge hardware)

    Usage:
        clf = DamageClassifier(model_path="models/weights/efficientnet_damage.pt")
        result = clf.classify(crop_array)
        if result.needs_vllm_review:
            # forward to Layer 4 VLLMReviewer
    """

    # Class indices match the training label map
    SEVERITY_LABELS = ["none", "minor", "moderate", "severe"]
    TYPE_LABELS = [
        "crush", "puncture", "water_stain", "open_flap", "seal_failure",
        "corner_damage", "top_damage", "sidewall_damage", "wrap_missing",
        "broken_stringer", "leaning_stack",
    ]

    def __init__(
        self,
        model_path: str = "models/weights/efficientnet_damage.pt",
        architecture: str = "efficientnet_b4",
        num_severity_classes: int = 4,
        confidence_threshold: float = 0.72,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.architecture = architecture
        self.num_severity_classes = num_severity_classes
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import torch
            import torchvision.models as tv_models

            if self.architecture == "efficientnet_b4":
                backbone = tv_models.efficientnet_b4(weights=None)
                in_features = backbone.classifier[1].in_features
                backbone.classifier[1] = torch.nn.Linear(
                    in_features, self.num_severity_classes
                )
            else:  # resnet50
                backbone = tv_models.resnet50(weights=None)
                backbone.fc = torch.nn.Linear(backbone.fc.in_features, self.num_severity_classes)

            if self.model_path.exists():
                state = torch.load(str(self.model_path), map_location=self.device)
                backbone.load_state_dict(state)
                logger.info("Classifier weights loaded from %s", self.model_path)
            else:
                logger.warning(
                    "Classifier weights not found at %s — using random weights (dev mode)",
                    self.model_path,
                )

            backbone.eval()
            self._model = backbone.to(self.device)
        except Exception as exc:
            logger.error("Failed to load classifier: %s", exc)
            raise

    def _preprocess(self, crop: np.ndarray) -> "torch.Tensor":
        import torch
        import torchvision.transforms.functional as F
        from PIL import Image

        img = Image.fromarray(crop)
        img = img.resize((380, 380))  # EfficientNet-B4 input size
        tensor = F.to_tensor(img)
        tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return tensor.unsqueeze(0).to(self.device)

    def classify(self, crop: np.ndarray) -> ClassificationResult:
        """
        Classify a damage crop extracted by the YOLO detector.

        Args:
            crop: HxWx3 numpy RGB array of the bounding-box region.

        Returns:
            ClassificationResult with severity, type, and confidence.
        """
        import torch

        self._load_model()
        tensor = self._preprocess(crop)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        severity_idx = int(np.argmax(probs))
        severity = DamageSeverity(self.SEVERITY_LABELS[severity_idx])
        confidence = float(probs[severity_idx])

        # Heuristic type assignment — replace with a dedicated type head in production
        damage_type = self._infer_type(probs, severity)

        return ClassificationResult(
            severity=severity,
            damage_type=damage_type,
            confidence=confidence,
            class_probabilities={
                label: float(probs[i]) for i, label in enumerate(self.SEVERITY_LABELS)
            },
            should_trigger_claim=SEVERITY_CLAIM_THRESHOLD[severity],
        )

    def _infer_type(self, probs: np.ndarray, severity: DamageSeverity) -> str:
        """
        Simple rule-based type inference from severity distribution shape.
        Replace with a multi-label type head trained on labelled crops.
        """
        if severity == DamageSeverity.NONE:
            return "none"
        # In a real model this would be a separate classification head
        return "crush"
