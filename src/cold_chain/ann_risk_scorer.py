"""
ANN (Artificial Neural Network) proactive risk scorer.

Analyses historical excursion data to assign risk scores to supply chain
legs, shifts, routes, and product-operator combinations BEFORE the sensor
data even starts moving.

Integrates: route profile, ambient weather forecast, SKU temperature sensitivity,
shift patterns, and equipment age.

Reference: 2021 cold transport study (−20°C coolbox) — ANN models prevented
over 82% of cold chain disruptions before occurrence.

Outputs:
  - Risk score 0.0–1.0 per shipment/pallet
  - High-risk flag (threshold configurable, default 0.7)
  - Recommended pre-action (extra monitoring, re-staging, priority loading)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Feature vector definition — keep in sync with training pipeline
ANN_FEATURES = [
    "route_ambient_forecast_max_c",    # max ambient temp along route (°C)
    "transit_duration_hours",
    "sku_sensitivity_score",           # 0–10 from product master
    "shift_code",                      # encoded: day=0, eve=1, night=2
    "equipment_age_years",
    "compressor_service_overdue_flag",
    "door_open_avg_1h",
    "last_excursion_days_ago",         # days since last excursion in this zone
    "pallet_stack_density",            # 0–1 relative to max rated capacity
    "humidity_forecast_pct",
]

HIGH_RISK_THRESHOLD = 0.70


@dataclass
class RiskScore:
    entity_id: str              # pallet SSCC, shipment ID, or zone ID
    entity_type: str            # "pallet" | "shipment" | "zone" | "shift"
    risk_score: float           # 0.0–1.0
    is_high_risk: bool
    recommended_action: str
    feature_contributions: dict[str, float]  # simplified SHAP-style
    model_version: str = "ann_v1"


RISK_ACTION_MAP = {
    (0.0, 0.3): "monitor_standard",
    (0.3, 0.5): "increase_sensor_frequency",
    (0.5, 0.7): "pre_cool_staging",
    (0.7, 0.85): "priority_loading_plus_alert",
    (0.85, 1.01): "hold_and_review",
}


def _map_action(score: float) -> str:
    for (lo, hi), action in RISK_ACTION_MAP.items():
        if lo <= score < hi:
            return action
    return "hold_and_review"


class ANNRiskScorer:
    """
    Feed-forward ANN trained on historical excursion data.

    Architecture (Keras/TensorFlow):
        Input(len(ANN_FEATURES))
        → Dense(64, relu) → Dropout(0.3)
        → Dense(32, relu) → Dropout(0.2)
        → Dense(16, relu)
        → Dense(1, sigmoid)   # risk probability

    Training:
        python scripts/train_ann.py \\
            --data data/processed/excursion_history.parquet \\
            --epochs 150 \\
            --output models/weights/ann_risk.h5

    Weather forecast integration:
        The scorer ingests route_ambient_forecast_max_c from an external
        weather API (OpenWeatherMap or on-site sensor). High ambient days
        should trigger pre-emptive high-risk flags across all outbound loads.
    """

    def __init__(
        self,
        model_path: str = "models/weights/ann_risk.h5",
        threshold: float = HIGH_RISK_THRESHOLD,
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import tensorflow as tf

            self._model = tf.keras.models.load_model(str(self.model_path))
            logger.info("ANN risk model loaded from %s", self.model_path)
        except Exception as exc:
            logger.warning("ANN model not found (%s) — building untrained model", exc)
            self._model = self._build_model()

    def _build_model(self):
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(len(ANN_FEATURES),)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def score(self, features: np.ndarray, entity_id: str = "", entity_type: str = "pallet") -> RiskScore:
        """
        Score one entity from its feature vector.

        Args:
            features:    1D array in ANN_FEATURES order.
            entity_id:   Pallet SSCC, shipment ID, or zone ID.
            entity_type: Type of entity being scored.

        Returns:
            RiskScore with risk probability and recommended action.
        """
        self._load_model()
        X = features.reshape(1, -1).astype(np.float32)
        risk_prob = float(self._model.predict(X, verbose=0)[0][0])

        # Simplified feature contribution — replace with SHAP in production
        contributions = {
            feat: float(features[i]) for i, feat in enumerate(ANN_FEATURES)
        }

        return RiskScore(
            entity_id=entity_id,
            entity_type=entity_type,
            risk_score=risk_prob,
            is_high_risk=risk_prob >= self.threshold,
            recommended_action=_map_action(risk_prob),
            feature_contributions=contributions,
        )

    def batch_score(
        self, feature_matrix: np.ndarray, entity_ids: list[str], entity_type: str = "pallet"
    ) -> list[RiskScore]:
        """Score a batch of entities (e.g., all pallets in a pick run)."""
        self._load_model()
        X = feature_matrix.astype(np.float32)
        probs = self._model.predict(X, verbose=0).flatten()

        return [
            RiskScore(
                entity_id=eid,
                entity_type=entity_type,
                risk_score=float(p),
                is_high_risk=float(p) >= self.threshold,
                recommended_action=_map_action(float(p)),
                feature_contributions={},
            )
            for eid, p in zip(entity_ids, probs)
        ]
