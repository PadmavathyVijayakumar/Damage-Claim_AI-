"""
Cold-chain prediction pipeline — 4-layer architecture:

  IoT sensors → Edge gateway (LSTM lite) → Cloud ML → Alert outputs

Layer 1: IoT sensor ingest (temp, humidity, door cycles, compressor data)
Layer 2: Edge lightweight LSTM + Isolation Forest pre-filter (ARM gateway)
Layer 3: Cloud LSTM forecaster + LSTM Autoencoder + Isolation Forest + ANN
Layer 4: Alert outputs (30–60 min ahead), compliance logs, dashboard events

Target: ≤ 90 sec deviation response time (vs. hours for manual monitoring).
Prevention benchmark: 82%+ cold chain disruptions preventable with this stack.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from .anomaly_detector import AnomalyScore, IsolationForestDetector, LSTMAutoencoder
from .ann_risk_scorer import ANNRiskScorer, RiskScore
from .lstm_forecaster import LSTMForecast, LSTMTemperatureForecaster

logger = logging.getLogger(__name__)

# Zone-specific temperature thresholds (°C)
ZONE_THRESHOLDS = {
    "pharma_vault":    {"min": 2.0, "max": 8.0,   "critical_max": 10.0},
    "frozen_storage":  {"min": -24.0, "max": -18.0, "critical_min": -15.0},
    "compressor_room": {"min": -20.0, "max": -16.0},
    "loading_dock":    {"min": -2.0, "max": 4.5,   "critical_max": 6.0},
    "ambient_buffer":  {"min": 10.0, "max": 25.0},
}


@dataclass
class ColdChainAlert:
    zone_id: str
    alert_level: str          # "warning" | "critical" | "action_required"
    source: str               # "lstm_forecast" | "ae_anomaly" | "iforest" | "threshold"
    message: str
    predicted_breach_at_min: Optional[int]
    current_temp: float
    threshold: float
    timestamp_iso: str
    recommended_action: str
    claim_risk: bool = False   # True when this event is likely to generate a claim


@dataclass
class ColdChainPipelineResult:
    zone_id: str
    timestamp_iso: str

    # Layer 3 outputs
    lstm_forecast: Optional[LSTMForecast] = None
    ae_score: Optional[AnomalyScore] = None
    iforest_score: Optional[AnomalyScore] = None
    ann_risk: Optional[RiskScore] = None

    # Derived
    alerts: list[ColdChainAlert] = field(default_factory=list)
    overall_risk_level: str = "normal"   # "normal" | "elevated" | "critical"

    @property
    def requires_immediate_action(self) -> bool:
        return self.overall_risk_level == "critical"


class ColdChainPipeline:
    """
    Runs the full 4-layer cold-chain prediction stack for one zone.

    Usage:
        pipeline = ColdChainPipeline.for_zone("pharma_vault")
        result = pipeline.evaluate(sensor_sequence, latest_reading)
        if result.requires_immediate_action:
            alert_ops(result.alerts)
    """

    def __init__(
        self,
        zone_id: str,
        lstm_forecaster: LSTMTemperatureForecaster,
        lstm_ae: LSTMAutoencoder,
        isolation_forest: IsolationForestDetector,
        ann_scorer: ANNRiskScorer,
    ) -> None:
        self.zone_id = zone_id
        self.lstm_forecaster = lstm_forecaster
        self.lstm_ae = lstm_ae
        self.isolation_forest = isolation_forest
        self.ann_scorer = ann_scorer

        self.thresholds = ZONE_THRESHOLDS.get(zone_id, {"min": -99, "max": 99})

    @classmethod
    def for_zone(cls, zone_id: str, settings=None) -> "ColdChainPipeline":
        """Factory — builds all 4 models with settings or defaults."""
        from config.settings import get_settings

        s = settings or get_settings()
        return cls(
            zone_id=zone_id,
            lstm_forecaster=LSTMTemperatureForecaster(model_path=s.lstm_model_path),
            lstm_ae=LSTMAutoencoder(model_path=s.lstm_ae_model_path),
            isolation_forest=IsolationForestDetector(),
            ann_scorer=ANNRiskScorer(model_path=s.ann_risk_model_path),
        )

    def evaluate(
        self,
        sensor_sequence: np.ndarray,     # (seq_len, n_features) recent history
        latest_multi_sensor: np.ndarray,  # (n_iforest_features,) current snapshot
        ann_features: Optional[np.ndarray] = None,
    ) -> ColdChainPipelineResult:
        """
        Run all models and aggregate alerts.

        Args:
            sensor_sequence:     Full history window for LSTM + AE (60+ rows × 8 features).
            latest_multi_sensor: Current 5-feature snapshot for Isolation Forest.
            ann_features:        10-feature vector for ANN risk scoring.

        Returns:
            ColdChainPipelineResult with alerts and risk level.
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        result = ColdChainPipelineResult(zone_id=self.zone_id, timestamp_iso=timestamp)
        alerts: list[ColdChainAlert] = []

        max_threshold = self.thresholds.get("max", 99)
        min_threshold = self.thresholds.get("min", -99)
        critical_max = self.thresholds.get("critical_max", max_threshold + 2)

        current_temp = float(sensor_sequence[-1, 0])  # first feature = temp_outlet

        # ── LSTM forecast ─────────────────────────────────────────────────
        try:
            result.lstm_forecast = self.lstm_forecaster.predict(
                sensor_sequence, zone_id=self.zone_id, alert_threshold=max_threshold
            )
            if result.lstm_forecast.predicted_excursion:
                min_to_breach = result.lstm_forecast.predicted_excursion_at_min
                level = "critical" if (min_to_breach or 99) < 30 else "warning"
                alerts.append(ColdChainAlert(
                    zone_id=self.zone_id,
                    alert_level=level,
                    source="lstm_forecast",
                    message=f"LSTM predicts temperature excursion in {min_to_breach} min "
                            f"(threshold {max_threshold}°C)",
                    predicted_breach_at_min=min_to_breach,
                    current_temp=current_temp,
                    threshold=max_threshold,
                    timestamp_iso=timestamp,
                    recommended_action="increase_monitoring" if level == "warning" else "corrective_action_now",
                    claim_risk=True,
                ))
        except Exception as exc:
            logger.error("LSTM forecast error: %s", exc)

        # ── LSTM Autoencoder anomaly ───────────────────────────────────────
        try:
            result.ae_score = self.lstm_ae.score(sensor_sequence, zone_id=self.zone_id)
            if result.ae_score.is_anomaly:
                alerts.append(ColdChainAlert(
                    zone_id=self.zone_id,
                    alert_level="warning",
                    source="ae_anomaly",
                    message=f"Anomalous sensor pattern detected (reconstruction error "
                            f"{result.ae_score.reconstruction_error:.4f})",
                    predicted_breach_at_min=None,
                    current_temp=current_temp,
                    threshold=result.ae_score.threshold,
                    timestamp_iso=timestamp,
                    recommended_action="investigate_equipment",
                    claim_risk=False,
                ))
        except Exception as exc:
            logger.error("LSTM AE error: %s", exc)

        # ── Isolation Forest cross-sensor ─────────────────────────────────
        try:
            result.iforest_score = self.isolation_forest.score(
                latest_multi_sensor, zone_id=self.zone_id
            )
            if result.iforest_score.is_anomaly:
                alerts.append(ColdChainAlert(
                    zone_id=self.zone_id,
                    alert_level="warning",
                    source="iforest",
                    message="Cross-sensor anomaly: unusual combination of sensor readings",
                    predicted_breach_at_min=None,
                    current_temp=current_temp,
                    threshold=0,
                    timestamp_iso=timestamp,
                    recommended_action="check_door_compressor",
                    claim_risk=False,
                ))
        except Exception as exc:
            logger.error("Isolation Forest error: %s", exc)

        # ── ANN risk scoring (optional — requires pre-computed features) ───
        if ann_features is not None:
            try:
                result.ann_risk = self.ann_scorer.score(
                    ann_features, entity_id=self.zone_id, entity_type="zone"
                )
                if result.ann_risk.is_high_risk:
                    alerts.append(ColdChainAlert(
                        zone_id=self.zone_id,
                        alert_level="warning",
                        source="ann_risk",
                        message=f"ANN risk score {result.ann_risk.risk_score:.2f} — proactive alert",
                        predicted_breach_at_min=None,
                        current_temp=current_temp,
                        threshold=0,
                        timestamp_iso=timestamp,
                        recommended_action=result.ann_risk.recommended_action,
                        claim_risk=True,
                    ))
            except Exception as exc:
                logger.error("ANN risk error: %s", exc)

        # ── Aggregate risk level ──────────────────────────────────────────
        critical_count = sum(1 for a in alerts if a.alert_level == "critical")
        warning_count = len(alerts)
        if critical_count > 0 or current_temp > critical_max:
            result.overall_risk_level = "critical"
        elif warning_count > 0:
            result.overall_risk_level = "elevated"
        else:
            result.overall_risk_level = "normal"

        result.alerts = alerts
        logger.info(
            "Cold chain evaluation complete zone=%s risk=%s alerts=%d",
            self.zone_id, result.overall_risk_level, len(alerts),
        )
        return result
