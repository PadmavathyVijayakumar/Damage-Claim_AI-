"""
Anomaly detectors for cold-chain monitoring:

  1. LSTM Autoencoder — trains only on normal behaviour, flags high reconstruction error.
     Handles novel failure modes (e.g., a compressor type never seen before).
     Pairs with the LSTM forecaster: forecaster predicts trajectory, AE flags
     when current pattern is already unusual.

  2. Isolation Forest — multi-sensor ensemble method.
     Works on (temperature + humidity + vibration + compressor current)
     simultaneously without deep learning. Edge-ready (scikit-learn ARM).
     Catches cross-sensor patterns the AE may miss, e.g. temp is fine but
     humidity + door-cycle frequency together are unusual.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnomalyScore:
    zone_id: str
    is_anomaly: bool
    score: float                     # higher = more anomalous
    method: str                      # "autoencoder" | "isolation_forest" | "combined"
    reconstruction_error: Optional[float] = None  # AE-specific
    threshold: float = 0.0
    triggered_features: list[str] = None  # which sensors drove the alert


# ── LSTM Autoencoder ─────────────────────────────────────────────────────────

class LSTMAutoencoder:
    """
    Sequence autoencoder trained only on normal temperature patterns.
    Anomaly score = reconstruction MSE vs. a threshold learned from validation data.

    Architecture:
        Encoder: LSTM(64) → LSTM(32)
        Bottleneck: Dense(16)
        Decoder: RepeatVector(seq_len) → LSTM(32) → LSTM(64) → Dense(features)
    """

    def __init__(
        self,
        model_path: str = "models/weights/lstm_ae.pt",
        sequence_length: int = 60,
        n_features: int = 8,
        threshold: float = 0.05,  # MSE threshold — calibrate from validation set
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.threshold = threshold
        self.device = device
        self._model = None

    def _build_model(self):
        import torch.nn as nn

        class AE(nn.Module):
            def __init__(self, seq_len, n_feat):
                super().__init__()
                self.encoder_lstm1 = nn.LSTM(n_feat, 64, batch_first=True)
                self.encoder_lstm2 = nn.LSTM(64, 32, batch_first=True)
                self.bottleneck = nn.Linear(32, 16)
                self.expand = nn.Linear(16, 32)
                self.decoder_lstm1 = nn.LSTM(32, 32, batch_first=True)
                self.decoder_lstm2 = nn.LSTM(32, 64, batch_first=True)
                self.output_layer = nn.Linear(64, n_feat)
                self.seq_len = seq_len

            def forward(self, x):
                out, _ = self.encoder_lstm1(x)
                out, _ = self.encoder_lstm2(out)
                z = self.bottleneck(out[:, -1, :])
                z = self.expand(z).unsqueeze(1).repeat(1, self.seq_len, 1)
                out, _ = self.decoder_lstm1(z)
                out, _ = self.decoder_lstm2(out)
                return self.output_layer(out)

        return AE(self.sequence_length, self.n_features)

    def _load_model(self):
        if self._model is not None:
            return
        import torch

        model = self._build_model()
        if self.model_path.exists():
            model.load_state_dict(torch.load(str(self.model_path), map_location=self.device))
            logger.info("LSTM AE loaded from %s", self.model_path)
        else:
            logger.warning("LSTM AE weights not found — using untrained model")
        model.eval()
        self._model = model.to(self.device)

    def score(self, sensor_sequence: np.ndarray, zone_id: str = "") -> AnomalyScore:
        """
        Compute reconstruction error for a sensor sequence.
        High error → anomaly (pattern the AE was not trained on).
        """
        import torch

        self._load_model()
        x = torch.tensor(sensor_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            reconstructed = self._model(x)

        mse = float(((x - reconstructed) ** 2).mean().item())
        is_anomaly = mse > self.threshold

        logger.debug("LSTM AE score zone=%s mse=%.4f anomaly=%s", zone_id, mse, is_anomaly)
        return AnomalyScore(
            zone_id=zone_id,
            is_anomaly=is_anomaly,
            score=mse,
            method="autoencoder",
            reconstruction_error=mse,
            threshold=self.threshold,
        )


# ── Isolation Forest ──────────────────────────────────────────────────────────

ISOLATION_FOREST_FEATURES = [
    "temp_outlet",
    "humidity",
    "door_open_freq_1h",
    "compressor_current",
    "compressor_rpm",
]


class IsolationForestDetector:
    """
    scikit-learn Isolation Forest for multi-sensor cross-channel anomaly detection.

    Best at catching patterns like:
      "temperature is marginally elevated, humidity is rising, door-open
       frequency is abnormally high — the combination is the signal."

    Edge-ready: runs efficiently on ARM microcontrollers with scikit-learn.

    Training:
        from src.cold_chain.anomaly_detector import IsolationForestDetector
        det = IsolationForestDetector()
        det.fit(normal_sensor_df)   # pass DataFrame with ISOLATION_FOREST_FEATURES
        det.save("models/weights/iforest.joblib")
    """

    def __init__(
        self,
        model_path: str = "models/weights/iforest.joblib",
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.model_path = Path(model_path)
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import joblib

            self._model = joblib.load(str(self.model_path))
            logger.info("Isolation Forest loaded from %s", self.model_path)
        except (FileNotFoundError, Exception) as exc:
            logger.warning("Isolation Forest model not found (%s) — training new", exc)
            from sklearn.ensemble import IsolationForest

            self._model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )

    def fit(self, sensor_df) -> None:
        """Train on a pandas DataFrame of normal sensor readings."""
        from sklearn.ensemble import IsolationForest

        X = sensor_df[ISOLATION_FOREST_FEATURES].values
        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self._model.fit(X)
        logger.info("Isolation Forest trained on %d samples", len(X))

    def save(self, path: Optional[str] = None) -> None:
        import joblib

        save_path = path or str(self.model_path)
        joblib.dump(self._model, save_path)

    def score(self, reading: np.ndarray, zone_id: str = "") -> AnomalyScore:
        """
        Score a single multi-sensor reading.

        Args:
            reading: 1D array in ISOLATION_FOREST_FEATURES order.
            zone_id: Zone identifier.
        """
        self._load_model()
        X = reading.reshape(1, -1)
        pred = self._model.predict(X)[0]  # +1 normal, -1 anomaly
        raw_score = -self._model.score_samples(X)[0]  # higher = more anomalous

        is_anomaly = pred == -1

        # Identify which features drove the anomaly
        triggered = []
        if is_anomaly:
            mean = self._model.estimators_[0].tree_.value  # simplified
            # In production: use SHAP or feature importance from the forest
            triggered = ISOLATION_FOREST_FEATURES  # placeholder

        return AnomalyScore(
            zone_id=zone_id,
            is_anomaly=is_anomaly,
            score=float(raw_score),
            method="isolation_forest",
            triggered_features=triggered if is_anomaly else [],
        )
