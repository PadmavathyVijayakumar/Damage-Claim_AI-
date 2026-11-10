"""
LSTM temperature forecaster — primary predictor in the cold-chain prediction pipeline.

Learns sequential temperature patterns (daytime cycles, door-open dips, compressor
ramp-up curves) and predicts the temperature trajectory 30–60 minutes ahead.

Benchmark: MAE ≈ 0.40°C, RMSE ≈ 0.52°C (Portugal 2025 cold-room study comparing
standard / stacked / bidirectional / convolutional LSTM variants against ARIMA).

Deployment:
  - Cloud (training + batch inference)
  - Edge: export to ONNX for ARM/Jetson Orin gateway (lightweight variant)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 60   # 60 one-minute readings = 1 hour lookback window
FORECAST_HORIZON = 60  # 30–60 minute alert window
FEATURE_COLS = [
    "temp_outlet",
    "temp_door_zone",
    "temp_cargo_centre",
    "humidity",
    "door_open_binary",
    "door_open_freq_1h",
    "compressor_current",
    "compressor_rpm",
]


@dataclass
class LSTMForecast:
    """Output from one LSTM inference call."""

    zone_id: str
    forecast_temps: list[float]           # predicted temperatures at each future step
    horizon_minutes: int                  # how far ahead this covers
    alert_threshold: float                # °C threshold for this zone
    predicted_excursion: bool
    predicted_excursion_at_min: Optional[int]  # first minute above threshold
    confidence_interval_low: list[float] = field(default_factory=list)
    confidence_interval_high: list[float] = field(default_factory=list)
    model_mae: float = 0.40               # expected MAE from benchmark

    @property
    def minutes_to_excursion(self) -> Optional[int]:
        return self.predicted_excursion_at_min


class LSTMTemperatureForecaster:
    """
    Multi-layer stacked LSTM trained on multi-point sensor sequences.

    Architecture (based on 2025 Portugal benchmark — stacked LSTM won):
        Input  → LSTM(128) → Dropout(0.2)
               → LSTM(64)  → Dropout(0.2)
               → Dense(32, relu)
               → Dense(FORECAST_HORIZON)

    Training:
        python scripts/train_lstm.py \\
            --data data/processed/cold_chain_sequences.parquet \\
            --epochs 100 \\
            --batch 64 \\
            --output models/weights/lstm_temp.pt
    """

    def __init__(
        self,
        model_path: str = "models/weights/lstm_temp.pt",
        sequence_length: int = SEQUENCE_LENGTH,
        forecast_horizon: int = FORECAST_HORIZON,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.device = device
        self._model = None
        self._scaler = None

    def _build_model(self):
        """Build the stacked LSTM architecture."""
        import torch
        import torch.nn as nn

        class StackedLSTM(nn.Module):
            def __init__(self, input_size: int, hidden_size: int, forecast_horizon: int):
                super().__init__()
                self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.dropout1 = nn.Dropout(0.2)
                self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True)
                self.dropout2 = nn.Dropout(0.2)
                self.fc1 = nn.Linear(hidden_size // 2, 32)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(32, forecast_horizon)

            def forward(self, x):
                out, _ = self.lstm1(x)
                out = self.dropout1(out)
                out, _ = self.lstm2(out)
                out = self.dropout2(out[:, -1, :])  # last timestep
                out = self.relu(self.fc1(out))
                return self.fc2(out)

        return StackedLSTM(
            input_size=len(FEATURE_COLS),
            hidden_size=128,
            forecast_horizon=self.forecast_horizon,
        )

    def _load_model(self):
        if self._model is not None:
            return
        import torch

        model = self._build_model()
        if self.model_path.exists():
            state = torch.load(str(self.model_path), map_location=self.device)
            model.load_state_dict(state)
            logger.info("LSTM model loaded from %s", self.model_path)
        else:
            logger.warning("LSTM weights not found at %s — using untrained model", self.model_path)
        model.eval()
        self._model = model.to(self.device)

    def predict(
        self,
        sensor_sequence: np.ndarray,
        zone_id: str,
        alert_threshold: float = 8.0,
    ) -> LSTMForecast:
        """
        Predict temperature trajectory for the next FORECAST_HORIZON minutes.

        Args:
            sensor_sequence: (sequence_length, n_features) numpy array of recent
                             sensor readings in FEATURE_COLS order.
            zone_id:         Identifier of the cold zone (e.g. "P-3").
            alert_threshold: Temperature ceiling for this zone in °C.

        Returns:
            LSTMForecast with predicted temperatures and excursion flag.
        """
        import torch

        self._load_model()

        if sensor_sequence.shape[0] < self.sequence_length:
            # Pad with the last known reading if we don't have a full window yet
            pad_rows = self.sequence_length - sensor_sequence.shape[0]
            sensor_sequence = np.vstack([
                np.tile(sensor_sequence[0], (pad_rows, 1)),
                sensor_sequence,
            ])

        x = torch.tensor(sensor_sequence[-self.sequence_length:], dtype=torch.float32)
        x = x.unsqueeze(0).to(self.device)  # (1, seq, features)

        with torch.no_grad():
            preds = self._model(x).squeeze().cpu().numpy()

        excursion_steps = np.where(preds > alert_threshold)[0]
        predicted_excursion = len(excursion_steps) > 0

        return LSTMForecast(
            zone_id=zone_id,
            forecast_temps=preds.tolist(),
            horizon_minutes=self.forecast_horizon,
            alert_threshold=alert_threshold,
            predicted_excursion=predicted_excursion,
            predicted_excursion_at_min=int(excursion_steps[0]) if predicted_excursion else None,
        )

    def export_onnx(self, output_path: str = "models/weights/lstm_temp.onnx") -> str:
        """Export to ONNX for edge gateway deployment (ARM microcontrollers)."""
        import torch

        self._load_model()
        dummy = torch.zeros(1, self.sequence_length, len(FEATURE_COLS))
        torch.onnx.export(
            self._model,
            dummy,
            output_path,
            input_names=["sensor_sequence"],
            output_names=["forecast"],
            dynamic_axes={"sensor_sequence": {0: "batch_size"}},
            opset_version=17,
        )
        logger.info("LSTM ONNX exported to %s", output_path)
        return output_path
