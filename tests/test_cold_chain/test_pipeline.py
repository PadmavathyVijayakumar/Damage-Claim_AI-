"""Tests for the cold-chain prediction pipeline."""
import numpy as np
import pytest

from src.cold_chain.pipeline import (
    ColdChainPipeline,
    ColdChainPipelineResult,
    ZONE_THRESHOLDS,
)
from src.cold_chain.anomaly_detector import IsolationForestDetector, LSTMAutoencoder
from src.cold_chain.lstm_forecaster import LSTMTemperatureForecaster
from src.cold_chain.ann_risk_scorer import ANNRiskScorer


def _make_sequence(n_rows: int = 60, temp: float = 4.0) -> np.ndarray:
    """Create a synthetic sensor sequence for testing."""
    rng = np.random.default_rng(42)
    seq = np.zeros((n_rows, 8), dtype=np.float32)
    seq[:, 0] = temp + rng.normal(0, 0.1, n_rows)   # temp_outlet
    seq[:, 1] = temp + rng.normal(0, 0.15, n_rows)  # temp_door_zone
    seq[:, 2] = temp + rng.normal(0, 0.05, n_rows)  # temp_cargo_centre
    seq[:, 3] = 65.0 + rng.normal(0, 1.0, n_rows)   # humidity
    seq[:, 6] = 5.2 + rng.normal(0, 0.1, n_rows)    # compressor_current
    seq[:, 7] = 2800 + rng.normal(0, 50, n_rows)     # compressor_rpm
    return seq


def _make_pipeline(zone_id: str = "pharma_vault") -> ColdChainPipeline:
    """Build a pipeline with untrained (default-init) models."""
    return ColdChainPipeline(
        zone_id=zone_id,
        lstm_forecaster=LSTMTemperatureForecaster(model_path="models/weights/nonexistent.pt"),
        lstm_ae=LSTMAutoencoder(model_path="models/weights/nonexistent.pt"),
        isolation_forest=IsolationForestDetector(model_path="models/weights/nonexistent.joblib"),
        ann_scorer=ANNRiskScorer(model_path="models/weights/nonexistent.h5"),
    )


class TestZoneThresholds:
    def test_pharma_vault_max(self):
        assert ZONE_THRESHOLDS["pharma_vault"]["max"] == 8.0

    def test_all_zones_have_min_max(self):
        for zone, thresholds in ZONE_THRESHOLDS.items():
            assert "min" in thresholds, f"{zone} missing 'min'"
            assert "max" in thresholds, f"{zone} missing 'max'"


class TestColdChainPipelineResult:
    def test_requires_immediate_action_false_by_default(self):
        result = ColdChainPipelineResult(zone_id="test", timestamp_iso="2024-01-01T00:00:00Z")
        assert not result.requires_immediate_action

    def test_requires_immediate_action_when_critical(self):
        result = ColdChainPipelineResult(
            zone_id="test",
            timestamp_iso="2024-01-01T00:00:00Z",
            overall_risk_level="critical",
        )
        assert result.requires_immediate_action


class TestColdChainPipelineIntegration:
    def test_evaluate_returns_result(self):
        """Pipeline should return a result without raising exceptions."""
        pipeline = _make_pipeline("pharma_vault")
        seq = _make_sequence()
        multi_sensor = seq[-1, [0, 3, 5, 6, 7]]  # 5 iforest features

        try:
            result = pipeline.evaluate(seq, multi_sensor)
            assert isinstance(result, ColdChainPipelineResult)
            assert result.zone_id == "pharma_vault"
            assert result.overall_risk_level in ("normal", "elevated", "critical")
        except ImportError:
            # torch / tensorflow not installed in test environment
            pytest.skip("ML dependencies not available")

    def test_high_temp_triggers_critical(self):
        """A sequence with temperature above critical_max should yield critical risk."""
        pipeline = _make_pipeline("pharma_vault")
        # temp above 10°C (critical_max for pharma_vault)
        seq = _make_sequence(temp=12.0)
        multi_sensor = seq[-1, [0, 3, 5, 6, 7]]

        try:
            result = pipeline.evaluate(seq, multi_sensor)
            # With temperature 12°C (above critical_max 10°C), expect critical
            assert result.overall_risk_level == "critical"
        except ImportError:
            pytest.skip("ML dependencies not available")
