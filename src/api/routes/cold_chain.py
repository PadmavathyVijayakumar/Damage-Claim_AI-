"""
Cold-chain API routes.

POST /api/v1/cold-chain/evaluate   — run full 4-layer prediction for a zone
GET  /api/v1/cold-chain/zones      — list all zones and current risk levels
GET  /api/v1/cold-chain/alerts     — active cold-chain alerts
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Models ────────────────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    temp_outlet: float
    temp_door_zone: float
    temp_cargo_centre: float
    humidity: float
    door_open_binary: float = 0.0
    door_open_freq_1h: float = 0.0
    compressor_current: float = 0.0
    compressor_rpm: float = 0.0


class EvaluateZoneRequest(BaseModel):
    zone_id: str
    sensor_sequence: list[SensorReading] = Field(
        ...,
        description="Time-ordered sensor readings (60 recommended)",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "zone_id": "pharma_vault",
            "sensor_sequence": [
                {
                    "temp_outlet": 4.5,
                    "temp_door_zone": 5.0,
                    "temp_cargo_centre": 4.2,
                    "humidity": 65.0,
                }
            ] * 5,
        }
    }}


class AlertResponse(BaseModel):
    zone_id: str
    alert_level: str
    source: str
    message: str
    predicted_breach_at_min: Optional[int]
    current_temp: float
    threshold: float
    timestamp: str
    recommended_action: str
    claim_risk: bool


class ZoneEvaluationResponse(BaseModel):
    zone_id: str
    overall_risk_level: str
    alerts: list[AlertResponse]
    requires_immediate_action: bool
    timestamp: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/evaluate", response_model=ZoneEvaluationResponse, status_code=status.HTTP_200_OK)
async def evaluate_zone(req: EvaluateZoneRequest):
    """
    Run the 4-layer cold-chain prediction for a zone.
    Returns risk level and any alerts.
    """
    from src.cold_chain.pipeline import ColdChainPipeline

    if len(req.sensor_sequence) == 0:
        raise HTTPException(status_code=422, detail="sensor_sequence must not be empty.")

    # Build numpy arrays
    feature_names = [
        "temp_outlet", "temp_door_zone", "temp_cargo_centre", "humidity",
        "door_open_binary", "door_open_freq_1h", "compressor_current", "compressor_rpm",
    ]
    rows = [[getattr(r, f, 0.0) for f in feature_names] for r in req.sensor_sequence]
    sensor_sequence = np.array(rows, dtype=np.float32)
    latest_multi_sensor = np.array([
        [
            req.sensor_sequence[-1].temp_outlet,
            req.sensor_sequence[-1].humidity,
            req.sensor_sequence[-1].door_open_freq_1h,
            req.sensor_sequence[-1].compressor_current,
            req.sensor_sequence[-1].compressor_rpm,
        ]
    ], dtype=np.float32)[0]

    pipeline = ColdChainPipeline.for_zone(req.zone_id)
    result = pipeline.evaluate(sensor_sequence, latest_multi_sensor)

    return ZoneEvaluationResponse(
        zone_id=result.zone_id,
        overall_risk_level=result.overall_risk_level,
        alerts=[
            AlertResponse(
                zone_id=a.zone_id,
                alert_level=a.alert_level,
                source=a.source,
                message=a.message,
                predicted_breach_at_min=a.predicted_breach_at_min,
                current_temp=a.current_temp,
                threshold=a.threshold,
                timestamp=a.timestamp_iso,
                recommended_action=a.recommended_action,
                claim_risk=a.claim_risk,
            )
            for a in result.alerts
        ],
        requires_immediate_action=result.requires_immediate_action,
        timestamp=result.timestamp_iso,
    )


@router.get("/zones")
async def list_zones():
    """Return all monitored zones with their configured thresholds."""
    from src.cold_chain.pipeline import ZONE_THRESHOLDS

    return [
        {"zone_id": zone_id, "thresholds": thresholds}
        for zone_id, thresholds in ZONE_THRESHOLDS.items()
    ]
