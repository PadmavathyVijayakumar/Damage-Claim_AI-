"""
Inspection API routes — dock camera pipeline results.

POST /api/v1/inspections/frame          — submit a camera frame for processing
GET  /api/v1/inspections/{id}           — fetch inspection result by ID
GET  /api/v1/inspections/zone/{zone_id} — recent inspections for a dock zone
"""
from __future__ import annotations

import base64
import logging
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / response models ─────────────────────────────────────────────────

class FrameSubmitRequest(BaseModel):
    camera_id: str
    zone: str
    frame_b64: str = Field(..., description="Base64-encoded JPEG frame")
    label_crop_b64: Optional[str] = Field(None, description="Optional label crop for OCR")

    model_config = {"json_schema_extra": {
        "example": {
            "camera_id": "CAM-B1",
            "zone": "B",
            "frame_b64": "<base64 jpeg>",
        }
    }}


class InspectionResponse(BaseModel):
    inspection_id: str
    camera_id: str
    zone: str
    timestamp: str
    has_damage: bool
    severity: str
    trigger_claim: bool
    escalate_to_human: bool
    pallet_sscc: Optional[str]
    inference_ms: float


# ── Dependency: pipeline ─────────────────────────────────────────────────────

def get_pipeline():
    """Return the dock camera pipeline (lazy singleton)."""
    from config.settings import get_settings
    from src.vision.pipeline import DockCameraPipeline

    settings = get_settings()
    return DockCameraPipeline.from_config(settings)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/frame", response_model=InspectionResponse, status_code=status.HTTP_201_CREATED)
async def submit_frame(
    req: FrameSubmitRequest,
    pipeline=Depends(get_pipeline),
):
    """
    Submit a camera frame for damage detection.
    Returns inspection result including severity and claim trigger flag.
    """
    import cv2

    try:
        frame_bytes = base64.b64decode(req.frame_b64)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid frame encoding: {exc}")

    label_crop = None
    if req.label_crop_b64:
        try:
            crop_bytes = base64.b64decode(req.label_crop_b64)
            nparr = np.frombuffer(crop_bytes, np.uint8)
            label_crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            pass  # optional — skip on bad crop

    result = pipeline.process_frame(frame, camera_id=req.camera_id, label_crop=label_crop)

    # Generate a simple inspection ID (use UUID in production)
    import hashlib
    inspection_id = hashlib.md5(
        f"{req.camera_id}{result.frame_id}{result.timestamp_iso}".encode()
    ).hexdigest()[:12]

    return InspectionResponse(
        inspection_id=inspection_id,
        camera_id=result.camera_id,
        zone=result.zone,
        timestamp=result.timestamp_iso,
        has_damage=result.detection.has_damage,
        severity=result.overall_severity.value,
        trigger_claim=result.trigger_claim,
        escalate_to_human=result.escalate_to_human,
        pallet_sscc=result.label.sscc if result.label else None,
        inference_ms=result.total_ms,
    )


@router.get("/zone/{zone_id}", response_model=list[InspectionResponse])
async def get_zone_inspections(zone_id: str, limit: int = 20):
    """
    Fetch recent inspections for a dock zone.
    In production: query PostgreSQL; returns empty list here.
    """
    return []
