"""
Claims API routes.

POST /api/v1/claims                    — open and process a new claim
GET  /api/v1/claims/{claim_id}         — fetch claim status
PUT  /api/v1/claims/{claim_id}/status  — update claim status
GET  /api/v1/claims/{claim_id}/letter  — get dispute letter (Markdown)
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / response models ─────────────────────────────────────────────────

class OpenClaimRequest(BaseModel):
    sscc: str
    sku: str
    po_number: str
    claimant_name: str
    claim_amount: Optional[float] = None
    inbound_severity: str = "none"
    outbound_severity: str = "none"
    carrier_exception_filed: bool = False
    cold_chain_excursion: bool = False
    claimant_severity: Optional[str] = None

    model_config = {"json_schema_extra": {
        "example": {
            "sscc": "003714000123456789",
            "sku": "SKU-9988",
            "po_number": "PO-44512",
            "claimant_name": "RetailCo West",
            "claim_amount": 3200.0,
            "inbound_severity": "none",
            "outbound_severity": "moderate",
        }
    }}


class ClaimResponse(BaseModel):
    claim_id: str
    sscc: str
    validity: str
    liability: str
    priority: str
    confidence: float
    final_action: str
    suggested_resolution: str
    auto_approve: bool
    escalate_to_human: bool
    dispute_letter_available: bool


class ClaimStatusUpdate(BaseModel):
    status: str = Field(..., description="new_status: open|in_review|closed|disputed")
    notes: Optional[str] = None


# ── In-memory claim store (replace with PostgreSQL in production) ─────────────
_claims: dict[str, dict] = {}
_claim_counter = 0


def _next_claim_id() -> str:
    global _claim_counter
    _claim_counter += 1
    return f"CLM-{_claim_counter:06d}"


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("", response_model=ClaimResponse, status_code=status.HTTP_201_CREATED)
async def open_claim(req: OpenClaimRequest):
    """
    Open and immediately process a damage claim through the AI pipeline.
    Returns the classification result and routing decision.
    """
    from config.settings import get_settings
    from src.claims.pipeline import ClaimsPipeline

    settings = get_settings()
    pipeline = ClaimsPipeline.from_config(settings)
    claim_id = _next_claim_id()

    result = pipeline.process(
        claim_id=claim_id,
        sscc=req.sscc,
        sku=req.sku,
        po_number=req.po_number,
        claimant_name=req.claimant_name,
        claim_amount=req.claim_amount,
        inbound_severity=req.inbound_severity,
        outbound_severity=req.outbound_severity,
        carrier_exception_filed=req.carrier_exception_filed,
        cold_chain_excursion=req.cold_chain_excursion,
        claimant_severity=req.claimant_severity,
    )

    if result.classification is None:
        raise HTTPException(status_code=500, detail="Claim classification failed.")

    # Store for later retrieval
    _claims[claim_id] = {
        "result": result,
        "status": "auto_approved" if result.final_action == "auto_approve" else "open",
        "notes": [],
    }

    return ClaimResponse(
        claim_id=claim_id,
        sscc=req.sscc,
        validity=result.classification.validity,
        liability=result.classification.liability,
        priority=result.classification.priority,
        confidence=result.classification.confidence,
        final_action=result.final_action,
        suggested_resolution=result.classification.suggested_resolution,
        auto_approve=result.classification.auto_approve,
        escalate_to_human=result.classification.escalate_to_human,
        dispute_letter_available=result.dispute_letter is not None,
    )


@router.get("/{claim_id}", response_model=ClaimResponse)
async def get_claim(claim_id: str):
    """Fetch a claim by ID."""
    stored = _claims.get(claim_id)
    if not stored:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found.")

    result = stored["result"]
    clf = result.classification

    return ClaimResponse(
        claim_id=claim_id,
        sscc=result.sscc,
        validity=clf.validity,
        liability=clf.liability,
        priority=clf.priority,
        confidence=clf.confidence,
        final_action=result.final_action,
        suggested_resolution=clf.suggested_resolution,
        auto_approve=clf.auto_approve,
        escalate_to_human=clf.escalate_to_human,
        dispute_letter_available=result.dispute_letter is not None,
    )


@router.get("/{claim_id}/letter")
async def get_dispute_letter(claim_id: str):
    """Return the dispute letter Markdown for a claim."""
    stored = _claims.get(claim_id)
    if not stored:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found.")
    result = stored["result"]
    if result.dispute_letter is None:
        raise HTTPException(status_code=404, detail="No dispute letter for this claim.")
    return {
        "claim_id": claim_id,
        "subject": result.dispute_letter.subject,
        "body_markdown": result.dispute_letter.body,
        "recommended_action": result.dispute_letter.recommended_action,
    }


@router.put("/{claim_id}/status")
async def update_claim_status(claim_id: str, update: ClaimStatusUpdate):
    """Update claim status (e.g. mark as closed after resolution)."""
    stored = _claims.get(claim_id)
    if not stored:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found.")
    stored["status"] = update.status
    if update.notes:
        stored["notes"].append(update.notes)
    return {"claim_id": claim_id, "status": update.status}
