"""
Enterprise-level risk scoring for inbound shipments.

Combines multiple risk signals into a composite score that prioritises
which pallets / shipments need closer inspection or should be held:

  1. Carrier historical damage rate         (carrier_risk)
  2. SKU fragility rating                   (sku_risk)
  3. Transit duration vs. expected          (transit_risk)
  4. Temperature excursion flags            (cold_chain_risk)
  5. Time of receipt (shift / day pattern)  (temporal_risk)
  6. Vendor scorecard                       (vendor_risk)

Output: RiskScore 0.0–1.0 with risk bucket (low / medium / high / critical)
and recommended action.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Carrier damage rates (from historical analytics; refreshed periodically)
# Format: {carrier_scac: rate_pct}
_DEFAULT_CARRIER_RATES: Dict[str, float] = {
    "FXFE": 1.2,   # FedEx Freight
    "ODFL": 1.5,   # Old Dominion
    "SAIA": 2.1,
    "RDWY": 2.4,   # XPO / Yellow
    "ECHO": 1.8,
    "CEVA": 2.8,
    "UNKNOWN": 3.5,
}

# SKU fragility scores (1 = robust, 10 = extremely fragile)
_DEFAULT_SKU_FRAGILITY: Dict[str, float] = {
    "PHARMA": 9.0,
    "ELECTRONICS": 8.5,
    "GLASS": 9.5,
    "FROZEN_FOOD": 7.0,
    "DRY_GOODS": 2.5,
    "AUTOMOTIVE": 4.0,
    "GENERAL": 3.0,
}

# Risk thresholds → bucket mapping
RISK_THRESHOLDS = {
    "low":      (0.00, 0.35),
    "medium":   (0.35, 0.60),
    "high":     (0.60, 0.80),
    "critical": (0.80, 1.01),
}

# Action by risk bucket
RISK_ACTIONS = {
    "low":      "standard_putaway",
    "medium":   "enhanced_visual_check",
    "high":     "full_inspection_required",
    "critical": "quarantine_and_hold",
}

# Weights for composite scoring (must sum to 1.0)
_WEIGHTS = {
    "carrier":    0.25,
    "sku":        0.20,
    "transit":    0.20,
    "cold_chain": 0.20,
    "temporal":   0.05,
    "vendor":     0.10,
}


# ---------------------------------------------------------------------------
# Input / output models
# ---------------------------------------------------------------------------

@dataclass
class ShipmentRiskInput:
    """All available signals for a single shipment / pallet."""
    sscc: str
    carrier_scac: str
    sku_category: str
    vendor_id: str

    # Transit
    planned_transit_days: float
    actual_transit_days: float

    # Cold chain (from cold_chain.pipeline)
    cold_chain_excursion: bool = False
    cold_chain_excursion_duration_min: float = 0.0
    cold_chain_risk_level: str = "normal"   # "normal" | "elevated" | "critical"

    # Temporal
    receipt_hour: int = 10              # 0-23
    receipt_day_of_week: int = 1        # 0=Mon, 6=Sun

    # Vendor
    vendor_quality_score: float = 7.0   # 1 (poor) to 10 (excellent)

    # Optional manual overrides
    carrier_historical_damage_rate: Optional[float] = None
    sku_fragility_score: Optional[float] = None


@dataclass
class RiskScore:
    """Composite risk output for a shipment."""
    sscc: str
    composite_score: float                       # 0.0 – 1.0
    risk_bucket: str                             # low / medium / high / critical
    recommended_action: str
    component_scores: Dict[str, float] = field(default_factory=dict)
    risk_flags: List[str] = field(default_factory=list)
    scored_at: str = ""

    @property
    def requires_hold(self) -> bool:
        return self.risk_bucket in ("high", "critical")

    @property
    def requires_inspection(self) -> bool:
        return self.risk_bucket != "low"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class ShipmentRiskScorer:
    """
    Deterministic (rule-based + weighted average) risk scorer for inbound
    shipments.  No model weights needed; uses lookup tables that can be
    periodically refreshed from the analytics database.
    """

    def __init__(
        self,
        carrier_rates: Optional[Dict[str, float]] = None,
        sku_fragility: Optional[Dict[str, float]] = None,
    ):
        self._carrier_rates = carrier_rates or _DEFAULT_CARRIER_RATES
        self._sku_fragility = sku_fragility or _DEFAULT_SKU_FRAGILITY

    # ------------------------------------------------------------------
    # Component scorers (each returns 0.0–1.0)
    # ------------------------------------------------------------------

    def _carrier_score(self, inp: ShipmentRiskInput) -> tuple[float, List[str]]:
        flags: List[str] = []
        rate = inp.carrier_historical_damage_rate
        if rate is None:
            rate = self._carrier_rates.get(
                inp.carrier_scac.upper(),
                self._carrier_rates["UNKNOWN"],
            )
        # Normalise: 0% → 0.0, 5%+ → 1.0
        score = min(rate / 5.0, 1.0)
        if rate >= 3.0:
            flags.append(f"HIGH_CARRIER_DAMAGE_RATE:{rate:.1f}%")
        return score, flags

    def _sku_score(self, inp: ShipmentRiskInput) -> tuple[float, List[str]]:
        flags: List[str] = []
        fragility = inp.sku_fragility_score
        if fragility is None:
            fragility = self._sku_fragility.get(
                inp.sku_category.upper(),
                self._sku_fragility["GENERAL"],
            )
        score = (fragility - 1.0) / 9.0   # 1→0.0, 10→1.0
        if fragility >= 8.0:
            flags.append(f"HIGH_FRAGILITY_SKU:{inp.sku_category}")
        return score, flags

    def _transit_score(self, inp: ShipmentRiskInput) -> tuple[float, List[str]]:
        flags: List[str] = []
        if inp.planned_transit_days <= 0:
            return 0.3, []
        ratio = inp.actual_transit_days / inp.planned_transit_days
        # > 1.5× planned = high risk (product sat on dock / in transit too long)
        if ratio > 1.5:
            flags.append(f"EXTENDED_TRANSIT:{ratio:.1f}x_planned")
            score = min((ratio - 1.0) / 1.5, 1.0)
        elif ratio > 1.2:
            flags.append("DELAYED_TRANSIT")
            score = (ratio - 1.0) / 1.5
        else:
            score = max(0.0, (ratio - 0.8) / 0.4)
        return score, flags

    def _cold_chain_score(self, inp: ShipmentRiskInput) -> tuple[float, List[str]]:
        flags: List[str] = []
        level_map = {"normal": 0.0, "elevated": 0.5, "critical": 1.0}
        base = level_map.get(inp.cold_chain_risk_level, 0.0)

        if inp.cold_chain_excursion:
            flags.append(f"TEMP_EXCURSION:{inp.cold_chain_excursion_duration_min:.0f}min")
            # Add bonus based on duration
            bonus = min(inp.cold_chain_excursion_duration_min / 120.0, 0.4)
            base = min(base + bonus + 0.2, 1.0)

        return base, flags

    def _temporal_score(self, inp: ShipmentRiskInput) -> tuple[float, List[str]]:
        """
        Late-night / early-morning and weekend receipts correlate with higher
        damage due to reduced staffing and rushed handling.
        """
        flags: List[str] = []
        score = 0.0

        # Off-hours: midnight–6am or after 10pm
        if inp.receipt_hour < 6 or inp.receipt_hour >= 22:
            score += 0.4
            flags.append("OFF_HOURS_RECEIPT")
        elif inp.receipt_hour < 7 or inp.receipt_hour >= 20:
            score += 0.2

        # Weekend receipts
        if inp.receipt_day_of_week >= 5:
            score += 0.3
            flags.append("WEEKEND_RECEIPT")

        return min(score, 1.0), flags

    def _vendor_score(self, inp: ShipmentRiskInput) -> tuple[float, List[str]]:
        """Low vendor quality score → higher risk. Scale: 1(poor)→1.0, 10(excellent)→0.0"""
        flags: List[str] = []
        score = (10.0 - inp.vendor_quality_score) / 9.0
        if inp.vendor_quality_score <= 4.0:
            flags.append(f"LOW_VENDOR_SCORE:{inp.vendor_id}")
        return score, flags

    # ------------------------------------------------------------------
    # Composite scorer
    # ------------------------------------------------------------------

    def score(self, inp: ShipmentRiskInput) -> RiskScore:
        """Compute composite risk score for one shipment."""
        component_results = {
            "carrier":    self._carrier_score(inp),
            "sku":        self._sku_score(inp),
            "transit":    self._transit_score(inp),
            "cold_chain": self._cold_chain_score(inp),
            "temporal":   self._temporal_score(inp),
            "vendor":     self._vendor_score(inp),
        }

        component_scores: Dict[str, float] = {}
        all_flags: List[str] = []

        composite = 0.0
        for name, (score, flags) in component_results.items():
            component_scores[name] = round(score, 4)
            composite += _WEIGHTS[name] * score
            all_flags.extend(flags)

        composite = round(min(composite, 1.0), 4)

        # Determine bucket
        risk_bucket = "low"
        for bucket, (lo, hi) in RISK_THRESHOLDS.items():
            if lo <= composite < hi:
                risk_bucket = bucket
                break

        return RiskScore(
            sscc=inp.sscc,
            composite_score=composite,
            risk_bucket=risk_bucket,
            recommended_action=RISK_ACTIONS[risk_bucket],
            component_scores=component_scores,
            risk_flags=all_flags,
            scored_at=datetime.utcnow().isoformat() + "Z",
        )

    def score_batch(self, inputs: List[ShipmentRiskInput]) -> List[RiskScore]:
        """Score a batch; returns sorted highest risk first."""
        results = [self.score(inp) for inp in inputs]
        return sorted(results, key=lambda r: r.composite_score, reverse=True)
