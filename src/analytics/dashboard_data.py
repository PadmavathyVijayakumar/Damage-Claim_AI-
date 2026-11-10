"""
Dashboard data aggregator.

Builds the data payloads consumed by the analytics dashboards:
  - Executive summary: claim rates, auto-resolution %, cost recovery
  - Operations: live dock activity, cold-chain zone status
  - Claims: open claims by status, aging, liability split
  - Trends: 30/90-day damage rate trends by zone/SKU/carrier
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExecutiveSummary:
    total_pallets_inspected: int
    total_damage_events: int
    damage_rate_pct: float
    total_claims_opened: int
    claims_auto_resolved_pct: float
    cost_recovered_usd: float
    cost_avoided_usd: float    # pre-existing damage disputes won
    period_days: int


@dataclass
class ZoneStatus:
    zone_id: str
    current_temp_c: Optional[float]
    risk_level: str            # "normal" | "elevated" | "critical"
    last_excursion_hours_ago: Optional[float]
    damage_rate_7d: float


@dataclass
class ClaimSummary:
    claim_id: str
    sscc: str
    sku: str
    status: str
    liability: str
    amount_usd: Optional[float]
    opened_at: str
    days_open: int


@dataclass
class DashboardPayload:
    """Full payload for the analytics dashboard API."""

    generated_at: str
    executive: Optional[ExecutiveSummary] = None
    zone_statuses: list[ZoneStatus] = field(default_factory=list)
    open_claims: list[ClaimSummary] = field(default_factory=list)
    damage_trend_7d: list[dict] = field(default_factory=list)   # [{date, rate}]
    damage_trend_30d: list[dict] = field(default_factory=list)
    top_root_causes: list[dict] = field(default_factory=list)   # from RootCauseReport


class DashboardDataAggregator:
    """
    Builds dashboard payloads from event DataFrames and DB queries.

    Usage:
        agg = DashboardDataAggregator()
        payload = agg.build(
            events_df=load_events(days=30),
            claims_df=load_open_claims(),
            zone_statuses=get_live_zone_status(),
        )
    """

    def build(
        self,
        events_df: pd.DataFrame,
        claims_df: Optional[pd.DataFrame] = None,
        zone_statuses: Optional[list[dict]] = None,
        period_days: int = 30,
    ) -> DashboardPayload:
        """Build a complete dashboard payload."""
        now = datetime.utcnow().isoformat() + "Z"
        payload = DashboardPayload(generated_at=now)

        # ── Executive summary ─────────────────────────────────────────────
        if not events_df.empty:
            total = len(events_df)
            damaged = events_df["severity"].isin(["minor", "moderate", "severe"]).sum()
            claims_opened = int(events_df.get("trigger_claim", pd.Series(dtype=bool)).sum()) if "trigger_claim" in events_df.columns else 0
            auto_resolved = 0
            cost_recovered = 0.0
            cost_avoided = 0.0

            if claims_df is not None and not claims_df.empty:
                auto_resolved = int((claims_df.get("auto_approve", pd.Series(dtype=bool)) == True).sum())
                cost_recovered = float(claims_df.get("amount_usd", pd.Series(dtype=float)).fillna(0).sum())
                # cost avoided = disputed pre-existing claims
                disputed_won = claims_df[
                    (claims_df.get("liability", pd.Series(dtype=str)) == "supplier")
                    & (claims_df.get("status", pd.Series(dtype=str)) == "closed")
                ]
                cost_avoided = float(disputed_won.get("amount_usd", pd.Series(dtype=float)).fillna(0).sum())

            payload.executive = ExecutiveSummary(
                total_pallets_inspected=total,
                total_damage_events=int(damaged),
                damage_rate_pct=round(damaged / total * 100, 2) if total else 0.0,
                total_claims_opened=claims_opened,
                claims_auto_resolved_pct=round(auto_resolved / max(1, claims_opened) * 100, 1),
                cost_recovered_usd=round(cost_recovered, 2),
                cost_avoided_usd=round(cost_avoided, 2),
                period_days=period_days,
            )

        # ── Zone statuses ─────────────────────────────────────────────────
        if zone_statuses:
            for z in zone_statuses:
                payload.zone_statuses.append(ZoneStatus(
                    zone_id=z.get("zone_id", ""),
                    current_temp_c=z.get("current_temp_c"),
                    risk_level=z.get("risk_level", "normal"),
                    last_excursion_hours_ago=z.get("last_excursion_hours_ago"),
                    damage_rate_7d=z.get("damage_rate_7d", 0.0),
                ))

        # ── Open claims ───────────────────────────────────────────────────
        if claims_df is not None and not claims_df.empty:
            today = datetime.utcnow().date()
            for _, row in claims_df.iterrows():
                try:
                    opened = datetime.fromisoformat(str(row.get("opened_at", "")).replace("Z", "")).date()
                    days_open = (today - opened).days
                except Exception:
                    days_open = 0
                payload.open_claims.append(ClaimSummary(
                    claim_id=str(row.get("claim_id", "")),
                    sscc=str(row.get("sscc", "")),
                    sku=str(row.get("sku", "")),
                    status=str(row.get("status", "")),
                    liability=str(row.get("liability", "")),
                    amount_usd=row.get("amount_usd"),
                    opened_at=str(row.get("opened_at", "")),
                    days_open=days_open,
                ))

        # ── Damage trend (7d and 30d) ─────────────────────────────────────
        if not events_df.empty and "timestamp" in events_df.columns:
            payload.damage_trend_7d = self._daily_rate(events_df, days=7)
            payload.damage_trend_30d = self._daily_rate(events_df, days=30)

        return payload

    @staticmethod
    def _daily_rate(df: pd.DataFrame, days: int) -> list[dict]:
        """Compute daily damage rate for the last N days."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df["damaged"] = df["severity"].isin(["minor", "moderate", "severe"])

        cutoff = datetime.utcnow().date() - timedelta(days=days)
        df = df[df["date"] >= cutoff]

        grouped = df.groupby("date").agg(total=("damaged", "count"), damaged=("damaged", "sum"))
        grouped["rate"] = (grouped["damaged"] / grouped["total"] * 100).round(2)

        return [
            {"date": str(date), "rate": float(row["rate"])}
            for date, row in grouped.iterrows()
        ]
