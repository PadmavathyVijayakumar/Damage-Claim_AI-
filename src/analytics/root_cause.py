"""
Root cause analysis for recurring damage patterns.

Analyses damage events over a rolling window to identify:
  - Camera zones with disproportionate damage rates (dock equipment issues)
  - SKUs with high damage frequency (packaging failure)
  - Carrier SCACodes with elevated damage-on-delivery rates
  - Shifts / operators with deviation from baseline
  - Cold-chain zones with repeated excursion patterns

Output feeds the analytics dashboard and WMS/ERP reporting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Thresholds for flagging a category as a root cause candidate
DAMAGE_RATE_THRESHOLD = 0.05       # 5% damage rate triggers a flag
MIN_EVENTS_FOR_ANALYSIS = 20       # minimum events before drawing conclusions
EXCURSION_ZONE_THRESHOLD = 3       # 3+ excursions in 30 days → zone-level finding


@dataclass
class RootCauseFinding:
    """A single root cause finding from the analysis."""

    category: str          # "zone" | "sku" | "carrier" | "shift" | "cold_chain_zone"
    identifier: str        # zone ID, SKU code, SCAC, etc.
    damage_rate: float     # fraction of events with damage
    event_count: int
    description: str
    severity: str          # "low" | "medium" | "high"
    recommended_action: str


@dataclass
class RootCauseReport:
    """Full root cause analysis report."""

    analysis_window_days: int
    total_events: int
    findings: list[RootCauseFinding] = field(default_factory=list)
    generated_at: str = ""

    @property
    def high_severity_findings(self) -> list[RootCauseFinding]:
        return [f for f in self.findings if f.severity == "high"]


class RootCauseAnalyzer:
    """
    Analyses damage event DataFrames to surface root causes.

    Expected DataFrame columns:
        timestamp, camera_zone, sku, severity, trigger_claim,
        carrier_scac, shift_code, cold_chain_zone (optional)

    Usage:
        analyzer = RootCauseAnalyzer()
        df = load_events_from_db(days=30)
        report = analyzer.analyze(df, window_days=30)
    """

    def __init__(
        self,
        damage_rate_threshold: float = DAMAGE_RATE_THRESHOLD,
        min_events: int = MIN_EVENTS_FOR_ANALYSIS,
    ) -> None:
        self.damage_rate_threshold = damage_rate_threshold
        self.min_events = min_events

    def analyze(self, df: pd.DataFrame, window_days: int = 30) -> RootCauseReport:
        """
        Run full root cause analysis on event DataFrame.

        Args:
            df:           DataFrame of inspection/claim events.
            window_days:  Lookback window in days (for report metadata).

        Returns:
            RootCauseReport with all findings.
        """
        from datetime import datetime

        findings: list[RootCauseFinding] = []
        total = len(df)

        if total == 0:
            return RootCauseReport(
                analysis_window_days=window_days,
                total_events=0,
                generated_at=datetime.utcnow().isoformat() + "Z",
            )

        # Binary damage column
        df = df.copy()
        df["has_damage"] = df["severity"].isin(["minor", "moderate", "severe"])

        # ── Camera zone analysis ──────────────────────────────────────────
        if "camera_zone" in df.columns:
            findings.extend(self._analyze_category(df, "camera_zone", "zone"))

        # ── SKU analysis ──────────────────────────────────────────────────
        if "sku" in df.columns:
            findings.extend(self._analyze_category(df, "sku", "sku"))

        # ── Carrier SCAC analysis ─────────────────────────────────────────
        if "carrier_scac" in df.columns:
            findings.extend(self._analyze_category(df, "carrier_scac", "carrier"))

        # ── Shift analysis ────────────────────────────────────────────────
        if "shift_code" in df.columns:
            findings.extend(self._analyze_category(df, "shift_code", "shift"))

        # ── Cold chain zone analysis ──────────────────────────────────────
        if "cold_chain_zone" in df.columns and "excursion" in df.columns:
            findings.extend(self._analyze_excursions(df))

        # Sort by damage_rate descending
        findings.sort(key=lambda f: f.damage_rate, reverse=True)

        logger.info(
            "Root cause analysis complete events=%d findings=%d",
            total, len(findings),
        )
        return RootCauseReport(
            analysis_window_days=window_days,
            total_events=total,
            findings=findings,
            generated_at=datetime.utcnow().isoformat() + "Z",
        )

    def _analyze_category(
        self, df: pd.DataFrame, column: str, category: str
    ) -> list[RootCauseFinding]:
        findings = []
        grouped = df.groupby(column)["has_damage"].agg(["sum", "count"])

        for identifier, row in grouped.iterrows():
            count = int(row["count"])
            damaged = int(row["sum"])
            if count < self.min_events:
                continue
            rate = damaged / count
            if rate > self.damage_rate_threshold:
                severity = "high" if rate > 0.15 else ("medium" if rate > 0.08 else "low")
                action = self._recommend_action(category, rate, identifier)
                findings.append(RootCauseFinding(
                    category=category,
                    identifier=str(identifier),
                    damage_rate=round(rate, 4),
                    event_count=count,
                    description=f"{category.title()} '{identifier}' has {rate:.1%} damage rate "
                                f"({damaged}/{count} events).",
                    severity=severity,
                    recommended_action=action,
                ))
        return findings

    def _analyze_excursions(self, df: pd.DataFrame) -> list[RootCauseFinding]:
        findings = []
        excursion_df = df[df["excursion"] == True]
        grouped = excursion_df.groupby("cold_chain_zone").size()

        for zone_id, count in grouped.items():
            if count >= EXCURSION_ZONE_THRESHOLD:
                severity = "high" if count >= 10 else "medium"
                findings.append(RootCauseFinding(
                    category="cold_chain_zone",
                    identifier=str(zone_id),
                    damage_rate=count / max(1, len(df[df["cold_chain_zone"] == zone_id])),
                    event_count=count,
                    description=f"Zone '{zone_id}' had {count} temperature excursions in the analysis window.",
                    severity=severity,
                    recommended_action=f"Schedule maintenance inspection for zone {zone_id}. "
                                       "Review compressor service history and door seal integrity.",
                ))
        return findings

    @staticmethod
    def _recommend_action(category: str, rate: float, identifier) -> str:
        if category == "zone":
            return f"Inspect dock equipment at zone {identifier}. Check for fork damage patterns."
        if category == "sku":
            return f"Review packaging spec for SKU {identifier}. Consider reinforced packaging."
        if category == "carrier":
            return f"Issue carrier scorecard to {identifier}. Review carrier contract SLA."
        if category == "shift":
            return f"Conduct refresher training for shift {identifier} on damage prevention."
        return "Investigate further."
