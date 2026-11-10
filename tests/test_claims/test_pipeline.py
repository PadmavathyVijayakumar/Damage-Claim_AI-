"""Tests for the claims intelligence pipeline."""
import pytest

from src.claims.classifier import ClaimClassifier, ClaimValidity, LiabilityParty
from src.claims.inconsistency import InconsistencyDetector
from src.evidence.comparator import ConditionComparator, DamageOrigin


class TestConditionComparator:
    def setup_method(self):
        self.comp = ConditionComparator(use_clip=False)

    def test_pre_existing_damage_detected(self):
        result = self.comp.compare("SSCC001", "minor", "moderate")
        assert result.damage_origin == DamageOrigin.PRE_EXISTING.value
        assert result.should_reject_claim

    def test_warehouse_damage_detected(self):
        result = self.comp.compare("SSCC002", "none", "severe")
        assert result.damage_origin == DamageOrigin.WAREHOUSE.value
        assert not result.should_reject_claim

    def test_carrier_responsibility(self):
        result = self.comp.compare("SSCC003", "none", "moderate", carrier_reported=True)
        assert result.damage_origin == DamageOrigin.CARRIER.value
        assert result.carrier_responsible

    def test_no_change_returns_unknown(self):
        result = self.comp.compare("SSCC004", "none", "none")
        assert result.damage_origin == DamageOrigin.UNKNOWN.value


class TestClaimClassifier:
    def setup_method(self):
        self.clf = ClaimClassifier()

    def test_pre_existing_produces_invalid_claim(self):
        result = self.clf.classify(
            claim_id="CLM-001",
            damage_origin="pre_existing",
            inbound_severity="minor",
            outbound_severity="minor",
        )
        assert result.validity == ClaimValidity.INVALID.value
        assert result.liability == LiabilityParty.SUPPLIER.value

    def test_warehouse_damage_produces_valid_claim(self):
        result = self.clf.classify(
            claim_id="CLM-002",
            damage_origin="warehouse",
            inbound_severity="none",
            outbound_severity="severe",
            claim_amount=500.0,
        )
        assert result.validity == ClaimValidity.VALID.value
        assert result.liability == LiabilityParty.WAREHOUSE.value

    def test_low_amount_valid_claim_auto_approves(self):
        result = self.clf.classify(
            claim_id="CLM-003",
            damage_origin="warehouse",
            inbound_severity="none",
            outbound_severity="minor",
            claim_amount=50.0,  # < $100
        )
        assert result.auto_approve is True

    def test_critical_amount_escalates(self):
        result = self.clf.classify(
            claim_id="CLM-004",
            damage_origin="warehouse",
            inbound_severity="none",
            outbound_severity="severe",
            claim_amount=15_000.0,  # > $10,000
        )
        assert result.escalate_to_human is True


class TestInconsistencyDetector:
    def setup_method(self):
        self.det = InconsistencyDetector()

    def test_severity_inflation_detected(self):
        report = self.det.check(
            claim_id="CLM-100",
            sscc="SSCC001",
            warehouse_severity="minor",
            claimant_severity="severe",
        )
        codes = [i.code for i in report.inconsistencies]
        assert "SEVERITY_INFLATION" in codes
        assert report.recommendation == "flag_for_fraud_review"

    def test_sscc_mismatch_detected(self):
        report = self.det.check(
            claim_id="CLM-101",
            sscc="SSCC001",
            warehouse_severity="minor",
            claimant_severity="minor",
            claimant_sscc_in_image="SSCC999",
        )
        codes = [i.code for i in report.inconsistencies]
        assert "SSCC_MISMATCH" in codes

    def test_duplicate_image_detected(self):
        seen = {"abc123"}
        report = self.det.check(
            claim_id="CLM-102",
            sscc="SSCC001",
            warehouse_severity="minor",
            claimant_severity="minor",
            claimant_image_hash="abc123",
            seen_hashes=seen,
        )
        codes = [i.code for i in report.inconsistencies]
        assert "DUPLICATE_IMAGE" in codes

    def test_clean_submission_has_no_issues(self):
        report = self.det.check(
            claim_id="CLM-103",
            sscc="SSCC001",
            warehouse_severity="moderate",
            claimant_severity="moderate",
        )
        assert len(report.inconsistencies) == 0
        assert report.recommendation == "no_issues_found"
