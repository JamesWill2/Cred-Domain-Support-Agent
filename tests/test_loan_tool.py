"""Tests for Loan Status Tool and Escalation Score Math."""
import pytest
from agent.tools import (
    check_loan_application_status,
    calculate_escalation_score,
    get_escalation_distribution_summary,
    ESCALATION_THRESHOLD,
)


def test_valid_record_lookup():
    """Verify lookup of valid record returns status, amount, and score."""
    res = check_loan_application_status("LOAN-1001")
    assert res["found"] is True
    assert res["record_id"] == "LOAN-1001"
    assert res["status"] in ["Submitted", "Under Review", "Approved", "Rejected", "Disbursed"]
    assert res["loan_amount_inr"] > 0
    assert 0.0 <= res["escalation_score"] <= 1.0
    assert isinstance(res["escalation_required"], bool)


def test_invalid_record_lookup():
    """Verify lookup of invalid record returns found=False and appropriate message."""
    res = check_loan_application_status("LOAN-9999")
    assert res["found"] is False
    assert res["status"] == "Not Found"
    assert res["escalation_score"] is None
    assert "not found" in res["message"].lower()


def test_escalation_score_math():
    """Verify mathematical formula bounds and weights."""
    # Min score: fraud=False, days=0 -> 0.0
    assert calculate_escalation_score(False, 0) == 0.0

    # Max score: fraud=True, days=30 -> 1.0
    assert calculate_escalation_score(True, 30) == 1.0

    # Intermediate checks:
    # fraud=False, days=15 -> 0.6*0 + 0.4*(15/30) = 0.20
    assert calculate_escalation_score(False, 15) == 0.20

    # fraud=True, days=0 -> 0.6*1 + 0.4*0 = 0.60
    assert calculate_escalation_score(True, 0) == 0.60

    # Clamping behavior: negative days clamped to 0, excessive days clamped to 30
    assert calculate_escalation_score(False, -5) == 0.0
    assert calculate_escalation_score(True, 50) == 1.0


def test_escalation_threshold_classification():
    """Verify threshold cleanly distinguishes low risk from high risk applications."""
    summary = get_escalation_distribution_summary()
    assert summary["min_score"] >= 0.0
    assert summary["max_score"] <= 1.0
    assert summary["threshold"] == ESCALATION_THRESHOLD
    assert summary["escalated_count"] >= 1

