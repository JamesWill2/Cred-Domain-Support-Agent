"""Tests for LOAN_APPLICATIONS Dataset Generator and Validator."""
import pytest
from dataset.dataset import (
    LOAN_APPLICATIONS,
    CATEGORIES,
    STATUSES,
    SEED,
    generate_dataset,
    validate_dataset,
    get_loan_record,
)


def test_dataset_structure():
    """Verify total records >= 40, presence of all required fields, and unique IDs."""
    assert len(LOAN_APPLICATIONS) >= 40
    assert len(LOAN_APPLICATIONS) == 45

    record_ids = set()
    required_fields = {
        "record_id",
        "category",
        "status",
        "loan_amount_inr",
        "days_since_created",
        "flagged_for_fraud_review",
    }

    for r in LOAN_APPLICATIONS:
        assert required_fields.issubset(r.keys())
        assert r["record_id"] not in record_ids
        record_ids.add(r["record_id"])
        assert isinstance(r["days_since_created"], int)
        assert 0 <= r["days_since_created"] <= 30
        assert isinstance(r["loan_amount_inr"], (int, float))
        assert r["loan_amount_inr"] > 0
        assert isinstance(r["flagged_for_fraud_review"], bool)


def test_dataset_thresholds():
    """Verify category counts >= 3, status counts >= 1, and fraud rate 10%-30%."""
    stats = validate_dataset(LOAN_APPLICATIONS)

    for cat in CATEGORIES:
        assert stats["category_counts"][cat] >= 3, f"Category {cat} below threshold 3"

    for stat in STATUSES:
        assert stats["status_counts"][stat] >= 1, f"Status {stat} below threshold 1"

    fraud_pct = stats["fraud_percentage"]
    assert 10.0 <= fraud_pct <= 30.0, f"Fraud rate {fraud_pct}% outside [10%, 30%]"


def test_dataset_determinism():
    """Verify regenerating with SEED produces identical records."""
    dataset_1 = generate_dataset(num_records=45, seed=SEED)
    dataset_2 = generate_dataset(num_records=45, seed=SEED)
    assert dataset_1 == dataset_2


def test_get_loan_record():
    """Test lookup helper."""
    rec = get_loan_record("LOAN-1001")
    assert rec is not None
    assert rec["record_id"] == "LOAN-1001"

    non_existent = get_loan_record("LOAN-9999")
    assert non_existent is None

