"""
Loan Application Lookup Tool and Escalation Score Engine.

Implements check_loan_application_status with a continuous, multi-factor
mathematical escalation formula based on fraud review flags and recency.
"""

from typing import Dict, Any, Optional
from dataset.dataset import get_loan_record, LOAN_APPLICATIONS

# Escalation threshold calibrated to the 85th percentile of the risk distribution
ESCALATION_THRESHOLD: float = 0.65


def calculate_escalation_score(
    flagged_for_fraud_review: bool,
    days_since_created: int,
    max_days: int = 30,
) -> float:
    """
    Compute continuous numerical escalation score in [0.0, 1.0].

    Mathematical Formula:
        recency_signal = days_since_created / max_days
        escalation_score = (0.60 * flagged_for_fraud_review) + (0.40 * recency_signal)

    Weights:
        - 0.60 (60%): Fraud review flag (primary institutional risk driver)
        - 0.40 (40%): Normalized application age (operational SLA aging driver)

    Bounds:
        Minimum: 0.00 (Unflagged, created today)
        Maximum: 1.00 (Fraud flagged, 30 days pending)
    """
    clamped_days = max(0, min(days_since_created, max_days))
    recency_signal = clamped_days / float(max_days)
    fraud_signal = 1.0 if flagged_for_fraud_review else 0.0

    score = (0.60 * fraud_signal) + (0.40 * recency_signal)
    return round(score, 4)


def check_loan_application_status(record_id: str) -> Dict[str, Any]:
    """
    Look up a loan application record by ID and evaluate its escalation score.

    Args:
        record_id: The unique identifier of the loan (e.g. 'LOAN-1001').

    Returns:
        Dictionary containing record details, escalation score, and recommendation.
    """
    cleaned_id = record_id.strip().upper()
    record = get_loan_record(cleaned_id)

    if not record:
        return {
            "found": False,
            "record_id": cleaned_id,
            "status": "Not Found",
            "loan_amount_inr": None,
            "escalation_score": None,
            "escalation_required": False,
            "message": f"Loan application '{cleaned_id}' was not found in Cred's records.",
        }

    fraud = record["flagged_for_fraud_review"]
    days = record["days_since_created"]
    escalation_score = calculate_escalation_score(fraud, days)
    escalate = escalation_score >= ESCALATION_THRESHOLD

    return {
        "found": True,
        "record_id": record["record_id"],
        "category": record["category"],
        "status": record["status"],
        "loan_amount_inr": record["loan_amount_inr"],
        "days_since_created": days,
        "flagged_for_fraud_review": fraud,
        "escalation_score": escalation_score,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "escalation_required": escalate,
        "message": (
            f"Loan application {record['record_id']} ({record['category']}) is currently "
            f"'{record['status']}' with principal amount INR {record['loan_amount_inr']:,}. "
            f"Escalation score is {escalation_score:.4f} "
            f"({'ESCALATE: Exceeds threshold ' + str(ESCALATION_THRESHOLD) if escalate else 'NORMAL: Within acceptable thresholds'})."
        ),
    }


def get_escalation_distribution_summary() -> Dict[str, Any]:
    """Calculate empirical distribution of escalation scores across the dataset."""
    scores = []
    for r in LOAN_APPLICATIONS:
        s = calculate_escalation_score(r["flagged_for_fraud_review"], r["days_since_created"])
        scores.append(s)

    scores.sort()
    n = len(scores)
    p50 = scores[int(0.50 * n)]
    p80 = scores[int(0.80 * n)]
    p85 = scores[int(0.85 * n)]
    p90 = scores[int(0.90 * n)]
    escalated_count = sum(1 for s in scores if s >= ESCALATION_THRESHOLD)

    return {
        "min_score": min(scores),
        "max_score": max(scores),
        "p50_median": p50,
        "p80": p80,
        "p85": p85,
        "p90": p90,
        "threshold": ESCALATION_THRESHOLD,
        "escalated_count": escalated_count,
        "total_records": n,
        "escalation_rate_pct": round((escalated_count / n) * 100, 2),
    }


if __name__ == "__main__":
    summary = get_escalation_distribution_summary()
    print("Escalation Score Distribution Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    print("\nSample Lookups:")
    print(check_loan_application_status("LOAN-1001"))
    print(check_loan_application_status("LOAN-9999"))

