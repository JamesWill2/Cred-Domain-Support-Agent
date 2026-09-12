"""
Deterministic Loan Applications Dataset Generator for Cred Domain Support Agent.

This module deterministically generates and validates the LOAN_APPLICATIONS dataset
meeting all structural, category, status, and fraud rate constraints specified
in the Cred Capstone Brief.
"""

from typing import Dict, List, Any, Optional
import random

# Documented deterministic seed ensuring strict reproducibility
SEED: int = 42

CATEGORIES: List[str] = [
    "Personal Loan",
    "Home Loan",
    "Auto Loan",
    "Education Loan",
    "Business Loan",
]

STATUSES: List[str] = [
    "Submitted",
    "Under Review",
    "Approved",
    "Rejected",
    "Disbursed",
]

# Realistic loan amount boundaries (in INR) per category reflecting Indian banking norms
CATEGORY_AMOUNT_RANGES: Dict[str, tuple[int, int]] = {
    "Personal Loan": (50_000, 1_500_000),      # Unsecured retail credit
    "Home Loan": (1_500_000, 15_000_000),     # Long-term secured property mortgages
    "Auto Loan": (300_000, 2_500_000),        # Secured passenger & commercial vehicles
    "Education Loan": (200_000, 3_500_000),   # Tier 1 domestic & overseas university tuition
    "Business Loan": (500_000, 10_000_000),   # MSME working capital & equipment financing
}


def generate_dataset(
    num_records: int = 45,
    seed: int = SEED,
    fraud_probability: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Generate a deterministic list of loan applications.

    Args:
        num_records: Total number of records to generate (must be >= 40).
        seed: Random seed for reproducibility.
        fraud_probability: Probability threshold for fraud review flagging.

    Returns:
        List of validated loan application records.
    """
    if num_records < 40:
        raise ValueError("Dataset generation requires at least 40 records.")

    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []

    for i in range(1, num_records + 1):
        record_id = f"LOAN-{1000 + i}"
        category = rng.choice(CATEGORIES)
        status = rng.choice(STATUSES)

        min_amt, max_amt = CATEGORY_AMOUNT_RANGES[category]
        # Round loan amount to nearest ₹1,000 for realistic institutional reporting
        loan_amount_inr = round(rng.randint(min_amt, max_amt) / 1000) * 1000

        days_since_created = rng.randint(0, 30)
        flagged_for_fraud_review = rng.random() < fraud_probability

        records.append(
            {
                "record_id": record_id,
                "category": category,
                "status": status,
                "loan_amount_inr": loan_amount_inr,
                "days_since_created": days_since_created,
                "flagged_for_fraud_review": flagged_for_fraud_review,
            }
        )

    # Validate generated records against acceptance criteria
    validate_dataset(records)
    return records


def validate_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate that the dataset satisfies all structural and business constraints:
    1. Total records >= 40
    2. Every required category occurs >= 3 times
    3. Every required status occurs >= 1 time
    4. days_since_created is integer in [0, 30]
    5. flagged_for_fraud_review percentage is strictly between 10% and 30%
    6. All required keys are present and data types correct

    Returns:
        Dictionary of computed validation statistics.
    """
    total = len(records)
    if total < 40:
        raise AssertionError(f"Expected >= 40 records, found {total}")

    category_counts: Dict[str, int] = {cat: 0 for cat in CATEGORIES}
    status_counts: Dict[str, int] = {stat: 0 for stat in STATUSES}
    fraud_count = 0
    record_ids = set()

    required_keys = {
        "record_id",
        "category",
        "status",
        "loan_amount_inr",
        "days_since_created",
        "flagged_for_fraud_review",
    }

    for r in records:
        # Check keys
        missing = required_keys - set(r.keys())
        if missing:
            raise AssertionError(f"Record {r.get('record_id')} missing keys: {missing}")

        # Check unique id
        rid = r["record_id"]
        if rid in record_ids:
            raise AssertionError(f"Duplicate record_id: {rid}")
        record_ids.add(rid)

        # Check category
        cat = r["category"]
        if cat not in category_counts:
            category_counts[cat] = 0
        category_counts[cat] += 1

        # Check status
        stat = r["status"]
        if stat not in status_counts:
            status_counts[stat] = 0
        status_counts[stat] += 1

        # Check days_since_created
        days = r["days_since_created"]
        if not isinstance(days, int) or not (0 <= days <= 30):
            raise AssertionError(f"Invalid days_since_created: {days} in record {rid}")

        # Check loan amount
        amt = r["loan_amount_inr"]
        if not isinstance(amt, (int, float)) or amt <= 0:
            raise AssertionError(f"Invalid loan_amount_inr: {amt} in record {rid}")

        # Check fraud flag
        fraud = r["flagged_for_fraud_review"]
        if not isinstance(fraud, bool):
            raise AssertionError(f"flagged_for_fraud_review must be boolean in {rid}")
        if fraud:
            fraud_count += 1

    # Check category counts >= 3
    for cat in CATEGORIES:
        count = category_counts.get(cat, 0)
        if count < 3:
            raise AssertionError(f"Category '{cat}' count {count} is below threshold of 3")

    # Check status counts >= 1
    for stat in STATUSES:
        count = status_counts.get(stat, 0)
        if count < 1:
            raise AssertionError(f"Status '{stat}' count {count} is below threshold of 1")

    # Check fraud percentage between 10% and 30%
    fraud_pct = (fraud_count / total) * 100.0
    if not (10.0 <= fraud_pct <= 30.0):
        raise AssertionError(
            f"Fraud review percentage {fraud_pct:.2f}% outside required range [10.0%, 30.0%]"
        )

    return {
        "total_records": total,
        "category_counts": category_counts,
        "status_counts": status_counts,
        "fraud_count": fraud_count,
        "fraud_percentage": fraud_pct,
    }


def report_dataset(records: Optional[List[Dict[str, Any]]] = None) -> str:
    """Produce a human-readable summary of the dataset."""
    if records is None:
        records = LOAN_APPLICATIONS

    stats = validate_dataset(records)
    lines = [
        "==================================================",
        "  LOAN APPLICATIONS DATASET REPORT (Cred FinTech) ",
        "==================================================",
        f"Total Records Generated: {stats['total_records']}",
        f"Random Seed Documented:  {SEED}",
        "",
        "--- CATEGORY COUNTS (Constraint: >= 3 each) ---",
    ]
    for cat, count in stats["category_counts"].items():
        lines.append(f"  * {cat:<18}: {count:>2} records")

    lines.append("")
    lines.append("--- STATUS COUNTS (Constraint: >= 1 each) ---")
    for stat, count in stats["status_counts"].items():
        lines.append(f"  * {stat:<18}: {count:>2} records")

    lines.append("")
    lines.append("--- FRAUD REVIEW METRICS (Constraint: 10% - 30%) ---")
    lines.append(f"  * Flagged Records: {stats['fraud_count']} / {stats['total_records']}")
    lines.append(f"  * Fraud Percentage: {stats['fraud_percentage']:.2f}% [PASS]")
    lines.append("==================================================")
    return "\n".join(lines)


# Singleton dataset instance initialized deterministically on load
LOAN_APPLICATIONS: List[Dict[str, Any]] = generate_dataset(num_records=45, seed=SEED)
_LOOKUP_MAP: Dict[str, Dict[str, Any]] = {r["record_id"]: r for r in LOAN_APPLICATIONS}


def get_loan_record(record_id: str) -> Optional[Dict[str, Any]]:
    """Look up a loan application record by its unique ID."""
    return _LOOKUP_MAP.get(record_id.strip().upper())


if __name__ == "__main__":
    print(report_dataset())

