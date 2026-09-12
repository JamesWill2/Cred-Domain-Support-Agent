"""
Knowledge Base Document Loader and Validator for Cred Domain Support Agent.

Validates that all >= 12 required banking topics are present and that each
document adheres strictly to the 2-5 sentence constraint.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import re

REQUIRED_TOPICS: List[str] = [
    "Loan eligibility criteria by loan type",
    "EMI calculation rules",
    "Credit-card fee structure",
    "KYC document requirements",
    "Fraud-dispute resolution process",
    "Account-closure process",
    "Interest-rate slabs",
    "Prepayment-penalty rules",
    "Minimum-balance requirements",
    "Credit-score impact factors",
    "Joint-account rules",
    "NRI-account eligibility",
]

DOCS_DIR = Path(__file__).resolve().parent / "docs"

# Mapping between doc filename and required topic
DOC_TOPIC_MAP: Dict[str, str] = {
    "doc_01_loan_eligibility.md": "Loan eligibility criteria by loan type",
    "doc_02_emi_calculation.md": "EMI calculation rules",
    "doc_03_credit_card_fees.md": "Credit-card fee structure",
    "doc_04_kyc_requirements.md": "KYC document requirements",
    "doc_05_fraud_dispute.md": "Fraud-dispute resolution process",
    "doc_06_account_closure.md": "Account-closure process",
    "doc_07_interest_rate_slabs.md": "Interest-rate slabs",
    "doc_08_prepayment_penalties.md": "Prepayment-penalty rules",
    "doc_09_minimum_balance.md": "Minimum-balance requirements",
    "doc_10_credit_score_factors.md": "Credit-score impact factors",
    "doc_11_joint_account_rules.md": "Joint-account rules",
    "doc_12_nri_account_rules.md": "NRI-account eligibility",
}


def count_sentences(text: str) -> int:
    """Count clean sentences in text by splitting on terminal punctuation."""
    # Remove markdown headers and newlines
    body = re.sub(r"^#.*$", "", text, flags=re.MULTILINE).strip()
    # Split on sentence boundaries: period, exclamation, question mark followed by space or end
    sentences = [s.strip() for s in re.split(r"[.!?]+(?:\s+|$)", body) if s.strip()]
    return len(sentences)


def load_knowledge_base(docs_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load and validate all knowledge base documents.

    Returns:
        List of document dictionaries containing doc_id, topic, title, content, sentence_count.
    """
    target_dir = docs_dir or DOCS_DIR
    if not target_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found at {target_dir}")

    files = sorted(target_dir.glob("*.md"))
    if len(files) < 12:
        raise AssertionError(f"Expected at least 12 KB documents, found {len(files)}")

    documents: List[Dict[str, Any]] = []
    covered_topics = set()

    for file_path in files:
        raw_text = file_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        
        # Extract title from first markdown header
        title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else file_path.stem
        # Body text without markdown header
        body = "\n".join(line for line in lines if not line.startswith("#")).strip()
        
        # Sentence count validation (2-5 sentences constraint)
        sentence_cnt = count_sentences(raw_text)
        if not (2 <= sentence_cnt <= 5):
            raise AssertionError(
                f"Document {file_path.name} violates sentence constraint: has {sentence_cnt} sentences (must be 2-5)"
            )

        topic = DOC_TOPIC_MAP.get(file_path.name, title)
        covered_topics.add(topic)
        doc_id = file_path.stem

        documents.append(
            {
                "doc_id": doc_id,
                "title": title,
                "topic": topic,
                "content": body,
                "raw_text": raw_text,
                "sentence_count": sentence_cnt,
                "file_path": str(file_path),
            }
        )

    # Verify all 12 required topics are covered
    missing_topics = set(REQUIRED_TOPICS) - covered_topics
    if missing_topics:
        raise AssertionError(f"Missing required KB topics: {missing_topics}")

    return documents


def get_kb_document_by_id(doc_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve single KB document by doc_id."""
    for doc in load_knowledge_base():
        if doc["doc_id"] == doc_id:
            return doc
    return None


def list_kb_topics() -> List[str]:
    """List all registered topics in KB."""
    return [d["topic"] for d in load_knowledge_base()]


if __name__ == "__main__":
    docs = load_knowledge_base()
    print(f"Successfully loaded and validated {len(docs)} documents.")
    for d in docs:
        print(f"[{d['doc_id']}] ({d['sentence_count']} sentences) Topic: {d['topic']}")

