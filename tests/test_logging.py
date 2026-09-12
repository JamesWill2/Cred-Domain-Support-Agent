"""Tests for Structured JSONL Logging and Zero Raw PII Guarantee."""
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.logger import get_jsonl_logger


def test_jsonl_logging_and_pii_masking():
    """Verify structured log entry is written for request and raw PII never appears on disk."""
    client = TestClient(app)
    logger = get_jsonl_logger()
    logger.clear_logs()

    test_pan = "BNZPA9999Z"
    test_aadhaar = "9988 7766 5544"
    test_account = "987654321098"

    query_with_pii = f"My PAN is {test_pan}, Aadhaar is {test_aadhaar}, and Account is {test_account}. What is the EMI rule?"

    res = client.post("/ask", json={"query": query_with_pii, "session_id": "pii-logging-session"})
    assert res.status_code == 200

    logs = logger.read_logs()
    assert len(logs) >= 1

    last_entry = logs[-1]
    assert "timestamp" in last_entry
    assert "trace_id" in last_entry
    assert "latency_ms" in last_entry
    assert "endpoint" in last_entry
    assert last_entry["endpoint"] == "/ask"
    assert last_entry["pii_masked"] is True

    # Critical Security Guarantee: Raw PII values must NEVER exist anywhere in the logged text
    all_logged_text = logger.log_path.read_text(encoding="utf-8")
    assert test_pan not in all_logged_text, f"Security Failure: Raw PAN {test_pan} found in log file!"
    assert test_aadhaar not in all_logged_text, f"Security Failure: Raw Aadhaar {test_aadhaar} found in log file!"
    assert test_account not in all_logged_text, f"Security Failure: Raw Account {test_account} found in log file!"

    # Assert masked placeholders are present
    assert "[MASKED_PAN]" in all_logged_text
    assert "[MASKED_AADHAAR]" in all_logged_text
    assert "[MASKED_ACCOUNT]" in all_logged_text

