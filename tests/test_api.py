"""Tests for FastAPI Deployment and Endpoints."""
import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_fastapi_health(client):
    """Verify GET /health returns 200 and valid system status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["total_loan_records"] >= 40
    assert data["fixed_chunks_count"] > 0
    assert data["sentence_chunks_count"] > 0


def test_fastapi_ask_endpoint(client):
    """Verify POST /ask endpoint processes query and returns structured AgentResponse."""
    payload = {
        "query": "Are there any prepayment penalties or foreclosure charges on retail loans?",
        "session_id": "api-test-session",
    }
    res = client.post("/ask", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "policy_rag"
    assert "prepayment" in data["response"].lower()
    assert data["grounded"] is True
    assert len(data["trace_id"]) > 0


def test_fastapi_add_document(client):
    """Verify POST /add-document indexes new content dynamically."""
    payload = {
        "doc_id": "doc_test_dynamic",
        "title": "Dynamic Test Policy",
        "topic": "Testing topic",
        "content": "This is a dynamic test policy for automated endpoint verification. It specifies terms and conditions for testing.",
    }
    res = client.post("/add-document", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["doc_id"] == "doc_test_dynamic"
    assert data["fixed_chunks_added"] > 0
    assert data["sentence_chunks_added"] > 0
