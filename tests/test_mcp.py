"""Tests for FastMCP Protocol Client-Server Round Trip."""
import pytest
import asyncio
from mcp_service.client import run_mcp_client


@pytest.mark.asyncio
async def test_mcp_client_roundtrip():
    """Verify FastMCP client connects to server and executes tool calls across >= 2 record IDs."""
    test_ids = ["LOAN-1001", "LOAN-1002"]
    results = await run_mcp_client(test_ids)

    assert len(results) >= 2
    for r in results:
        assert r["is_error"] is False
        assert r["record_id"] in test_ids
        structured = r["structured_data"]
        assert structured["found"] is True
        assert structured["record_id"] == r["record_id"]
        assert "status" in structured
        assert "loan_amount_inr" in structured
        assert "escalation_score" in structured

