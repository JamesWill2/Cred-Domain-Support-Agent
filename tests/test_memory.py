"""Tests for Persistent Multi-Turn Conversation Memory."""
import pytest
from agent.memory import get_memory_manager
from agent.graph import run_cred_agent


def test_multiturn_state_retention():
    """Verify session memory carries state across turns, resolving 'that loan'."""
    mem = get_memory_manager()
    session_id = "test-multiturn-session"
    mem.clear_session(session_id)

    # Turn 1: Inquire about LOAN-1001
    turn1_res = run_cred_agent(
        user_input="Please check the status of my loan LOAN-1001",
        session_id=session_id,
    )
    assert turn1_res["intent"] == "loan_status"
    assert "LOAN-1001" in turn1_res["sources"]
    assert mem.get_last_loan_id(session_id) == "LOAN-1001"

    # Turn 2: Follow-up referring to 'that loan'
    turn2_res = run_cred_agent(
        user_input="What is the escalation score of that loan?",
        session_id=session_id,
    )
    assert turn2_res["intent"] == "loan_status"
    assert "LOAN-1001" in turn2_res["sources"]
    assert turn2_res["escalation_score"] is not None


def test_fresh_conversation_reset():
    """Verify a fresh session has no prior loan context and prompts user for ID."""
    fresh_session = "fresh-session-isolated"
    mem = get_memory_manager()
    mem.clear_session(fresh_session)

    assert mem.get_last_loan_id(fresh_session) is None

    # Asking about 'that loan' in a fresh session cannot resolve coreference
    res = run_cred_agent(
        user_input="What is the status of that loan?",
        session_id=fresh_session,
    )
    assert res["intent"] == "loan_status"
    assert "no active loan ID was found in our current conversation history" in res["response"]

