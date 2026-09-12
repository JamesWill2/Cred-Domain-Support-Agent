"""
LangGraph Agent State Schema for Cred Domain Support Agent.
"""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    """Unified state flowing across the LangGraph nodes."""

    session_id: str
    trace_id: str
    raw_input: str
    sanitized_input: str
    pii_metadata: Dict[str, Any]
    is_injection: bool
    injection_reason: Optional[str]
    intent: str
    target_loan_id: Optional[str]
    retrieved_chunks: List[Dict[str, Any]]
    loan_record_result: Optional[Dict[str, Any]]
    synthesized_response: str
    sources: List[str]
    tool_used: Optional[str]
    escalation_score: Optional[float]
    grounded: bool
    refusal_reason: Optional[str]
    final_output: Optional[Dict[str, Any]]

