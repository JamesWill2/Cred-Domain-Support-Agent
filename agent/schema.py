"""
Structured Output Response Schema for Cred Domain Support Agent.

Enforces strict Pydantic and JSON Schema validation on all agent responses.
No arbitrary dictionaries are permitted from the final agent layer.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
import uuid


class AgentResponse(BaseModel):
    """Strict structured response model for all agent turns."""

    model_config = ConfigDict(extra="forbid")

    response: str = Field(
        ...,
        description="The primary natural language response synthesized for the user.",
    )
    intent: str = Field(
        ...,
        description="Classified intent: 'policy_rag', 'loan_status', or 'refusal'.",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of parent knowledge-base document IDs referenced for groundedness.",
    )
    tool_used: Optional[str] = Field(
        default=None,
        description="Name of the tool executed, e.g. 'check_loan_application_status' or 'cred_kb_retriever'.",
    )
    escalation_score: Optional[float] = Field(
        default=None,
        description="Computed continuous escalation score in [0.0, 1.0] when a loan record is looked up.",
    )
    grounded: bool = Field(
        default=True,
        description="Whether the response is fully grounded in retrieved context or verified record data.",
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Explicit reason if the query was refused due to guardrails or unknown threshold fallback.",
    )
    trace_id: str = Field(
        default_factory=lambda: f"trace-{uuid.uuid4().hex[:12]}",
        description="Unique distributed tracing identifier for observability and JSONL logging.",
    )


def validate_agent_response(data: dict) -> AgentResponse:
    """Validate dictionary data against the strict AgentResponse Pydantic schema."""
    return AgentResponse.model_validate(data)

