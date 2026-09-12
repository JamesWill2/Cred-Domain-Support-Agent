"""
Pydantic Request and Response Models for FastAPI Endpoints.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from agent.schema import AgentResponse


class AskRequest(BaseModel):
    """Request schema for POST /ask."""

    query: str = Field(
        ...,
        description="The customer or support staff inquiry.",
        examples=["What are the KYC document requirements?"],
    )
    session_id: Optional[str] = Field(
        default="default-session",
        description="Conversation session ID for multi-turn memory.",
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="Optional custom distributed tracing ID.",
    )


class AddDocumentRequest(BaseModel):
    """Request schema for POST /add-document."""

    doc_id: str = Field(
        ...,
        description="Unique identifier for the new document, e.g. 'doc_13_cybersecurity'.",
    )
    title: str = Field(
        ...,
        description="Descriptive title of the policy clause.",
    )
    topic: str = Field(
        ...,
        description="Category topic of the policy.",
    )
    content: str = Field(
        ...,
        description="Full text content (2-5 sentences) of the policy clause.",
    )


class AddDocumentResponse(BaseModel):
    """Response schema for POST /add-document."""

    success: bool
    doc_id: str
    fixed_chunks_added: int
    sentence_chunks_added: int
    total_fixed_count: int
    total_sentence_count: int
    message: str


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    app_name: str
    version: str
    total_loan_records: int
    fixed_chunks_count: int
    sentence_chunks_count: int


class SimpleResponse(BaseModel):
    """Compact response containing only the answer string."""
    answer: str = Field(
        ..., description="The primary natural language answer for the user.")
