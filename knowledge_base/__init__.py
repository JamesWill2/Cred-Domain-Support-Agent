"""Knowledge base package for Cred Domain Support Agent."""
from knowledge_base.loader import (
    load_knowledge_base,
    get_kb_document_by_id,
    list_kb_topics,
    REQUIRED_TOPICS,
)

__all__ = [
    "load_knowledge_base",
    "get_kb_document_by_id",
    "list_kb_topics",
    "REQUIRED_TOPICS",
]

