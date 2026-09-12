"""
Chunking Strategies for Cred Domain Knowledge Base.

Implements two independent chunking strategies:
A. Fixed-size chunks with character overlap.
B. Sentence-based chunks splitting on linguistic sentence boundaries.
"""

from typing import Dict, List, Any
import re
from knowledge_base.loader import load_knowledge_base


def chunk_documents_fixed(
    documents: List[Dict[str, Any]],
    chunk_size: int = 180,
    chunk_overlap: int = 40,
) -> List[Dict[str, Any]]:
    """
    Strategy A: Fixed-size chunking with overlap.

    Args:
        documents: List of KB documents.
        chunk_size: Size of chunk in characters.
        chunk_overlap: Overlapping character count.

    Returns:
        List of chunk dictionaries.
    """
    chunks: List[Dict[str, Any]] = []

    for doc in documents:
        text = doc["content"]
        doc_id = doc["doc_id"]
        topic = doc["topic"]
        title = doc["title"]

        start = 0
        idx = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = f"{doc_id}_fixed_{idx:02d}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "parent_doc_id": doc_id,
                        "text": chunk_text,
                        "strategy": "fixed",
                        "metadata": {
                            "parent_doc_id": doc_id,
                            "topic": topic,
                            "title": title,
                            "strategy": "fixed",
                            "chunk_index": idx,
                            "char_start": start,
                            "char_end": end,
                        },
                    }
                )
                idx += 1
            if end >= text_len:
                break
            start += chunk_size - chunk_overlap

    return chunks


def chunk_documents_sentence(
    documents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Strategy B: Sentence-based chunking.
    Splits text on sentence boundaries so each chunk represents an atomic policy statement.

    Args:
        documents: List of KB documents.

    Returns:
        List of chunk dictionaries.
    """
    chunks: List[Dict[str, Any]] = []

    for doc in documents:
        text = doc["content"]
        doc_id = doc["doc_id"]
        topic = doc["topic"]
        title = doc["title"]

        # Extract sentences by terminal punctuation (.!?)
        raw_sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()
        ]

        for idx, sentence in enumerate(raw_sentences):
            chunk_id = f"{doc_id}_sent_{idx:02d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "parent_doc_id": doc_id,
                    "text": sentence,
                    "strategy": "sentence",
                    "metadata": {
                        "parent_doc_id": doc_id,
                        "topic": topic,
                        "title": title,
                        "strategy": "sentence",
                        "chunk_index": idx,
                    },
                }
            )

    return chunks


if __name__ == "__main__":
    docs = load_knowledge_base()
    fixed_chunks = chunk_documents_fixed(docs)
    sent_chunks = chunk_documents_sentence(docs)
    print(f"Strategy A (Fixed-size with overlap): {len(fixed_chunks)} chunks generated.")
    print(f"Strategy B (Sentence-based): {len(sent_chunks)} chunks generated.")
    print("Sample fixed chunk:", fixed_chunks[0])
    print("Sample sentence chunk:", sent_chunks[0])

