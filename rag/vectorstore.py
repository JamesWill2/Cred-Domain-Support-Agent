"""
Vector Store Manager for Cred Knowledge Base using ChromaDB and all-MiniLM-L6-v2.

Manages two independent collections:
- 'cred_kb_fixed': Indexed with Strategy A (fixed-size with overlap)
- 'cred_kb_sentence': Indexed with Strategy B (sentence-based chunks)
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from knowledge_base.loader import load_knowledge_base
from rag.chunking import chunk_documents_fixed, chunk_documents_sentence

DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_data"
COLLECTION_FIXED = "cred_kb_fixed"
COLLECTION_SENTENCE = "cred_kb_sentence"


class VectorStoreManager:
    """Manages dual ChromaDB collections with local all-MiniLM-L6-v2 embeddings."""

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or DEFAULT_CHROMA_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Free local SentenceTransformer embedding function (all-MiniLM-L6-v2, 384 dims)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        self._init_collections()

    def _init_collections(self):
        """Initialize or get both ChromaDB collections."""
        self.col_fixed = self.client.get_or_create_collection(
            name=COLLECTION_FIXED,
            embedding_function=self.embedding_fn,
            metadata={"description": "Cred KB Fixed-size chunks with overlap (Strategy A)", "hnsw:space": "cosine"},
        )
        self.col_sentence = self.client.get_or_create_collection(
            name=COLLECTION_SENTENCE,
            embedding_function=self.embedding_fn,
            metadata={"description": "Cred KB Sentence-based chunks (Strategy B)", "hnsw:space": "cosine"},
        )

    def build_indexes(self, force_rebuild: bool = False) -> Dict[str, int]:
        """
        Build and populate both ChromaDB collections from KB documents.

        Args:
            force_rebuild: If True, clears existing collections before re-indexing.

        Returns:
            Dictionary containing chunk counts for each collection.
        """
        if force_rebuild:
            try:
                self.client.delete_collection(COLLECTION_FIXED)
            except Exception:
                pass
            try:
                self.client.delete_collection(COLLECTION_SENTENCE)
            except Exception:
                pass
            self._init_collections()

        # If already indexed and not force rebuilding, return current counts
        fixed_count = self.col_fixed.count()
        sentence_count = self.col_sentence.count()
        if fixed_count > 0 and sentence_count > 0 and not force_rebuild:
            return {
                COLLECTION_FIXED: fixed_count,
                COLLECTION_SENTENCE: sentence_count,
            }

        docs = load_knowledge_base()
        fixed_chunks = chunk_documents_fixed(docs)
        sentence_chunks = chunk_documents_sentence(docs)

        # Index Strategy A (Fixed chunks)
        if fixed_chunks:
            self.col_fixed.upsert(
                ids=[c["chunk_id"] for c in fixed_chunks],
                documents=[c["text"] for c in fixed_chunks],
                metadatas=[c["metadata"] for c in fixed_chunks],
            )

        # Index Strategy B (Sentence chunks)
        if sentence_chunks:
            self.col_sentence.upsert(
                ids=[c["chunk_id"] for c in sentence_chunks],
                documents=[c["text"] for c in sentence_chunks],
                metadatas=[c["metadata"] for c in sentence_chunks],
            )

        return {
            COLLECTION_FIXED: self.col_fixed.count(),
            COLLECTION_SENTENCE: self.col_sentence.count(),
        }

    def query(
        self,
        query_text: str,
        collection_name: str = COLLECTION_SENTENCE,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Query a specific collection and return formatted search results with cosine similarity.

        Args:
            query_text: User question or search phrase.
            collection_name: 'cred_kb_sentence' or 'cred_kb_fixed'.
            n_results: Maximum top results to retrieve.

        Returns:
            List of dictionaries with chunk_text, parent_doc_id, similarity, distance, metadata.
        """
        collection = self.col_sentence if collection_name == COLLECTION_SENTENCE else self.col_fixed

        results = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, max(1, collection.count())),
            include=["documents", "metadatas", "distances"],
        )

        formatted_results: List[Dict[str, Any]] = []

        if not results or not results["documents"] or not results["documents"][0]:
            return formatted_results

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
        ids = results["ids"][0] if results["ids"] else [""] * len(docs)

        for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            # Chroma with cosine distance: distance is in [0, 2], where 0 is identical
            # Similarity in [0, 1]: 1.0 - (dist / 2.0) or max(0.0, 1.0 - dist)
            # In cosine space, cosine similarity = 1 - cosine distance
            cosine_similarity = round(max(0.0, min(1.0, 1.0 - float(dist))), 4)

            formatted_results.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_text": doc_text,
                    "parent_doc_id": meta.get("parent_doc_id", "unknown"),
                    "distance": round(float(dist), 4),
                    "similarity": cosine_similarity,
                    "metadata": meta,
                }
            )

        return formatted_results

    def add_document(
        self,
        doc_id: str,
        title: str,
        topic: str,
        content: str,
    ) -> Dict[str, Any]:
        """
        Add a single new document dynamically to both collections.
        Used by the FastAPI POST /add-document endpoint.
        """
        doc = {
            "doc_id": doc_id,
            "title": title,
            "topic": topic,
            "content": content,
            "sentence_count": len([s for s in content.split(".") if s.strip()]),
        }

        fixed_chunks = chunk_documents_fixed([doc])
        sentence_chunks = chunk_documents_sentence([doc])

        if fixed_chunks:
            self.col_fixed.upsert(
                ids=[c["chunk_id"] for c in fixed_chunks],
                documents=[c["text"] for c in fixed_chunks],
                metadatas=[c["metadata"] for c in fixed_chunks],
            )

        if sentence_chunks:
            self.col_sentence.upsert(
                ids=[c["chunk_id"] for c in sentence_chunks],
                documents=[c["text"] for c in sentence_chunks],
                metadatas=[c["metadata"] for c in sentence_chunks],
            )

        return {
            "doc_id": doc_id,
            "fixed_chunks_added": len(fixed_chunks),
            "sentence_chunks_added": len(sentence_chunks),
            "total_fixed_count": self.col_fixed.count(),
            "total_sentence_count": self.col_sentence.count(),
        }


# Singleton manager instance
_vs_manager: Optional[VectorStoreManager] = None


def get_vector_store() -> VectorStoreManager:
    """Get or create singleton VectorStoreManager instance."""
    global _vs_manager
    if _vs_manager is None:
        _vs_manager = VectorStoreManager()
        _vs_manager.build_indexes()
    return _vs_manager


if __name__ == "__main__":
    vsm = get_vector_store()
    counts = vsm.build_indexes(force_rebuild=True)
    print("Indexes successfully built:")
    for k, v in counts.items():
        print(f"  Collection '{k}': {v} chunks")

    sample_q = "What are the KYC documents required?"
    print(f"\nQuerying '{COLLECTION_SENTENCE}' for: '{sample_q}'")
    hits = vsm.query(sample_q, collection_name=COLLECTION_SENTENCE, n_results=2)
    for h in hits:
        print(f"  - Parent: {h['parent_doc_id']} | Sim: {h['similarity']:.4f} | Text: {h['chunk_text'][:80]}...")

