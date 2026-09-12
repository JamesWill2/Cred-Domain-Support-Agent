"""
Grounded Generation Engine for Cred Knowledge Base.

Enforces strict groundedness: answers rely ONLY on retrieved context.
If top-1 retrieval similarity falls below the empirically calibrated threshold,
the system triggers an 'I don't know' fallback refusal rather than hallucinating.
"""

from typing import Dict, List, Any, Optional
from rag.vectorstore import get_vector_store, COLLECTION_SENTENCE
from agent.mock_llm import get_mock_llm

# Empirically calibrated fallback threshold separating in-scope from out-of-scope queries
# Measured via rag/calibrate.py: in-scope top-1 similarity ~ 0.55 - 0.72; out-of-scope ~ 0.15 - 0.28
DEFAULT_FALLBACK_THRESHOLD: float = 0.40


class GroundedGenerator:
    """Generates grounded responses based strictly on retrieved ChromaDB context."""

    def __init__(
        self,
        fallback_threshold: float = DEFAULT_FALLBACK_THRESHOLD,
        collection_name: str = COLLECTION_SENTENCE,
    ):
        self.fallback_threshold = fallback_threshold
        self.collection_name = collection_name
        self.vector_store = get_vector_store()
        self.llm = get_mock_llm()

    def generate(
        self,
        query: str,
        n_results: int = 5,
        threshold_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve context and generate grounded answer.

        Args:
            query: User's banking question.
            n_results: Number of chunks to retrieve.
            threshold_override: Optional custom similarity threshold.

        Returns:
            Dictionary with response, sources, top_similarity, is_fallback, refusal_reason.
        """
        threshold = threshold_override if threshold_override is not None else self.fallback_threshold

        # 1. Retrieve top chunks from vector store
        hits = self.vector_store.query(
            query_text=query,
            collection_name=self.collection_name,
            n_results=n_results,
        )

        top_similarity = hits[0]["similarity"] if hits else 0.0
        is_fallback = top_similarity < threshold

        # 2. Synthesize answer via deterministic LLM provider
        llm_result = self.llm.generate_grounded_answer(
            query=query,
            retrieved_chunks=hits,
            is_fallback=is_fallback,
        )

        return {
            "query": query,
            "response": llm_result["answer"],
            "sources": llm_result["sources"],
            "grounded": llm_result["grounded"],
            "refusal_reason": llm_result["refusal_reason"],
            "top_similarity": top_similarity,
            "threshold": threshold,
            "is_fallback": is_fallback,
            "retrieved_chunks": hits,
        }


# Singleton generator instance
_grounded_generator: Optional[GroundedGenerator] = None


def get_grounded_generator(
    fallback_threshold: float = DEFAULT_FALLBACK_THRESHOLD,
    collection_name: str = COLLECTION_SENTENCE,
) -> GroundedGenerator:
    """Get singleton GroundedGenerator."""
    global _grounded_generator
    if _grounded_generator is None:
        _grounded_generator = GroundedGenerator(
            fallback_threshold=fallback_threshold,
            collection_name=collection_name,
        )
    return _grounded_generator


if __name__ == "__main__":
    gen = get_grounded_generator()
    in_scope_test = "What documents do I need for KYC verification?"
    out_scope_test = "How do I make chocolate chip cookies at home?"

    print(f"--- Testing In-Scope Query: '{in_scope_test}' ---")
    res1 = gen.generate(in_scope_test)
    print(f"Top Similarity: {res1['top_similarity']:.4f} (Threshold: {res1['threshold']})")
    print(f"Is Fallback: {res1['is_fallback']}")
    print(f"Sources: {res1['sources']}")
    print(f"Response: {res1['response']}\n")

    print(f"--- Testing Out-of-Scope Query: '{out_scope_test}' ---")
    res2 = gen.generate(out_scope_test)
    print(f"Top Similarity: {res2['top_similarity']:.4f} (Threshold: {res2['threshold']})")
    print(f"Is Fallback: {res2['is_fallback']}")
    print(f"Refusal Reason: {res2['refusal_reason']}")
    print(f"Response: {res2['response']}")

