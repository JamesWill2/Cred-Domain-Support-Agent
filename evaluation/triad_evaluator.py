"""
RAG Triad Evaluator for Cred Domain Support Agent.

Evaluates:
1. Context Relevance (0.0 - 1.0)
2. Groundedness (0.0 - 1.0)
3. Answer Relevance (0.0 - 1.0)
Across 15 benchmark queries covering all 12 KB topics and out-of-scope/edge queries.
"""

from typing import Dict, List, Any
from pathlib import Path
import json
from evaluation.test_queries import TEST_QUERIES_15
from agent.graph import run_cred_agent
from agent.mock_llm import get_mock_llm
from rag.vectorstore import get_vector_store

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_JSON = RESULTS_DIR / "rag_triad_results.json"


def run_rag_triad_evaluation() -> Dict[str, Any]:
    """Execute evaluation across all 15 queries and persist findings."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    vsm = get_vector_store()
    judge = get_mock_llm()

    results: List[Dict[str, Any]] = []
    total_ctx_rel = 0.0
    total_groundedness = 0.0
    total_ans_rel = 0.0

    print("=========================================================================================")
    print(" RAG TRIAD EVALUATION (15 Queries across all 12 KB Topics + 3 Edge Cases)")
    print("=========================================================================================")

    for item in TEST_QUERIES_15:
        qid = item["id"]
        topic = item["topic"]
        qtext = item["query"]
        is_out = item["is_out_of_scope"]

        # Run full agent
        agent_out = run_cred_agent(
            user_input=qtext,
            session_id=f"eval-session-{qid}",
        )

        response_text = agent_out.get("response", "")

        # Retrieve top context chunks for judging
        hits = vsm.query(qtext, n_results=2)
        context_str = " ".join(h["chunk_text"] for h in hits)

        # Judge RAG triad
        scores = judge.judge_rag_triad(
            query=qtext,
            retrieved_context=context_str,
            answer=response_text,
            is_out_of_scope=is_out,
        )

        c_rel = scores["context_relevance"]
        grd = scores["groundedness"]
        a_rel = scores["answer_relevance"]

        total_ctx_rel += c_rel
        total_groundedness += grd
        total_ans_rel += a_rel

        query_record = {
            "query_id": qid,
            "topic": topic,
            "query": qtext,
            "is_out_of_scope": is_out,
            "agent_response": response_text,
            "sources": agent_out.get("sources", []),
            "refusal_reason": agent_out.get("refusal_reason"),
            "context_relevance": c_rel,
            "groundedness": grd,
            "answer_relevance": a_rel,
        }
        results.append(query_record)

        status_tag = "[OUT-OF-SCOPE/REFUSED]" if is_out else "[IN-SCOPE/GROUNDED]"
        print(f"{qid} | {status_tag} Topic: {topic[:30]:<30}")
        print(f"     Query: {qtext[:60]}...")
        print(f"     Context Relevance: {c_rel:.2f} | Groundedness: {grd:.2f} | Answer Relevance: {a_rel:.2f}\n")

    n = len(TEST_QUERIES_15)
    avg_ctx_rel = round(total_ctx_rel / n, 4)
    avg_groundedness = round(total_groundedness / n, 4)
    avg_ans_rel = round(total_ans_rel / n, 4)

    summary = {
        "total_queries_evaluated": n,
        "average_context_relevance": avg_ctx_rel,
        "average_groundedness": avg_groundedness,
        "average_answer_relevance": avg_ans_rel,
        "detailed_results": results,
    }

    # Save to JSON
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=========================================================================================")
    print(" RAG TRIAD SUMMARY REPORT")
    print("=========================================================================================")
    print(f"Total Queries Evaluated:    {n}")
    print(f"Average Context Relevance:  {avg_ctx_rel:.4f}")
    print(f"Average Groundedness:       {avg_groundedness:.4f}")
    print(f"Average Answer Relevance:   {avg_ans_rel:.4f}")
    print(f"Results saved to:           {OUTPUT_JSON}")
    print("=========================================================================================\n")

    return summary


if __name__ == "__main__":
    run_rag_triad_evaluation()

