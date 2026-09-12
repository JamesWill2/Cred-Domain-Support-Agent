"""
Empirical Unknown Fallback Threshold Calibration Script.

Measures top-1 cosine similarity for in-scope vs deliberately out-of-scope queries
to empirically calibrate the 'I don't know' fallback threshold, avoiding
arbitrary tutorial defaults.
"""

from typing import Dict, List, Any
from rag.vectorstore import get_vector_store, COLLECTION_SENTENCE, COLLECTION_FIXED

# At least 5 in-scope banking queries
IN_SCOPE_QUERIES: List[str] = [
    "What are the KYC document requirements for account opening?",
    "How is loan EMI calculated using reducing balance method?",
    "What is the annual membership fee and waiver limit for credit cards?",
    "What are the interest rate slabs for floating home loans and term deposits?",
    "Are there any prepayment penalties or foreclosure charges on retail loans?",
]

# At least 3 deliberately out-of-scope queries
OUT_OF_SCOPE_QUERIES: List[str] = [
    "What is the weather forecast and rainfall probability in Mumbai tomorrow?",
    "How do I bake a chocolate sponge cake with frosting at home?",
    "Who won the cricket world cup tournament in 2023?",
]


def run_calibration(collection_name: str = COLLECTION_SENTENCE) -> Dict[str, Any]:
    """
    Run empirical similarity measurement and derive calibrated fallback threshold.

    Returns:
        Dictionary containing measurements, separation margin, and recommended threshold.
    """
    vsm = get_vector_store()

    in_scope_results = []
    print(f"==================================================")
    print(f" EMPIRICAL THRESHOLD CALIBRATION ({collection_name})")
    print(f"==================================================")
    print("Measuring Top-1 Cosine Similarity for In-Scope Queries:")

    for q in IN_SCOPE_QUERIES:
        hits = vsm.query(q, collection_name=collection_name, n_results=1)
        sim = hits[0]["similarity"] if hits else 0.0
        parent = hits[0]["parent_doc_id"] if hits else "none"
        in_scope_results.append({"query": q, "similarity": sim, "parent": parent})
        print(f"  [In-Scope] Sim: {sim:.4f} | Parent: {parent:<26} | Query: '{q}'")

    print("\nMeasuring Top-1 Cosine Similarity for Out-of-Scope Queries:")
    out_scope_results = []
    for q in OUT_OF_SCOPE_QUERIES:
        hits = vsm.query(q, collection_name=collection_name, n_results=1)
        sim = hits[0]["similarity"] if hits else 0.0
        parent = hits[0]["parent_doc_id"] if hits else "none"
        out_scope_results.append({"query": q, "similarity": sim, "parent": parent})
        print(f"  [Out-Scope] Sim: {sim:.4f} | Parent: {parent:<26} | Query: '{q}'")

    min_in_scope = min(r["similarity"] for r in in_scope_results)
    max_out_scope = max(r["similarity"] for r in out_scope_results)
    margin = min_in_scope - max_out_scope

    # Midpoint calibration between in-scope cluster floor and out-of-scope ceiling
    calibrated_threshold = round((min_in_scope + max_out_scope) / 2.0, 4)

    print("\n--- CALIBRATION ANALYSIS ---")
    print(f"  Lowest In-Scope Top-1 Similarity:   {min_in_scope:.4f}")
    print(f"  Highest Out-of-Scope Top-1 Sim:    {max_out_scope:.4f}")
    print(f"  Separation Margin:                 {margin:.4f}")
    print(f"  Empirically Calibrated Threshold:  {calibrated_threshold:.4f}")
    print("==================================================")

    return {
        "collection": collection_name,
        "in_scope_measurements": in_scope_results,
        "out_scope_measurements": out_scope_results,
        "min_in_scope": min_in_scope,
        "max_out_scope": max_out_scope,
        "separation_margin": margin,
        "calibrated_threshold": calibrated_threshold,
    }


if __name__ == "__main__":
    run_calibration()

