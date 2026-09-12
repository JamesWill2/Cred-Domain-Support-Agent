"""Tests for Knowledge Base, Dual Chunking, Vectorstore, and Calibration."""
import pytest
from knowledge_base.loader import load_knowledge_base, REQUIRED_TOPICS
from rag.chunking import chunk_documents_fixed, chunk_documents_sentence
from rag.vectorstore import get_vector_store, COLLECTION_FIXED, COLLECTION_SENTENCE
from rag.calibrate import run_calibration
from rag.evaluate_chunks import compare_chunking_strategies
from rag.grounded_generator import get_grounded_generator


def test_kb_topics_and_sentences():
    """Verify >= 12 docs, all 12 required topics present, 2-5 sentences each."""
    docs = load_knowledge_base()
    assert len(docs) >= 12
    covered_topics = {d["topic"] for d in docs}
    for topic in REQUIRED_TOPICS:
        assert topic in covered_topics

    for d in docs:
        assert 2 <= d["sentence_count"] <= 5


def test_chunking_strategies():
    """Verify both chunking strategies generate valid non-empty chunks."""
    docs = load_knowledge_base()
    fixed_chunks = chunk_documents_fixed(docs)
    sent_chunks = chunk_documents_sentence(docs)

    assert len(fixed_chunks) > 0
    assert len(sent_chunks) > 0

    for c in fixed_chunks:
        assert c["strategy"] == "fixed"
        assert "parent_doc_id" in c
        assert len(c["text"]) > 0

    for c in sent_chunks:
        assert c["strategy"] == "sentence"
        assert "parent_doc_id" in c
        assert len(c["text"]) > 0


def test_dual_collections():
    """Verify both ChromaDB collections are independently queryable."""
    vsm = get_vector_store()
    hits_fixed = vsm.query("What are the KYC documents?", collection_name=COLLECTION_FIXED, n_results=2)
    hits_sent = vsm.query("What are the KYC documents?", collection_name=COLLECTION_SENTENCE, n_results=2)

    assert len(hits_fixed) > 0
    assert len(hits_sent) > 0
    assert hits_fixed[0]["similarity"] > 0.0
    assert hits_sent[0]["similarity"] > 0.0


def test_calibration_logic():
    """Verify empirical calibration measures separation margin between in-scope and out-of-scope."""
    calib = run_calibration(COLLECTION_SENTENCE)
    assert calib["min_in_scope"] > calib["max_out_scope"]
    assert calib["separation_margin"] > 0.15
    assert 0.20 <= calib["calibrated_threshold"] <= 0.50


def test_in_scope_and_fallback():
    """Verify grounded answer for in-scope query and refusal for out-of-scope query."""
    gen = get_grounded_generator()

    in_scope = gen.generate("What is the interest rate slab for home loans?")
    assert in_scope["is_fallback"] is False
    assert in_scope["grounded"] is True
    assert len(in_scope["sources"]) > 0
    assert "interest" in in_scope["response"].lower()

    out_scope = gen.generate("How to make pizza dough at home?")
    assert out_scope["is_fallback"] is True
    assert "I do not have sufficient information" in out_scope["response"]


def test_precision_recall_eval():
    """Verify Precision@3 and Recall@3 evaluation runs and produces recommendation."""
    eval_res = compare_chunking_strategies()
    assert "strategy_a_fixed" in eval_res
    assert "strategy_b_sentence" in eval_res
    assert eval_res["strategy_b_sentence"]["mean_recall_at_3"] >= 0.80
    assert "Recommendation:" in eval_res["recommendation"]

