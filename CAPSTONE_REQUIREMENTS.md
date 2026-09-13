# Capstone Requirements & Verification

This document maps all 32 acceptance criteria and requirements from the official Cred Domain Support Agent capstone brief to their corresponding implementation files, automated pytest tests, and demonstration scripts/transcripts.

| ID | Requirement Description | Implementation File | Demonstration / Script |
|---|---|---|---|
| **REQ-01** | Deterministic LOAN_APPLICATIONS dataset (≥40 records, 5 categories, 5 statuses, seeded) | `dataset/dataset.py` | `scripts/run_all_demos.py` |
| **REQ-02** | Structural thresholds: each category ≥3, each status ≥1, fraud rate 10%–30% | `dataset/dataset.py` | `scripts/run_all_demos.py` |
| **REQ-03** | Original Knowledge Base with ≥12 documents (2–5 sentences) covering all required topics | `knowledge_base/docs/*.md`, `knowledge_base/loader.py` | `knowledge_base/loader.py` |
| **REQ-04** | Dual chunking strategies: Fixed-size overlap & Sentence-based | `rag/chunking.py` | `rag/chunking.py` |
| **REQ-05** | SentenceTransformers embedding (`all-MiniLM-L6-v2`) | `rag/vectorstore.py` | `rag/vectorstore.py` |
| **REQ-06** | Two separate ChromaDB collections (`cred_kb_fixed`, `cred_kb_sentence`) | `rag/vectorstore.py` | `rag/vectorstore.py` |
| **REQ-07** | Grounded generation: answers rely strictly on retrieved context without hallucination | `rag/grounded_generator.py` | `rag/grounded_generator.py` |
| **REQ-08** | Empirical "I don't know" threshold calibration (≥3 in-scope vs ≥2 out-of-scope) | `rag/calibrate.py` | `scripts/run_all_demos.py` |
| **REQ-09** | Demonstration of ≥5 in-scope queries + 1 out-of-scope fallback trigger | `rag/grounded_generator.py` | `scripts/run_all_demos.py` |
| **REQ-10** | Document-level Precision@3 & Recall@3 evaluation with deduplication & visible arithmetic | `rag/evaluate_chunks.py` | `scripts/run_all_demos.py` |
| **REQ-11** | Chunking strategy recommendation based on empirical numbers | `rag/evaluate_chunks.py`, `README.md` | `README.md` |
| **REQ-12** | `check_loan_application_status` tool returning status, amount, and escalation score | `agent/tools.py` | `scripts/run_all_demos.py` |
| **REQ-13** | Mathematically designed escalation score in [0, 1] combining fraud flag & recency | `agent/tools.py` | `scripts/run_all_demos.py` |
| **REQ-14** | Escalation threshold justified against dataset distribution (e.g. 80th percentile) | `agent/tools.py`, `README.md` | `agent/tools.py` |
| **REQ-15** | LangGraph graph with ≥4 nodes | `agent/graph.py` | `scripts/run_all_demos.py` |
| **REQ-16** | Genuine conditional edge routing to RAG or Loan tool based on intent | `agent/graph.py` | `scripts/run_all_demos.py` |
| **REQ-17** | Persisted multi-turn conversation memory to JSON | `agent/memory.py` | `transcripts/multi_turn_memory.log` |
| **REQ-18** | Fresh-conversation demonstration showing absent/reset memory | `agent/memory.py` | `transcripts/fresh_conversation.log` |
| **REQ-19** | Strict structured output schema (Pydantic / JSON Schema validation) | `agent/schema.py` | `scripts/run_all_demos.py` |
| **REQ-20** | Input guardrail: Fixed-format PII masking (PAN, Aadhaar, Bank Account #) | `agent/guardrails.py` | `transcripts/guardrails_demo.log` |
| **REQ-21** | Input guardrail: Prompt-injection detection | `agent/guardrails.py` | `transcripts/guardrails_demo.log` |
| **REQ-22** | Output guardrail: Groundedness check refusing unsupported queries | `agent/guardrails.py` | `transcripts/guardrails_demo.log` |
| **REQ-23** | FastAPI deployment exposing `POST /ask`, `POST /add-document`, `GET /health` | `api/main.py`, `api/models.py` | `scripts/run_all_demos.py` |
| **REQ-24** | Structured JSONL logging with trace_id, timing, and masked PII (never raw PII on disk) | `api/logger.py` | `logs/agent_requests.jsonl` |
| **REQ-25** | RAG-triad evaluation over 15 queries (12 topics + 2 out-of-scope/edge) | `evaluation/triad_evaluator.py` | `evaluation/results/rag_triad_results.json` |
| **REQ-26** | Deterministic MOCK_LLM provider (zero API key / zero network requirement) | `agent/mock_llm.py` | `scripts/run_all_demos.py` |
| **REQ-27** | FastMCP server exposing `check_loan_application_status` at `/mcp` | `mcp/server.py` | `scripts/run_all_demos.py` |
| **REQ-28** | Separate MCP client script calling server for ≥2 record IDs via round trip | `mcp/client.py` | `transcripts/mcp_roundtrip.log` |
| **REQ-29** | SQLite-based LangGraph checkpointing (`checkpoints.sqlite`) with interrupt & resume | `resilience/checkpoint.py` | `transcripts/checkpoint_resume.log` |
| **REQ-30** | Exponential-backoff retry policy recovering simulated transient failure | `resilience/retry.py` | `transcripts/retry_timeout_demo.log` |
| **REQ-31** | Per-node timeout and global graph timeout preventing hangs and cancelling runs | `resilience/timeouts.py` |`transcripts/retry_timeout_demo.log` |
| **REQ-32** | Final automated capstone verification audit script producing final report | `scripts/verify_capstone.py` | `evaluation/results/final_capstone_report.md` |

