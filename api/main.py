"""
FastAPI Application for Cred Domain Support Agent.

Exposes:
- POST /ask: Primary conversational support endpoint using LangGraph and Guardrails
- POST /add-document: Dynamic knowledge base ingestion into dual ChromaDB collections
- GET /health: Operational readiness and collection statistics
"""

import time
import uuid
from typing import Dict, Any
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.models import (
    AskRequest,
    AddDocumentRequest,
    AddDocumentResponse,
    HealthResponse,
    SimpleResponse,
)
from agent.schema import AgentResponse
from agent.graph import run_cred_agent
from api.logger import get_jsonl_logger
from rag.vectorstore import get_vector_store
from dataset.dataset import LOAN_APPLICATIONS

app = FastAPI(
    title="Cred Domain Support Agent API",
    description="Production-grade AI Domain Support Agent for Cred Lending Operations",
    version="1.0.0",
)
# Serve the UI at the root path
from fastapi.responses import HTMLResponse, FileResponse

import os

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Return the static index.html page for the web UI."""
    index_path = os.path.join("frontend", "index.html")
    return FileResponse(index_path, media_type="text/html")

# If you ever add assets (CSS, JS, images) place them inside the frontend folder and expose them under /static
app.mount("/static", StaticFiles(directory="frontend", html=False), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    """Timing and structured logging middleware recording requests to JSONL."""
    start_time = time.time()
    trace_id = request.headers.get("X-Trace-Id", f"trace-{uuid.uuid4().hex[:12]}")
    request.state.trace_id = trace_id

    # Process request
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000.0

    return response


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """Operational health check returning system metrics and index counts."""
    vsm = get_vector_store()
    return HealthResponse(
        status="healthy",
        app_name="Cred Domain Support Agent",
        version="1.0.0",
        total_loan_records=len(LOAN_APPLICATIONS),
        fixed_chunks_count=vsm.col_fixed.count(),
        sentence_chunks_count=vsm.col_sentence.count(),
    )


@app.get("/ask", tags=["Agent"])
async def ask_get():
    """Return a friendly message for GET requests to /ask.
    The agent expects a POST with JSON payload.
    """
    return {"detail": "Please send a POST request with JSON body {\"query\": \"...\", \"session_id\": \"...\"}"}


@app.post("/ask", response_model=SimpleResponse, tags=["Agent"])
async def ask_agent(req: AskRequest, request: Request):
    """Main conversational agent endpoint.
    Processes user queries through LangGraph orchestration, guardrails, and tools.
    """
    start_time = time.time()
    trace_id = req.trace_id or getattr(request.state, "trace_id", f"trace-{uuid.uuid4().hex[:12]}")
    logger = get_jsonl_logger()
    try:
        # Execute LangGraph workflow
        agent_out = run_cred_agent(
            user_input=req.query,
            session_id=req.session_id or "default-session",
            trace_id=trace_id,
        )
        latency_ms = (time.time() - start_time) * 1000.0
        # Log to structured JSONL with zero raw PII
        logger.log_request(
            endpoint="/ask",
            trace_id=trace_id,
            status_code=status.HTTP_200_OK,
            latency_ms=latency_ms,
            raw_request_payload=req.model_dump(),
            response_payload=agent_out,
            client_ip=request.client.host if request.client else "127.0.0.1",
            extra_metadata={"session_id": req.session_id, "intent": agent_out.get("intent")},
        )
        return SimpleResponse(answer=agent_out.get("response", ""))
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000.0
        error_resp = {"error": str(e), "trace_id": trace_id}
        logger.log_request(
            endpoint="/ask",
            trace_id=trace_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            latency_ms=latency_ms,
            raw_request_payload=req.model_dump(),
            response_payload=error_resp,
            client_ip=request.client.host if request.client else "127.0.0.1",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution error: {str(e)}",
        )


@app.post("/add-document", response_model=AddDocumentResponse, tags=["Knowledge Base"])
async def add_document(req: AddDocumentRequest, request: Request):
    """
    Dynamically ingest a new policy document into both ChromaDB collections.
    """
    start_time = time.time()
    trace_id = getattr(request.state, "trace_id", f"trace-{uuid.uuid4().hex[:12]}")
    logger = get_jsonl_logger()

    try:
        vsm = get_vector_store()
        ingest_res = vsm.add_document(
            doc_id=req.doc_id,
            title=req.title,
            topic=req.topic,
            content=req.content,
        )

        latency_ms = (time.time() - start_time) * 1000.0
        resp_data = {
            "success": True,
            "doc_id": req.doc_id,
            "fixed_chunks_added": ingest_res["fixed_chunks_added"],
            "sentence_chunks_added": ingest_res["sentence_chunks_added"],
            "total_fixed_count": ingest_res["total_fixed_count"],
            "total_sentence_count": ingest_res["total_sentence_count"],
            "message": f"Successfully indexed '{req.doc_id}' into both ChromaDB collections.",
        }

        logger.log_request(
            endpoint="/add-document",
            trace_id=trace_id,
            status_code=status.HTTP_200_OK,
            latency_ms=latency_ms,
            raw_request_payload=req.model_dump(),
            response_payload=resp_data,
            client_ip=request.client.host if request.client else "127.0.0.1",
        )

        return AddDocumentResponse.model_validate(resp_data)

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000.0
        error_resp = {"error": str(e), "trace_id": trace_id}
        logger.log_request(
            endpoint="/add-document",
            trace_id=trace_id,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            latency_ms=latency_ms,
            raw_request_payload=req.model_dump(),
            response_payload=error_resp,
            client_ip=request.client.host if request.client else "127.0.0.1",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
