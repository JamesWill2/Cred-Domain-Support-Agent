"""
SQLite Checkpointing and Graph State Persistence for Cred Domain Support Agent.

Demonstrates:
(a) Graph executes at least two nodes (guardrail_input_node and intent_router_node)
(b) Execution deliberately stopped/interrupted before remaining nodes run
(c) Resuming the SAME thread ID and completing the run
(d) Explicitly showing that already-completed nodes are NOT re-executed
(e) State is restored from checkpoints.sqlite
"""

from typing import Dict, Any, Optional, List
import sqlite3
import os
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.guardrails import mask_pii, detect_prompt_injection
from agent.tools import check_loan_application_status
from agent.schema import validate_agent_response

CHECKPOINTS_DB_PATH = Path(__file__).resolve().parent.parent / "checkpoints.sqlite"


def get_sqlite_checkpointer(db_path: Optional[Path] = None) -> SqliteSaver:
    """Create a persistent SQLite checkpointer."""
    path = str(db_path or CHECKPOINTS_DB_PATH)
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)


def build_checkpointable_graph(checkpointer: SqliteSaver, interrupt_before_nodes: Optional[List[str]] = None):
    """
    Builds a tracked 4-node graph for checkpoint demonstration:
    - Node 1: tracked_guardrail_node
    - Node 2: tracked_router_node
    - Node 3: tracked_tool_node
    - Node 4: tracked_synth_node
    Tracks node execution count in state to mathematically prove non-reexecution.
    """
    class CheckpointState(AgentState):
        node_execution_log: List[str]
        step_counter: int

    def tracked_guardrail_node(state: CheckpointState) -> Dict[str, Any]:
        log = list(state.get("node_execution_log", []))
        log.append("tracked_guardrail_node")
        print("  [NODE 1 EXECUTED] tracked_guardrail_node: Masking PII & scanning injection")
        sanitized, meta = mask_pii(state.get("raw_input", ""))
        is_inj, reason = detect_prompt_injection(sanitized)
        return {
            "sanitized_input": sanitized,
            "pii_metadata": meta,
            "is_injection": is_inj,
            "injection_reason": reason,
            "node_execution_log": log,
            "step_counter": state.get("step_counter", 0) + 1,
        }

    def tracked_router_node(state: CheckpointState) -> Dict[str, Any]:
        log = list(state.get("node_execution_log", []))
        log.append("tracked_router_node")
        print("  [NODE 2 EXECUTED] tracked_router_node: Routing intent to loan_status")
        return {
            "intent": "loan_status",
            "target_loan_id": "LOAN-1001",
            "node_execution_log": log,
            "step_counter": state.get("step_counter", 0) + 1,
        }

    def tracked_tool_node(state: CheckpointState) -> Dict[str, Any]:
        log = list(state.get("node_execution_log", []))
        log.append("tracked_tool_node")
        print("  [NODE 3 EXECUTED] tracked_tool_node: Calling check_loan_application_status('LOAN-1001')")
        rec = check_loan_application_status("LOAN-1001")
        return {
            "loan_record_result": rec,
            "synthesized_response": rec["message"],
            "tool_used": "check_loan_application_status",
            "escalation_score": rec.get("escalation_score"),
            "node_execution_log": log,
            "step_counter": state.get("step_counter", 0) + 1,
        }

    def tracked_synth_node(state: CheckpointState) -> Dict[str, Any]:
        log = list(state.get("node_execution_log", []))
        log.append("tracked_synth_node")
        print("  [NODE 4 EXECUTED] tracked_synth_node: Assembling final validated structured response")
        resp_data = {
            "response": state.get("synthesized_response", ""),
            "intent": state.get("intent", "loan_status"),
            "sources": ["LOAN-1001"],
            "tool_used": state.get("tool_used"),
            "escalation_score": state.get("escalation_score"),
            "grounded": True,
            "refusal_reason": None,
            "trace_id": state.get("trace_id", "trace-ckpt-001"),
        }
        validated = validate_agent_response(resp_data)
        return {
            "final_output": validated.model_dump(),
            "node_execution_log": log,
            "step_counter": state.get("step_counter", 0) + 1,
        }

    builder = StateGraph(CheckpointState)
    builder.add_node("tracked_guardrail_node", tracked_guardrail_node)
    builder.add_node("tracked_router_node", tracked_router_node)
    builder.add_node("tracked_tool_node", tracked_tool_node)
    builder.add_node("tracked_synth_node", tracked_synth_node)

    builder.set_entry_point("tracked_guardrail_node")
    builder.add_edge("tracked_guardrail_node", "tracked_router_node")
    builder.add_edge("tracked_router_node", "tracked_tool_node")
    builder.add_edge("tracked_tool_node", "tracked_synth_node")
    builder.add_edge("tracked_synth_node", END)

    interrupts = interrupt_before_nodes or ["tracked_tool_node"]
    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupts)


def demonstrate_checkpoint_resume(
    thread_id: str = "cred-thread-session-42",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes and records the complete interruption and resumption demonstration.

    Returns:
        Summary of demonstration results and assertion verifications.
    """
    checkpointer = get_sqlite_checkpointer(db_path)
    app = build_checkpointable_graph(checkpointer, interrupt_before_nodes=["tracked_tool_node"])

    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {
        "session_id": "session-ckpt-demo",
        "trace_id": "trace-ckpt-1001",
        "raw_input": "Please check the status of my loan LOAN-1001",
        "sanitized_input": "",
        "pii_metadata": {},
        "is_injection": False,
        "injection_reason": None,
        "intent": "policy_rag",
        "target_loan_id": None,
        "retrieved_chunks": [],
        "loan_record_result": None,
        "synthesized_response": "",
        "sources": [],
        "tool_used": None,
        "escalation_score": None,
        "grounded": True,
        "refusal_reason": None,
        "final_output": None,
        "node_execution_log": [],
        "step_counter": 0,
    }

    log_lines = []
    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    log("=========================================================================")
    log(" SQLITE CHECKPOINT & RESUMPTION DEMONSTRATION")
    log(f" Database: {CHECKPOINTS_DB_PATH} | Thread ID: {thread_id}")
    log("=========================================================================")

    # PHASE 1: Execution until deliberate interrupt before Node 3
    log("\n--- PHASE 1: Launching initial run (Interrupt set before 'tracked_tool_node') ---")
    state_step1 = app.invoke(initial_input, config=config)
    
    executed_phase1 = state_step1.get("node_execution_log", [])
    log(f"Phase 1 Execution Halted at Interrupt. Nodes executed: {executed_phase1}")
    log(f"Phase 1 State Step Counter: {state_step1.get('step_counter')}")
    log(f"State stored in checkpoints.sqlite under thread_id: '{thread_id}'")

    # Verify that exactly Node 1 and Node 2 ran
    assert executed_phase1 == ["tracked_guardrail_node", "tracked_router_node"], (
        f"Expected exactly 2 nodes in Phase 1, got {executed_phase1}"
    )

    # PHASE 2: Resuming with the same thread ID
    log("\n--- PHASE 2: Resuming execution on SAME thread_id (None input payload) ---")
    state_step2 = app.invoke(None, config=config)

    executed_phase2 = state_step2.get("node_execution_log", [])
    log(f"Phase 2 Execution Completed. Cumulative Nodes executed: {executed_phase2}")
    log(f"Final State Step Counter: {state_step2.get('step_counter')}")
    log(f"Final Validated Output: {state_step2.get('final_output')}")

    # VERIFICATION: Node 1 and Node 2 must appear EXACTLY ONCE across the entire lifecycle
    count_node1 = executed_phase2.count("tracked_guardrail_node")
    count_node2 = executed_phase2.count("tracked_router_node")
    count_node3 = executed_phase2.count("tracked_tool_node")
    count_node4 = executed_phase2.count("tracked_synth_node")

    log("\n--- NODE EXECUTION COUNT AUDIT ---")
    log(f"  * Node 1 (tracked_guardrail_node) executions: {count_node1} [Must be 1]")
    log(f"  * Node 2 (tracked_router_node) executions:    {count_node2} [Must be 1]")
    log(f"  * Node 3 (tracked_tool_node) executions:      {count_node3} [Must be 1]")
    log(f"  * Node 4 (tracked_synth_node) executions:     {count_node4} [Must be 1]")

    assert count_node1 == 1, "FAILURE: Node 1 was re-executed upon resume!"
    assert count_node2 == 1, "FAILURE: Node 2 was re-executed upon resume!"
    assert count_node3 == 1, "FAILURE: Node 3 did not execute on resume!"
    assert count_node4 == 1, "FAILURE: Node 4 did not execute on resume!"
    assert state_step2.get("step_counter") == 4, "FAILURE: Step counter mismatch!"

    log("=========================================================================")
    log(" CHECKPOINT VERIFICATION SUCCESSFUL: ZERO RE-EXECUTION OF PRIOR NODES [PASS]")
    log("=========================================================================\n")

    return {
        "thread_id": thread_id,
        "phase1_executed_nodes": executed_phase1,
        "cumulative_executed_nodes": executed_phase2,
        "re_execution_prevented": (count_node1 == 1 and count_node2 == 1),
        "final_output": state_step2.get("final_output"),
        "transcript": "\n".join(log_lines),
    }


if __name__ == "__main__":
    demonstrate_checkpoint_resume()

