"""Tests for Resilience: SQLite Checkpointing, Retry, and Timeouts."""
import pytest
import time
from resilience.checkpoint import demonstrate_checkpoint_resume
from resilience.retry import demonstrate_retry_recovery, exponential_backoff_retry, TransientServiceSimulator, TransientNetworkError
from resilience.timeouts import (
    run_with_node_timeout,
    run_with_global_timeout,
    NodeTimeoutError,
    GlobalGraphTimeoutError,
)


def test_sqlite_checkpoint_resume(tmp_path):
    """Verify LangGraph resumes from SQLite checkpoint without re-executing completed nodes."""
    db_file = tmp_path / "test_checkpoints.sqlite"
    res = demonstrate_checkpoint_resume(thread_id="test-thread-pytest-1", db_path=db_file)

    assert res["re_execution_prevented"] is True
    assert res["phase1_executed_nodes"] == ["tracked_guardrail_node", "tracked_router_node"]
    assert res["cumulative_executed_nodes"] == [
        "tracked_guardrail_node",
        "tracked_router_node",
        "tracked_tool_node",
        "tracked_synth_node",
    ]
    assert res["final_output"]["intent"] == "loan_status"


def test_exponential_backoff_retry():
    """Verify retry policy recovers transient failure within max attempts."""
    res = demonstrate_retry_recovery()
    assert res["call_count"] == 3
    assert res["final_result"]["status"] == "success"


def test_retry_exhaustion():
    """Verify retry policy raises exception if failures exceed max_attempts."""
    stub = TransientServiceSimulator(failures_before_success=5)

    @exponential_backoff_retry(max_attempts=2, initial_interval=0.01, max_interval=0.05, jitter=False)
    def fail_always():
        return stub.execute_call({"test": True})

    with pytest.raises(TransientNetworkError):
        fail_always()

    assert stub.call_count == 2


def test_per_node_timeout():
    """Verify per-node timeout raises NodeTimeoutError cleanly without hanging."""
    def slow_node():
        time.sleep(1.0)
        return "finished"

    with pytest.raises(NodeTimeoutError) as exc_info:
        run_with_node_timeout(slow_node, timeout_seconds=0.2, node_name="simulated_slow_node")

    assert "simulated_slow_node" in str(exc_info.value)
    assert "exceeded maximum timeout" in str(exc_info.value)


def test_global_graph_timeout():
    """Verify global graph timeout raises GlobalGraphTimeoutError and cancels execution."""
    def slow_graph():
        time.sleep(0.8)
        return "graph_done"

    with pytest.raises(GlobalGraphTimeoutError) as exc_info:
        run_with_global_timeout(slow_graph, global_timeout_seconds=0.2)

    assert "exceeded total graph deadline" in str(exc_info.value)

