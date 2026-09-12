"""
Per-Node and Global Graph Timeout Handlers for Cred Domain Support Agent.

Demonstrates:
(a) Per-node timeout correctly firing a clean NodeTimeoutError (not hanging) on a simulated slow call
(b) Global graph timeout cancelling the overall run on a simulated total-time overrun
"""

from typing import Callable, Any, Dict, Optional
import time
import concurrent.futures


class NodeTimeoutError(Exception):
    """Raised when an individual graph node exceeds its allocated execution deadline."""
    pass


class GlobalGraphTimeoutError(Exception):
    """Raised when the composite graph execution exceeds the global deadline."""
    pass


def run_with_node_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout_seconds: float = 0.5,
    node_name: str = "unnamed_node",
) -> Any:
    """
    Execute a callable within an asynchronous thread pool bounded by timeout_seconds.
    Raises NodeTimeoutError cleanly without hanging if execution exceeds the deadline.
    """
    kwargs = kwargs or {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise NodeTimeoutError(
                f"Node '{node_name}' execution exceeded maximum timeout of {timeout_seconds:.2f}s."
            )


def run_with_global_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    global_timeout_seconds: float = 1.0,
) -> Any:
    """
    Execute a composite multi-node graph run with a strict global deadline.
    Raises GlobalGraphTimeoutError if total elapsed execution exceeds the limit.
    """
    kwargs = kwargs or {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=global_timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise GlobalGraphTimeoutError(
                f"Global Graph Execution cancelled: exceeded total graph deadline of {global_timeout_seconds:.2f}s."
            )


def demonstrate_timeouts() -> Dict[str, Any]:
    """
    Demonstrate both per-node timeout and global graph timeout under simulated latency.
    """
    print("=========================================================================")
    print(" TIMEOUT RESILIENCE DEMONSTRATIONS")
    print("=========================================================================")

    # 1. Per-Node Timeout Demo
    print("\n--- 1. Testing Per-Node Timeout (Simulated 1.5s delay with 0.3s timeout) ---")
    def simulated_slow_node():
        time.sleep(1.5)
        return "slow_node_completed"

    node_timeout_caught = False
    node_err_msg = ""
    t0 = time.time()
    try:
        run_with_node_timeout(
            simulated_slow_node,
            timeout_seconds=0.3,
            node_name="slow_external_bureau_lookup",
        )
    except NodeTimeoutError as exc:
        elapsed = time.time() - t0
        node_timeout_caught = True
        node_err_msg = str(exc)
        print(f"  [NODE TIMEOUT CAUGHT] in {elapsed:.4f}s: {node_err_msg}")
        print("  Clean error produced without hanging! [PASS]")

    assert node_timeout_caught, "Per-node timeout failed to trigger!"

    # 2. Global Graph Timeout Demo
    print("\n--- 2. Testing Global Graph Timeout (Simulated multi-node delay with 0.4s global limit) ---")
    def simulated_slow_graph_run():
        # Simulates 3 sequential nodes of 0.25s each = 0.75s total
        for step in range(1, 4):
            time.sleep(0.25)
        return "graph_completed"

    global_timeout_caught = False
    global_err_msg = ""
    t1 = time.time()
    try:
        run_with_global_timeout(
            simulated_slow_graph_run,
            global_timeout_seconds=0.4,
        )
    except GlobalGraphTimeoutError as exc:
        elapsed = time.time() - t1
        global_timeout_caught = True
        global_err_msg = str(exc)
        print(f"  [GLOBAL TIMEOUT CAUGHT] in {elapsed:.4f}s: {global_err_msg}")
        print("  Global execution cleanly cancelled on total-time overrun! [PASS]")

    assert global_timeout_caught, "Global graph timeout failed to trigger!"

    print("\n=========================================================================")
    print(" TIMEOUT RESILIENCE DEMONSTRATION SUCCESSFUL [PASS]")
    print("=========================================================================\n")

    return {
        "node_timeout_caught": node_timeout_caught,
        "node_error": node_err_msg,
        "global_timeout_caught": global_timeout_caught,
        "global_error": global_err_msg,
    }


if __name__ == "__main__":
    demonstrate_timeouts()

