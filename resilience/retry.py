"""
Exponential-Backoff Retry Engine for Cred Domain Support Agent.

Parameters:
- max_attempts: 3
- initial_interval: 0.1s
- max_interval: 1.0s
- jitter: True (randomized uniform variation preventing thundering herds)

Includes a deterministic TransientServiceSimulator that deliberately fails the first
two calls and succeeds on the third attempt, proving automated self-healing.
"""

from typing import Callable, Any, Dict, Optional
import time
import random
import functools


class TransientNetworkError(Exception):
    """Simulated transient 503 / network timeout error."""
    pass


def exponential_backoff_retry(
    max_attempts: int = 3,
    initial_interval: float = 0.05,
    max_interval: float = 0.5,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (TransientNetworkError,),
):
    """
    Decorator implementing exponential backoff with jitter.

    Delay formula for attempt i (0-indexed):
        interval = min(initial_interval * (backoff_factor ** i), max_interval)
        if jitter:
            interval += random.uniform(0, 0.05)
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    if attempt >= max_attempts:
                        print(f"  [RETRY EXHAUSTED] Failed after {attempt} attempts: {exc}")
                        raise

                    # Compute backoff interval
                    delay = min(initial_interval * (backoff_factor ** (attempt - 1)), max_interval)
                    if jitter:
                        delay += random.uniform(0.005, 0.02)

                    print(
                        f"  [RETRY ATTEMPT {attempt}/{max_attempts}] Caught transient error: '{exc}'. "
                        f"Backing off for {delay:.4f}s before retry..."
                    )
                    time.sleep(delay)

        return wrapper
    return decorator


class TransientServiceSimulator:
    """
    Deterministic simulated microservice that fails exactly the first N calls
    with a transient network error, and succeeds on subsequent attempts.
    """

    def __init__(self, failures_before_success: int = 2):
        self.failures_before_success = failures_before_success
        self.call_count = 0
        self.history = []

    def execute_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Unprotected raw call tracking attempts."""
        self.call_count += 1
        attempt = self.call_count

        if attempt <= self.failures_before_success:
            err_msg = f"Simulated Transient Error: HTTP 503 Service Unavailable (Attempt {attempt})"
            self.history.append({"attempt": attempt, "status": "failed", "error": err_msg})
            raise TransientNetworkError(err_msg)

        success_res = {
            "status": "success",
            "attempt": attempt,
            "data": payload,
            "message": f"Service successfully responded on attempt {attempt} after {self.failures_before_success} prior transient failures.",
        }
        self.history.append({"attempt": attempt, "status": "success", "result": success_res})
        return success_res


def demonstrate_retry_recovery() -> Dict[str, Any]:
    """
    Demonstrates exponential backoff recovering a transient service failure.
    """
    print("=========================================================================")
    print(" EXPONENTIAL BACKOFF RETRY DEMONSTRATION")
    print(" Configuration: max_attempts=3, initial_interval=0.05s, max_interval=0.5s, jitter=True")
    print("=========================================================================")

    simulator = TransientServiceSimulator(failures_before_success=2)

    @exponential_backoff_retry(
        max_attempts=3,
        initial_interval=0.05,
        max_interval=0.5,
        jitter=True,
    )
    def call_simulated_service(req: Dict[str, Any]) -> Dict[str, Any]:
        return simulator.execute_call(req)

    print("Initiating call to transient service (programmed to fail calls 1 and 2)...")
    t0 = time.time()
    result = call_simulated_service({"record_id": "LOAN-1001", "action": "verify_bureau"})
    duration = time.time() - t0

    print(f"\nResult: {result['message']}")
    print(f"Total Attempts Made: {simulator.call_count}")
    print(f"Total Duration (including backoff delays): {duration:.4f}s")
    print("=========================================================================")
    print(" RETRY RECOVERY DEMONSTRATION SUCCESSFUL [PASS]")
    print("=========================================================================\n")

    assert simulator.call_count == 3, f"Expected 3 attempts, got {simulator.call_count}"
    assert result["status"] == "success"

    return {
        "call_count": simulator.call_count,
        "duration_s": round(duration, 4),
        "final_result": result,
    }


if __name__ == "__main__":
    demonstrate_retry_recovery()

