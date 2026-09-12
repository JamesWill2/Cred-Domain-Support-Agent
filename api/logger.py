"""
Structured JSONL Request Logger with Zero Raw PII Guarantee.

Logs each incoming request and response cycle as a single structured JSON-Lines entry.
Sanitizes all payload text through mask_pii to guarantee that raw PAN, Aadhaar,
or Bank Account numbers NEVER reach the disk.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import datetime
import threading
from agent.guardrails import mask_pii

DEFAULT_LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "agent_requests.jsonl"


class StructuredJSONLLogger:
    """Thread-safe structured JSONL logger with strict PII masking."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or DEFAULT_LOG_FILE
        self._lock = threading.Lock()
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                self.log_path.touch()

    def log_request(
        self,
        endpoint: str,
        trace_id: str,
        status_code: int,
        latency_ms: float,
        raw_request_payload: Any,
        response_payload: Any,
        client_ip: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Sanitize payloads and write a single JSON-Lines record to disk.

        Returns:
            The logged structured entry dictionary.
        """
        # Strictly sanitize request and response text to eliminate raw PII
        req_str = json.dumps(raw_request_payload) if not isinstance(raw_request_payload, str) else raw_request_payload
        sanitized_req_str, pii_meta = mask_pii(req_str)

        resp_str = json.dumps(response_payload) if not isinstance(response_payload, str) else response_payload
        sanitized_resp_str, _ = mask_pii(resp_str)

        entry: Dict[str, Any] = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "trace_id": trace_id,
            "endpoint": endpoint,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "client_ip": client_ip or "127.0.0.1",
            "request_payload_sanitized": sanitized_req_str,
            "response_payload_sanitized": sanitized_resp_str,
            "pii_masked": pii_meta.get("pii_detected", False),
            "pii_detection_counts": {
                "pan": pii_meta.get("pan_count", 0),
                "aadhaar": pii_meta.get("aadhaar_count", 0),
                "account": pii_meta.get("account_count", 0),
            },
            "metadata": extra_metadata or {},
        }

        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        return entry

    def read_logs(self) -> List[Dict[str, Any]]:
        """Read all JSONL log entries from disk."""
        with self._lock:
            if not self.log_path.exists():
                return []
            entries = []
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            continue
            return entries

    def clear_logs(self):
        """Truncate the log file."""
        with self._lock:
            if self.log_path.exists():
                self.log_path.write_text("", encoding="utf-8")


# Singleton logger instance
_jsonl_logger: Optional[StructuredJSONLLogger] = None


def get_jsonl_logger() -> StructuredJSONLLogger:
    """Get singleton StructuredJSONLLogger."""
    global _jsonl_logger
    if _jsonl_logger is None:
        _jsonl_logger = StructuredJSONLLogger()
    return _jsonl_logger
