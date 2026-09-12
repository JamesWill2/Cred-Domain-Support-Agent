"""
JSON-Backed Persistent Conversation Memory for Cred Domain Support Agent.

Persists multi-turn dialog history and contextual session state (such as last referenced loan ID)
to a JSON file, enabling coreference resolution across conversation turns.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import threading

DEFAULT_MEMORY_FILE = Path(__file__).resolve().parent.parent / "conversations_memory.json"


class ConversationMemoryManager:
    """Manages thread-safe multi-turn conversation persistence using JSON storage."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or DEFAULT_MEMORY_FILE
        self._lock = threading.Lock()
        self._ensure_storage()

    def _ensure_storage(self):
        with self._lock:
            if not self.storage_path.exists():
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                self.storage_path.write_text("{}", encoding="utf-8")

    def _read_all(self) -> Dict[str, Any]:
        with self._lock:
            try:
                content = self.storage_path.read_text(encoding="utf-8")
                return json.loads(content) if content.strip() else {}
            except Exception:
                return {}

    def _write_all(self, data: Dict[str, Any]):
        with self._lock:
            self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Load session state and message history."""
        data = self._read_all()
        return data.get(session_id, {"messages": [], "context": {}})

    def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: Dict[str, Any],
        extracted_loan_id: Optional[str] = None,
    ):
        """Append user and assistant messages and update contextual state."""
        data = self._read_all()
        session = data.get(session_id, {"messages": [], "context": {}})

        session["messages"].append({"role": "user", "content": user_message})
        session["messages"].append(
            {
                "role": "assistant",
                "content": assistant_response.get("response", ""),
                "intent": assistant_response.get("intent", ""),
                "tool_used": assistant_response.get("tool_used"),
            }
        )

        # Update context tracking
        if extracted_loan_id:
            session["context"]["last_loan_id"] = extracted_loan_id
        if assistant_response.get("tool_used") == "check_loan_application_status":
            loan_id = assistant_response.get("sources", [None])[0] if assistant_response.get("sources") else None
            if loan_id:
                session["context"]["last_loan_id"] = loan_id

        data[session_id] = session
        self._write_all(data)

    def get_last_loan_id(self, session_id: str) -> Optional[str]:
        """Retrieve the last mentioned loan record ID in the session context."""
        session = self.get_session(session_id)
        return session.get("context", {}).get("last_loan_id")

    def clear_session(self, session_id: str):
        """Reset and remove session history from persistent storage."""
        data = self._read_all()
        if session_id in data:
            del data[session_id]
            self._write_all(data)

    def clear_all(self):
        """Wipe entire memory file."""
        self._write_all({})


# Singleton memory manager instance
_memory_manager: Optional[ConversationMemoryManager] = None


def get_memory_manager() -> ConversationMemoryManager:
    """Get singleton ConversationMemoryManager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = ConversationMemoryManager()
    return _memory_manager


if __name__ == "__main__":
    mem = get_memory_manager()
    sess = "demo-session-001"
    mem.clear_session(sess)

    print("Initial session state:", mem.get_session(sess))
    mem.save_turn(
        session_id=sess,
        user_message="What is the status of LOAN-1001?",
        assistant_response={"response": "LOAN-1001 is Submitted", "intent": "loan_status", "tool_used": "check_loan_application_status"},
        extracted_loan_id="LOAN-1001",
    )
    print("After Turn 1, remembered loan ID:", mem.get_last_loan_id(sess))

    fresh_sess = "demo-session-fresh"
    print("Fresh session remembered loan ID:", mem.get_last_loan_id(fresh_sess))
