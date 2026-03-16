from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class WorkingMemory:
    """Short-lived runtime memory for active goals and task state."""

    _session_state: dict[str, dict[str, Any]] = field(default_factory=dict)

    def set(self, session_id: str, key: str, value: Any) -> None:
        state = self._session_state.setdefault(session_id, {})
        state[key] = value

    def get(self, session_id: str, key: str, default: Any = None) -> Any:
        return self._session_state.get(session_id, {}).get(key, default)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        return dict(self._session_state.get(session_id, {}))

    def clear(self, session_id: str) -> None:
        self._session_state.pop(session_id, None)

