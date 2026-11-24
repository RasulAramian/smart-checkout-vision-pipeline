from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

from src.basket_tracker import (
    SharedYOLODetector,
    SmartBasketTracker,
)


@dataclass
class CheckoutSession:
    """Store independent tracking and basket state for one customer."""

    tracker: SmartBasketTracker
    lock: Lock


class SessionManager:
    """Create and manage isolated checkout sessions."""

    def __init__(self, detector: SharedYOLODetector) -> None:
        self.detector = detector
        self.sessions: dict[str, CheckoutSession] = {}
        self._lock = Lock()

    def create_session(self) -> str:
        """Create a new checkout session."""

        session_id = str(uuid4())

        session = CheckoutSession(
            tracker=SmartBasketTracker(detector=self.detector),
            lock=Lock(),
        )

        with self._lock:
            self.sessions[session_id] = session

        return session_id

    def get_session(self, session_id: str) -> CheckoutSession | None:
        """Retrieve an existing checkout session."""

        with self._lock:
            return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a checkout session."""

        with self._lock:
            if session_id not in self.sessions:
                return False

            del self.sessions[session_id]
            return True

    def get_active_sessions_count(self) -> int:
        """Return the number of active checkout sessions."""

        with self._lock:
            return len(self.sessions)
        