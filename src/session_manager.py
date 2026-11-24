from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from typing import Iterator
from uuid import uuid4

from src.basket_tracker import (
    SharedYOLODetector,
    SmartBasketTracker,
)


class SessionNotFoundError(Exception):
    """Raised when a checkout session does not exist or has expired."""


@dataclass
class CheckoutSession:
    """Store independent tracking and basket state for one customer."""

    tracker: SmartBasketTracker
    lock: Lock = field(default_factory=Lock)
    last_activity: float = field(default_factory=monotonic)
    active_requests: int = 0


class SessionManager:
    """Create and manage isolated checkout sessions."""

    def __init__(
        self,
        detector: SharedYOLODetector,
        idle_timeout_seconds: float = 1800.0,
    ) -> None:
        if idle_timeout_seconds <= 0:
            raise ValueError("idle_timeout_seconds must be positive")

        self.detector = detector
        self.idle_timeout_seconds = idle_timeout_seconds

        self.sessions: dict[str, CheckoutSession] = {}
        self._lock = Lock()

    def create_session(self) -> str:
        """Create a new checkout session."""

        session_id = str(uuid4())

        session = CheckoutSession(
            tracker=SmartBasketTracker(detector=self.detector),
        )

        with self._lock:
            self.sessions[session_id] = session

        return session_id

    def get_session(self, session_id: str) -> CheckoutSession | None:
        """Retrieve an existing session and update its activity."""

        with self._lock:
            session = self.sessions.get(session_id)

            if session is None:
                return None

            now = monotonic()

            if (
                session.active_requests == 0
                and now - session.last_activity >= self.idle_timeout_seconds
            ):
                del self.sessions[session_id]
                return None

            session.last_activity = now

            return session

    @contextmanager
    def use_session(
        self,
        session_id: str,
    ) -> Iterator[CheckoutSession]:
        """Acquire a session safely while a request is using it."""

        with self._lock:
            session = self.sessions.get(session_id)

            if session is None:
                raise SessionNotFoundError("Session not found")

            now = monotonic()

            if (
                session.active_requests == 0
                and now - session.last_activity >= self.idle_timeout_seconds
            ):
                del self.sessions[session_id]
                raise SessionNotFoundError("Session has expired")

            session.active_requests += 1
            session.last_activity = now

        try:
            with session.lock:
                yield session

        finally:
            with self._lock:
                session.active_requests -= 1
                session.last_activity = monotonic()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session only when it has no active requests."""

        with self._lock:
            session = self.sessions.get(session_id)

            if session is None:
                return False

            if session.active_requests > 0:
                return False

            del self.sessions[session_id]

            return True

    def cleanup_expired_sessions(self) -> int:
        """Remove inactive sessions whose idle timeout has elapsed."""

        now = monotonic()

        with self._lock:
            expired_ids = [
                session_id
                for session_id, session in self.sessions.items()
                if (
                    session.active_requests == 0
                    and now - session.last_activity
                    >= self.idle_timeout_seconds
                )
            ]

            for session_id in expired_ids:
                del self.sessions[session_id]

            return len(expired_ids)

    def get_active_sessions_count(self) -> int:
        """Return the number of registered sessions."""

        with self._lock:
            return len(self.sessions)