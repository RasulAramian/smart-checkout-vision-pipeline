from pathlib import Path
from unittest.mock import patch

import pytest

from src.basket_tracker import SharedYOLODetector
from src.session_manager import SessionManager, SessionNotFoundError


@pytest.fixture
def manager():
    detector = SharedYOLODetector(
        model_path=Path("models/best.pt")
    )

    return SessionManager(
        detector=detector,
        idle_timeout_seconds=30.0,
    )


def test_sessions_have_independent_trackers_and_baskets(manager):
    session_a_id = manager.create_session()
    session_b_id = manager.create_session()

    session_a = manager.get_session(session_a_id)
    session_b = manager.get_session(session_b_id)

    assert session_a is not None
    assert session_b is not None

    assert session_a.tracker is not session_b.tracker
    assert session_a.tracker.basket is not session_b.tracker.basket

    assert session_a.tracker.detector is session_b.tracker.detector

    session_a.tracker.basket.add_item(1, "Apple")

    assert session_a.tracker.basket.get_summary()["total_items"] == 1
    assert session_b.tracker.basket.get_summary()["total_items"] == 0

    assert manager.get_active_sessions_count() == 2


def test_delete_session(manager):
    session_id = manager.create_session()

    assert manager.delete_session(session_id) is True
    assert manager.get_session(session_id) is None


def test_delete_unknown_session(manager):
    assert manager.delete_session("unknown-session") is False


def test_expired_session_is_removed_on_access(manager):
    session_id = manager.create_session()

    with patch("src.session_manager.monotonic") as mock_time:
        mock_time.return_value = 100.0

        session = manager.get_session(session_id)
        assert session is not None

        mock_time.return_value = 131.0

        expired_session = manager.get_session(session_id)

    assert expired_session is None
    assert manager.get_active_sessions_count() == 0


def test_cleanup_expired_sessions(manager):
    session_a = manager.create_session()
    session_b = manager.create_session()

    with patch("src.session_manager.monotonic") as mock_time:
        mock_time.return_value = 100.0

        manager.get_session(session_a)
        manager.get_session(session_b)

        mock_time.return_value = 131.0

        removed_count = manager.cleanup_expired_sessions()

    assert removed_count == 2
    assert manager.get_active_sessions_count() == 0


def test_active_session_is_not_removed_by_cleanup(manager):
    session_id = manager.create_session()

    with patch("src.session_manager.monotonic") as mock_time:
        mock_time.return_value = 100.0

        with manager.use_session(session_id):
            mock_time.return_value = 200.0

            removed_count = manager.cleanup_expired_sessions()

            assert removed_count == 0
            assert manager.get_active_sessions_count() == 1

    assert manager.get_active_sessions_count() == 1


def test_use_session_updates_activity(manager):
    session_id = manager.create_session()

    with patch("src.session_manager.monotonic") as mock_time:
        mock_time.return_value = 100.0

        with manager.use_session(session_id) as session:
            assert session is not None
            assert session.active_requests == 1

        assert session.active_requests == 0
        assert session.last_activity == 100.0


def test_use_unknown_session_raises_error(manager):
    with pytest.raises(SessionNotFoundError):
        with manager.use_session("unknown-session"):
            pass


def test_session_cannot_be_deleted_while_in_use(manager):
    session_id = manager.create_session()

    with manager.use_session(session_id):
        assert manager.delete_session(session_id) is False

    assert manager.delete_session(session_id) is True


def test_invalid_idle_timeout():
    detector = SharedYOLODetector(
        model_path=Path("models/best.pt")
    )

    with pytest.raises(ValueError):
        SessionManager(
            detector=detector,
            idle_timeout_seconds=0,
        )