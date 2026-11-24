from pathlib import Path

from src.basket_tracker import SharedYOLODetector
from src.session_manager import SessionManager


def test_sessions_have_independent_trackers_and_baskets():
    detector = SharedYOLODetector(model_path=Path("models/best.pt"))
    manager = SessionManager(detector=detector)

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