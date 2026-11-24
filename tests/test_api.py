from pathlib import Path

from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def create_session() -> str:
    """Create a checkout session and return its ID."""

    response = client.post("/sessions")

    assert response.status_code == 200

    return response.json()["session_id"]


def test_create_session():
    response = client.post("/sessions")

    assert response.status_code == 200

    data = response.json()

    assert "session_id" in data
    assert data["status"] == "created"


def test_get_basket():
    session_id = create_session()

    response = client.get(
        f"/sessions/{session_id}/basket"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["session_id"] == session_id
    assert data["total_unique_tracked_items"] == 0
    assert data["basket_summary"] == {}


def test_reset_basket():
    session_id = create_session()

    response = client.post(
        f"/sessions/{session_id}/basket/reset"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "basket reset successfully"


def test_sessions_are_isolated():
    session_a = create_session()
    session_b = create_session()

    response_a = client.get(
        f"/sessions/{session_a}/basket"
    )

    response_b = client.get(
        f"/sessions/{session_b}/basket"
    )

    assert response_a.status_code == 200
    assert response_b.status_code == 200

    assert response_a.json()["basket_summary"] == {}
    assert response_b.json()["basket_summary"] == {}


def test_detect_frame_does_not_affect_another_session():
    session_a = create_session()
    session_b = create_session()

    image_path = Path("assets/val_batch0_pred.jpg")

    assert image_path.exists(), "Sample image does not exist."

    with image_path.open("rb") as image_file:
        response = client.post(
            f"/sessions/{session_a}/detect-frame",
            files={
                "file": (
                    "sample.jpg",
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    detection_data = response.json()

    assert detection_data["session_id"] == session_a
    assert detection_data["active_detections_count"] > 0

    response_b = client.get(
        f"/sessions/{session_b}/basket"
    )

    assert response_b.status_code == 200

    basket_data = response_b.json()

    assert basket_data["total_unique_tracked_items"] == 0
    assert basket_data["basket_summary"] == {}


def test_delete_session():
    session_id = create_session()

    response = client.delete(
        f"/sessions/{session_id}"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "deleted"

    response = client.get(
        f"/sessions/{session_id}/basket"
    )

    assert response.status_code == 404


def test_unknown_session_returns_404():
    response = client.get(
        "/sessions/unknown-session-id/basket"
    )

    assert response.status_code == 404


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["classes_count"] > 0
    assert data["active_sessions"] >= 0
