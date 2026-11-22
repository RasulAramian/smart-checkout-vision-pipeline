import base64
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient


# Mock YOLO tracker before importing the FastAPI application.
with patch("src.basket_tracker.SmartBasketTracker") as mock_tracker_class:
    mock_tracker_class.return_value.basket.get_summary.return_value = {
        "total_items": 0,
        "basket_summary": {},
    }

    import app as app_module


client = TestClient(app_module.app)


def test_get_basket():
    response = client.get("/basket")

    assert response.status_code == 200

    data = response.json()

    assert data["total_unique_tracked_items"] == 0
    assert data["basket_summary"] == {}


def test_reset_basket():
    with patch.object(app_module.tracker, "reset_basket") as mock_reset:
        response = client.post("/basket/reset")

        assert response.status_code == 200

        assert response.json() == {
            "status": "basket reset successfully"
        }

        mock_reset.assert_called_once()


def test_detect_frame_with_invalid_image():
    response = client.post(
        "/detect-frame",
        files={
            "file": (
                "invalid.jpg",
                b"this is not a valid image",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Invalid or corrupted image file."
    }

    app_module.tracker.process_frame.assert_not_called()


def test_detect_frame_with_valid_image():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    success, encoded = cv2.imencode(".jpg", frame)

    assert success is True

    app_module.tracker.process_frame.return_value = (
        frame,
        {
            "active_detections_count": 1,
            "active_items": [
                {
                    "track_id": 1,
                    "product": "Coca-Cola",
                    "confidence": 0.95,
                    "bbox": [10, 10, 50, 50],
                }
            ],
            "total_unique_tracked_items": 1,
            "basket_summary": {"Coca-Cola": 1},
        },
    )

    response = client.post(
        "/detect-frame",
        params={"return_image": True},
        files={
            "file": (
                "sample.jpg",
                encoded.tobytes(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["active_detections_count"] == 1
    assert data["basket_summary"] == {"Coca-Cola": 1}

    assert "annotated_image_base64" in data
    assert isinstance(data["annotated_image_base64"], str)

    decoded_image = base64.b64decode(
        data["annotated_image_base64"]
    )

    decoded_frame = cv2.imdecode(
        np.frombuffer(decoded_image, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    assert decoded_frame is not None
    assert decoded_frame.shape == frame.shape

    app_module.tracker.process_frame.assert_called_once()