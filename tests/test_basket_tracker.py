from pathlib import Path
import cv2
import numpy as np
from src.basket_tracker import SmartBasketTracker


def test_tracker_on_sample_image():
    model_path = Path("models/best.pt")
    sample_path = Path("assets/val_batch0_pred.jpg")
    assert model_path.exists()
    assert sample_path.exists()

    frame = cv2.imread(str(sample_path))
    assert frame is not None and isinstance(frame, np.ndarray)

    tracker = SmartBasketTracker(model_path=model_path)
    annotated, summary = tracker.process_frame(frame)

    assert annotated.shape == frame.shape
    assert summary["active_detections_count"] > 0
    assert summary["total_unique_tracked_items"] == summary["active_detections_count"]
    assert isinstance(summary["basket_summary"], dict)

    tracker.reset_basket()
    assert tracker.basket_counts == {}
