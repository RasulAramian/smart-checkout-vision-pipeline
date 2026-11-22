from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from src.basket_manager import BasketManager


class SmartBasketTracker:
    """Detect and track products while managing an independent basket."""

    def __init__(
        self,
        model_path: str | Path = "models/best.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker_config = tracker_config

        self.basket = BasketManager()

    def reset_basket(self) -> None:
        """Reset basket contents and tracking memory."""
        self.basket.reset()

        predictor = getattr(self.model, "predictor", None)

        if predictor is not None:
            trackers = getattr(predictor, "trackers", [])

            for tracker in trackers:
                tracker.reset()

    def process_frame(
        self,
        frame: np.ndarray,
        persist: bool = True,
        draw: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Detect and track products in a single BGR frame."""

        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array")

        if frame.size == 0:
            raise ValueError("frame cannot be empty")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have three color channels")

        results = self.model.track(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker_config,
            persist=persist,
            verbose=False,
        )

        annotated = frame.copy()
        active_items: list[dict[str, Any]] = []

        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int)

            for box, conf, cls_id, track_id in zip(
                xyxy, confs, cls_ids, track_ids
            ):
                class_name = str(
                    self.model.names.get(cls_id, f"class_{cls_id}")
                )

                x1, y1, x2, y2 = map(int, box)

                self.basket.add_item(track_id, class_name)

                active_items.append(
                    {
                        "track_id": int(track_id),
                        "product": class_name,
                        "confidence": round(float(conf), 3),
                        "bbox": [x1, y1, x2, y2],
                    }
                )

                if draw:
                    color = (0, 200, 90)

                    cv2.rectangle(
                        annotated,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    label = f"ID:{track_id} {class_name} ({conf:.2f})"

                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )

        basket_summary = self.basket.get_summary()

        summary = {
            "active_detections_count": len(active_items),
            "active_items": active_items,
            "total_unique_tracked_items": basket_summary["total_items"],
            "basket_summary": basket_summary["basket_summary"],
        }

        return annotated, summary
