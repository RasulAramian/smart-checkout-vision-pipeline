from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


class SmartBasketTracker:
    """Detects shelf products with YOLOv8 and tracks them across frames via ByteTrack."""

    def __init__(
        self,
        model_path: str | Path = "models/best.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker_config = tracker_config

        # Tracked items history: track_id -> product class name
        self.seen_tracks: dict[int, str] = {}
        self.basket_counts: Counter[str] = Counter()

    def reset_basket(self) -> None:
        """Reset tracked basket state and reset tracker memory."""
        self.seen_tracks.clear()
        self.basket_counts.clear()
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            if hasattr(self.model.predictor, "trackers"):
                for tracker in self.model.predictor.trackers:
                    tracker.reset()

    def process_frame(
        self,
        frame: np.ndarray,
        persist: bool = True,
        draw: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run detection + ByteTrack on a single BGR frame and return annotated image + summary."""
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

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            track_ids = (
                boxes.id.cpu().numpy().astype(int)
                if boxes.id is not None
                else np.arange(1, len(boxes) + 1)
            )

            for box, conf, cls_id, track_id in zip(xyxy, confs, cls_ids, track_ids):
                class_name = str(self.model.names.get(cls_id, f"class_{cls_id}"))
                x1, y1, x2, y2 = map(int, box)

                # Register each unique track ID once in the basket summary
                if int(track_id) not in self.seen_tracks:
                    self.seen_tracks[int(track_id)] = class_name
                    self.basket_counts[class_name] += 1

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
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{int(track_id)} {class_name} ({conf:.2f})"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                    )
                    top_y = max(y1 - th - 10, 0)
                    cv2.rectangle(
                        annotated,
                        (x1, top_y),
                        (x1 + tw + 8, top_y + th + 6),
                        color,
                        -1,
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (x1 + 4, top_y + th + 1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )

        summary = {
            "active_detections_count": len(active_items),
            "active_items": active_items,
            "total_unique_tracked_items": sum(self.basket_counts.values()),
            "basket_summary": dict(self.basket_counts),
        }
        return annotated, summary
