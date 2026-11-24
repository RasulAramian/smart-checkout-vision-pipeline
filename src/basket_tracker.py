from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils.checks import check_yaml

from src.basket_manager import BasketManager


class SharedYOLODetector:
    """Share one YOLO model across multiple independent sessions."""

    def __init__(
        self,
        model_path: str | Path = "models/best.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self._inference_lock = Lock()

    def detect(self, frame: np.ndarray) -> Any:
        """Run YOLO detection safely using the shared model."""

        with self._inference_lock:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )

        return results[0]


class SmartBasketTracker:
    """Manage independent tracking and basket state for one session."""

    def __init__(
        self,
        model_path: str | Path = "models/best.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        tracker_config: str = "bytetrack.yaml",
        detector: SharedYOLODetector | None = None,
    ) -> None:
        if detector is None:
            detector = SharedYOLODetector(
                model_path=model_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

        self.detector = detector
        self.model = detector.model

        config_path = check_yaml(tracker_config)
        tracker_args = get_cfg(config_path)

        self.tracker = BYTETracker(tracker_args)

        self.basket = BasketManager()

    def reset_basket(self) -> None:
        """Reset basket contents and tracking memory."""

        self.basket.reset()
        self.tracker.reset()

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

        result = self.detector.detect(frame)

        detections = result.boxes

        annotated = frame.copy()
        active_items: list[dict[str, Any]] = []

        if detections is not None and len(detections) > 0:
            detections = detections.cpu().numpy()

            tracks = self.tracker.update(detections)

            for track in tracks:
                x1, y1, x2, y2 = map(int, track[:4])

                track_id = int(track[4])
                confidence = float(track[5])
                class_id = int(track[6])
                detection_index = int(track[7])

                class_name = str(
                    self.model.names.get(
                        class_id,
                        f"class_{class_id}",
                    )
                )

                self.basket.add_item(track_id, class_name)

                active_items.append(
                    {
                        "track_id": track_id,
                        "product": class_name,
                        "confidence": round(confidence, 3),
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

                    label = (
                        f"ID:{track_id} "
                        f"{class_name} ({confidence:.2f})"
                    )

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
