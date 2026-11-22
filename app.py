from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.basket_tracker import SmartBasketTracker


MODEL_PATH = Path("models/best.pt")

app = FastAPI(
    title="AKNA Smart Checkout Vision API",
    description=(
        "Real-time product detection and ByteTrack basket tracking "
        "with YOLOv8 and OpenCV."
    ),
    version="1.0.0",
)

tracker = SmartBasketTracker(model_path=MODEL_PATH)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a BGR NumPy array."""

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)

    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file.",
        )

    return frame


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Return API and model health information."""

    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "classes_count": len(tracker.model.names),
    }


@app.get("/basket")
def get_basket_state() -> dict[str, Any]:
    """Return the current basket summary."""

    basket_summary = tracker.basket.get_summary()

    return {
        "total_unique_tracked_items": basket_summary["total_items"],
        "basket_summary": basket_summary["basket_summary"],
    }


@app.post("/basket/reset")
def reset_basket_state() -> dict[str, str]:
    """Reset the current basket."""

    tracker.reset_basket()

    return {
        "status": "basket reset successfully",
    }


@app.post("/detect-frame")
async def detect_frame(
    file: UploadFile = File(...),
    return_image: bool = False,
) -> JSONResponse:
    """Detect and track products in an uploaded image."""

    content = await file.read()

    frame = _decode_image(content)

    annotated_frame, summary = tracker.process_frame(
        frame=frame,
        persist=True,
        draw=True,
    )

    payload: dict[str, Any] = dict(summary)

    if return_image:
        success, encoded = cv2.imencode(".jpg", annotated_frame)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to encode annotated image.",
            )

        payload["annotated_image_base64"] = base64.b64encode(
            encoded.tobytes()
        ).decode("utf-8")

    return JSONResponse(content=payload)