from __future__ import annotations

import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.basket_tracker import SharedYOLODetector
from src.session_manager import SessionManager, SessionNotFoundError


MODEL_PATH = Path("models/best.pt")
CLEANUP_INTERVAL_SECONDS = 60

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


detector = SharedYOLODetector(model_path=MODEL_PATH)
session_manager = SessionManager(detector=detector)


async def _periodic_session_cleanup() -> None:
    """Periodically remove expired checkout sessions."""

    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)

        try:
            removed_count = session_manager.cleanup_expired_sessions()

            if removed_count > 0:
                logger.info(
                    "Session cleanup completed: %s expired sessions removed.",
                    removed_count,
                )

        except Exception:
            logger.exception("Session cleanup failed.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage background tasks during the application lifecycle."""

    cleanup_task = asyncio.create_task(
        _periodic_session_cleanup()
    )

    logger.info("Session cleanup background task started.")

    try:
        yield

    finally:
        cleanup_task.cancel()

        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

        logger.info("Session cleanup background task stopped.")


app = FastAPI(
    title="AKNA Smart Checkout Vision API",
    description=(
        "Real-time product detection and session-based "
        "ByteTrack basket tracking with YOLOv8 and OpenCV."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


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


def _session_not_found() -> HTTPException:
    """Create a consistent 404 response for missing sessions."""

    return HTTPException(
        status_code=404,
        detail="Session not found or expired.",
    )


def _process_frame_for_session(
    session_id: str,
    frame: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Process a frame while safely holding the session lock."""

    with session_manager.use_session(session_id) as session:
        return session.tracker.process_frame(
            frame=frame,
            persist=True,
            draw=True,
        )


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Return API and model health information."""

    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "classes_count": len(detector.model.names),
        "active_sessions": session_manager.get_active_sessions_count(),
    }


@app.post("/sessions")
def create_session() -> dict[str, str]:
    """Create a new checkout session."""

    session_id = session_manager.create_session()

    return {
        "session_id": session_id,
        "status": "created",
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    """Delete an existing checkout session."""

    session = session_manager.get_session(session_id)

    if session is None:
        raise _session_not_found()

    deleted = session_manager.delete_session(session_id)

    if not deleted:
        raise HTTPException(
            status_code=409,
            detail="Session cannot be deleted while requests are active.",
        )

    return {
        "session_id": session_id,
        "status": "deleted",
    }


@app.get("/sessions/{session_id}/basket")
def get_basket_state(session_id: str) -> dict[str, Any]:
    """Return the basket summary for one session."""

    try:
        with session_manager.use_session(session_id) as session:
            basket_summary = session.tracker.basket.get_summary()

    except SessionNotFoundError:
        raise _session_not_found()

    return {
        "session_id": session_id,
        "total_unique_tracked_items": basket_summary["total_items"],
        "basket_summary": basket_summary["basket_summary"],
    }


@app.post("/sessions/{session_id}/basket/reset")
def reset_basket_state(session_id: str) -> dict[str, str]:
    """Reset the basket and tracking state for one session."""

    try:
        with session_manager.use_session(session_id) as session:
            session.tracker.reset_basket()

    except SessionNotFoundError:
        raise _session_not_found()

    return {
        "session_id": session_id,
        "status": "basket reset successfully",
    }


@app.post("/sessions/{session_id}/detect-frame")
async def detect_frame(
    session_id: str,
    file: UploadFile = File(...),
    return_image: bool = False,
) -> JSONResponse:
    """Detect and track products within one checkout session."""

    content = await file.read()

    frame = _decode_image(content)

    try:
        annotated_frame, summary = await run_in_threadpool(
            _process_frame_for_session,
            session_id,
            frame,
        )

    except SessionNotFoundError:
        raise _session_not_found()

    payload: dict[str, Any] = {
        "session_id": session_id,
        **summary,
    }

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