"""Exception types and handlers."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class ModelNotLoadedError(RuntimeError):
    """Raised when the checkpoint has not been loaded into the running app yet."""


class InferenceError(RuntimeError):
    """Raised on irrecoverable errors during forward-pass (corrupt image, CUDA OOM)."""


async def inference_error_handler(request: Request, exc: InferenceError) -> JSONResponse:
    """FastAPI handler: map `InferenceError` to a 503 with a structured payload."""
    return JSONResponse(
        status_code=503,
        content={
            "error": "inference_failed",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError) -> JSONResponse:
    """FastAPI handler: map `ModelNotLoadedError` to a 503 `model_not_ready` response."""
    return JSONResponse(
        status_code=503,
        content={
            "error": "model_not_ready",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None),
        },
    )
