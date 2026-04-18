"""Pydantic request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response payload of `/health` — liveness plus whether the model is loaded."""

    status: str = "ok"
    model_loaded: bool
    version: str


class PredictionResponse(BaseModel):
    """Response payload of `/predict` — argmax class index plus full softmax probabilities."""

    pred: int = Field(..., description="Argmax class index")
    probs: list[float]
