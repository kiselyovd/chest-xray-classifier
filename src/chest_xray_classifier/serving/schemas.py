"""Pydantic request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    version: str


class PredictionResponse(BaseModel):
    pred: int = Field(..., description="Argmax class index")
    probs: list[float]
