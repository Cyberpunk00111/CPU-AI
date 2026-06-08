"""Health and status endpoints."""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()
START_TIME = time.monotonic()


class HealthResponse(BaseModel):
    """API health response body."""

    ok: bool
    uptime_seconds: float


class StatusResponse(BaseModel):
    """Runtime status response body for the frontend status bar."""

    cpu_percent: float
    ram_mb: float
    tokens_per_sec: float
    model_loaded: bool


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return API liveness."""
    return HealthResponse(ok=True, uptime_seconds=time.monotonic() - START_TIME)


@router.get("/status", response_model=StatusResponse)
async def status(request: Request) -> StatusResponse:
    """Return lightweight runtime metrics."""
    cpu_percent = 0.0
    ram_mb = 0.0
    try:
        import psutil

        process = psutil.Process(os.getpid())
        cpu_percent = float(psutil.cpu_percent(interval=None))
        ram_mb = float(process.memory_info().rss / (1024 * 1024))
    except ImportError:
        ram_mb = 0.0

    return StatusResponse(
        cpu_percent=cpu_percent,
        ram_mb=ram_mb,
        tokens_per_sec=float(getattr(request.app.state, "tokens_per_sec", 0.0)),
        model_loaded=bool(getattr(request.app.state, "model_loaded", False)),
    )
