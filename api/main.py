"""FastAPI entry point for CogniCore."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.routes import chat, health
from api.utils.streaming import StreamManager

logger = logging.getLogger(__name__)


class RootResponse(BaseModel):
    """Human-readable API root response."""

    name: str
    status_endpoint: str
    docs_endpoint: str


def create_app() -> FastAPI:
    """Create and configure the CogniCore API application."""
    app = FastAPI(title="CogniCore API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("COGNICORE_CORS_ORIGINS", "http://localhost:5173").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.stream_manager = StreamManager()
    app.state.tokens_per_sec = 0.0
    app.state.model_loaded = False

    config_path = os.getenv("COGNICORE_CONFIG")
    checkpoint_path = os.getenv("COGNICORE_CHECKPOINT")
    if config_path and checkpoint_path:
        try:
            from cognicore.model.inference import CogniCoreInference

            app.state.inference = CogniCoreInference.from_paths(config_path, checkpoint_path)
            app.state.model_loaded = True
        except Exception as exc:
            logger.exception("Failed to load CogniCore inference engine: %s", exc)

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(chat.router, prefix="/api", tags=["chat"])

    @app.get("/", response_model=RootResponse)
    async def root() -> RootResponse:
        """Return API discovery metadata instead of a browser-facing 404."""
        return RootResponse(
            name="CogniCore API",
            status_endpoint="/api/status",
            docs_endpoint="/docs",
        )

    return app


app = create_app()
