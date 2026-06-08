"""Chat and streaming endpoints."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.rag.context import build_context_prompt
from api.rag.search import search_web
from api.utils.streaming import StreamManager, sse_generator

router = APIRouter()


class QueryRequest(BaseModel):
    """User query payload."""

    message: str = Field(min_length=1)
    use_web_search: bool = False
    max_tokens: int = Field(default=128, ge=1, le=2048)


class QueryResponse(BaseModel):
    """Acknowledgement returned after generation starts."""

    accepted: bool
    stream: str


@dataclass
class LocalResponder:
    """Deterministic fallback responder when trained weights are not loaded."""

    product_name: str = "CogniCore"

    def tokens(self, prompt: str, max_tokens: int) -> list[str]:
        """Return a compact fallback answer as token-like chunks."""
        answer = (
            f"{self.product_name} has accepted the query. A trained checkpoint is not loaded yet, "
            "so this development server is exercising the streaming path rather than claiming "
            f"model intelligence. Prompt received: {prompt[:240]}"
        )
        return answer.split(" ")[:max_tokens]


async def _generate_response(request: Request, payload: QueryRequest) -> None:
    """Generate and publish token events for the active stream."""
    manager: StreamManager = request.app.state.stream_manager
    start_time = time.monotonic()
    token_count = 0

    search_results = search_web(payload.message, max_results=3) if payload.use_web_search else []
    prompt = build_context_prompt(payload.message, search_results)
    inference = getattr(request.app.state, "inference", None)

    if inference is not None:
        token_iterable = inference.stream(prompt, payload.max_tokens, temperature=0.8, top_k=50)
    else:
        token_iterable = LocalResponder().tokens(prompt, payload.max_tokens)

    try:
        for token in token_iterable:
            token_count += 1
            await manager.publish(f"{token} ")
            elapsed = max(time.monotonic() - start_time, 1.0e-6)
            request.app.state.tokens_per_sec = token_count / elapsed
            await asyncio.sleep(0)
    finally:
        await manager.publish("", done=True)


@router.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> QueryResponse:
    """Start generation for a user query."""
    background_tasks.add_task(_generate_response, request, payload)
    return QueryResponse(accepted=True, stream="/api/stream")


@router.get("/stream")
async def stream(request: Request) -> StreamingResponse:
    """Stream token events as Server-Sent Events."""
    manager: StreamManager = request.app.state.stream_manager
    return StreamingResponse(sse_generator(manager), media_type="text/event-stream")
