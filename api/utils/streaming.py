"""Server-Sent Events primitives for CogniCore token streaming."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class StreamManager:
    """In-memory single-process token queue for SSE streaming."""

    queue: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue)

    async def publish(self, token: str, done: bool = False) -> None:
        """Publish a token event to waiting SSE clients."""
        await self.queue.put({"token": token, "done": done})

    async def events(self) -> str:
        """Return one SSE-formatted event from the queue."""
        payload = await self.queue.get()
        return f"data: {json.dumps(payload)}\n\n"


async def sse_generator(manager: StreamManager) -> AsyncIterator[str]:
    """Yield SSE events until a done event is observed."""
    while True:
        event = await manager.events()
        yield event
        if '"done": true' in event:
            break
