"""DuckDuckGo search wrapper used for optional web-assisted answers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """Normalised web search result."""

    title: str
    url: str
    snippet: str


def search_web(query: str, max_results: int) -> list[SearchResult]:
    """Run a DuckDuckGo web search.

    Args:
        query: User query.
        max_results: Maximum number of results to return.

    Returns:
        Normalised search results. Returns an empty list if the optional search
        dependency is unavailable or the network request fails.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("duckduckgo-search is not installed; continuing without web results")
        return []

    try:
        with DDGS() as ddgs:
            raw_results = ddgs.text(query, max_results=max_results)
            return [
                SearchResult(
                    title=str(result.get("title", "")),
                    url=str(result.get("href", "")),
                    snippet=str(result.get("body", "")),
                )
                for result in raw_results
            ]
    except Exception as exc:
        logger.warning("DuckDuckGo search failed for query %r: %s", query, exc)
        return []
