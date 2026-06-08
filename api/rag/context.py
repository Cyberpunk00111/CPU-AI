"""Prompt context construction for web-assisted CogniCore answers."""

from __future__ import annotations

from api.rag.search import SearchResult


def build_context_prompt(query: str, results: list[SearchResult]) -> str:
    """Build a compact prompt from a user query and search snippets."""
    if not results:
        return query
    sections = ["Use these web search snippets when helpful:"]
    for index, result in enumerate(results, start=1):
        sections.append(f"[{index}] {result.title}\nURL: {result.url}\nSnippet: {result.snippet}")
    sections.append(f"User query: {query}")
    return "\n\n".join(sections)
