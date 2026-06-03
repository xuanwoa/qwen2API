from __future__ import annotations

import hashlib
import logging
from typing import Any

from backend.core.config import settings
from backend.core.request_logging import get_request_context, update_request_context


TEST_MARKERS = (
    "TEST_DELETE_20260601",
    "TEST_LONG_INPUT_20260601",
)


def prompt_tail(text: str) -> str:
    tail_chars = max(0, int(getattr(settings, "TRACE_RESPONSE_TAIL_CHARS", 240) or 0))
    return text[-tail_chars:] if tail_chars else ""


def find_test_markers(text: str) -> list[str]:
    text = text or ""
    return [marker for marker in TEST_MARKERS if marker in text]


def set_trace_markers(markers: list[str] | tuple[str, ...] | None) -> None:
    marker_text = ",".join(markers or []) or "-"
    update_request_context(test_marker=marker_text)


def trace_context_fields() -> dict[str, Any]:
    ctx = get_request_context()
    return {
        "req_id": ctx.get("req_id", "-"),
        "surface": ctx.get("surface", "-"),
        "marker": ctx.get("test_marker", "-"),
        "chat_id": ctx.get("chat_id", "-"),
        "attempt": ctx.get("upstream_attempt", "-"),
    }


def log_test_prompt(
    log: logging.Logger,
    *,
    stage: str = "entry",
    surface: str,
    model: str,
    stream: bool,
    tools: list[Any],
    prompt: str,
) -> list[str]:
    markers = find_test_markers(prompt)
    if not markers:
        set_trace_markers([])
        return []
    set_trace_markers(markers)
    digest = hashlib.sha256((prompt or "").encode("utf-8", errors="replace")).hexdigest()
    log.info(
        "[TestTrace] stage=%s marker=%s surface=%s model=%s stream=%s tools=%s prompt_len=%s prompt_tail=%r prompt_sha256=%s",
        stage,
        ",".join(markers),
        surface,
        model,
        stream,
        tools,
        len(prompt or ""),
        prompt_tail(prompt or ""),
        digest,
    )
    return markers
