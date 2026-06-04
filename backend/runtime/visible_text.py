from __future__ import annotations

import re
from typing import Any


_HIDDEN_START = "HIDDEN_INSTRUCTION_START"
_HIDDEN_END = "HIDDEN_INSTRUCTION_END"
_INTERNAL_MARKER_RE = re.compile(
    r"(?:<!--\s*)?(?:</?\s*)?(HIDDEN_INSTRUCTION_(?:START|END))(?:\s*(?:-->|/?>))?",
    re.IGNORECASE,
)
_PARTIAL_MARKER_PREFIXES = tuple(
    prefix.lower()
    for marker in (_HIDDEN_START, _HIDDEN_END)
    for prefix in (
        marker,
        f"<{marker}",
        f"</{marker}",
        f"<!--{marker}",
        f"<!-- {marker}",
    )
)


def internal_marker_partial_suffix_len(text: str) -> int:
    if not text:
        return 0
    lowered = text.lower()
    best = 0
    for prefix in _PARTIAL_MARKER_PREFIXES:
        limit = min(len(prefix) - 1, len(lowered))
        for length in range(1, limit + 1):
            if lowered.endswith(prefix[:length]):
                best = max(best, length)
    return best


class VisibleTextSanitizer:
    """Stateful sanitizer for streamed assistant-visible text."""

    def __init__(self) -> None:
        self._pending = ""
        self._inside_hidden_block = False

    def reset(self) -> None:
        self._pending = ""
        self._inside_hidden_block = False

    def feed(self, text: str | None) -> str:
        if not text:
            return ""
        combined = self._pending + text
        hold = internal_marker_partial_suffix_len(combined)
        if hold:
            process_text = combined[:-hold]
            self._pending = combined[-hold:]
        else:
            process_text = combined
            self._pending = ""
        return self._sanitize_complete_text(process_text)

    def flush(self) -> str:
        if not self._pending:
            return ""
        pending = self._pending
        self._pending = ""
        return self._sanitize_complete_text(pending)

    def _find_end_marker(self, text: str, pos: int) -> re.Match[str] | None:
        for match in _INTERNAL_MARKER_RE.finditer(text, pos):
            if match.group(1).upper() == _HIDDEN_END:
                return match
        return None

    def _sanitize_complete_text(self, text: str) -> str:
        if not text:
            return ""
        output: list[str] = []
        pos = 0
        while pos < len(text):
            if self._inside_hidden_block:
                end_match = self._find_end_marker(text, pos)
                if end_match is None:
                    return "".join(output)
                pos = end_match.end()
                self._inside_hidden_block = False
                continue

            match = _INTERNAL_MARKER_RE.search(text, pos)
            if match is None:
                output.append(text[pos:])
                break

            output.append(text[pos:match.start()])
            marker = match.group(1).upper()
            pos = match.end()
            if marker == _HIDDEN_START:
                self._inside_hidden_block = True
            # Standalone END markers are dropped.

        return "".join(output)


def sanitize_visible_text(text: str | None) -> str:
    if not text:
        return ""
    sanitizer = VisibleTextSanitizer()
    return sanitizer.feed(text) + sanitizer.flush()


def sanitize_visible_text_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized_blocks: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "text":
            sanitized_blocks.append(block)
            continue
        text = block.get("text")
        if not isinstance(text, str):
            sanitized_blocks.append(block)
            continue
        cleaned = sanitize_visible_text(text)
        if not cleaned:
            continue
        sanitized_block = dict(block)
        sanitized_block["text"] = cleaned
        sanitized_blocks.append(sanitized_block)
    return sanitized_blocks
