from __future__ import annotations

import re

from .normalize import normalize_arguments, normalize_tool_name


_ARGUMENT_KEY_ALIASES = {
    "file_path": ("file_path", "path", "target_file", "filename", "file"),
    "content": ("content", "text", "body", "data", "file_content", "contents", "value"),
}


def _extract_loose_string_field(text: str, names: tuple[str, ...], *, multiline: bool = True) -> str | None:
    if not text:
        return None
    name_pattern = "|".join(re.escape(name) for name in names)
    match = re.search(
        rf"(?is)(?:\"({name_pattern})\"|'({name_pattern})'|({name_pattern}))\s*:\s*",
        text,
    )
    if not match:
        return None
    pos = match.end()
    tail = text[pos:].lstrip()
    if not tail:
        return ""

    if tail.startswith(('"""', "'''")):
        quote = tail[:3]
        end = tail.rfind(quote)
        if end > 0:
            return tail[3:end]
        return tail[3:]

    if tail[0] in {'"', "'"}:
        quote = tail[0]
        if not multiline:
            escaped = False
            for idx, ch in enumerate(tail[1:], start=1):
                if ch == "\\" and not escaped:
                    escaped = True
                    continue
                if ch == quote and not escaped:
                    return tail[1:idx]
                escaped = False
            return tail[1:].splitlines()[0].rstrip(",")
        end = tail.rfind(quote)
        if end > 0:
            return tail[1:end]
        return tail[1:]

    end_match = re.search(r"(?m)^\s*(?:\"?[A-Za-z_][A-Za-z0-9_]*\"?\s*:|[}\]])", tail)
    value = tail[: end_match.start()] if end_match else tail
    return value.strip().rstrip(",").rstrip("}").strip()


def _normalize_textkv_arguments(arguments: str | None) -> dict[str, object]:
    parsed = normalize_arguments(arguments)
    if not (isinstance(parsed, dict) and set(parsed.keys()) == {"value"} and parsed.get("value") == arguments):
        return parsed

    raw = arguments or ""
    loose: dict[str, object] = {}
    for canonical, aliases in _ARGUMENT_KEY_ALIASES.items():
        value = _extract_loose_string_field(raw, aliases, multiline=canonical == "content")
        if value is not None:
            loose[canonical] = value
    return loose or parsed


def parse_textkv_format(text: str, allowed_names: set[str]) -> list[dict[str, object]]:
    name = None
    arguments = None
    current_key = None
    values = {
        "name": [],
        "arguments": [],
    }

    key_aliases = {
        "function.name": "name",
        "name": "name",
        "tool": "name",
        "tool.name": "name",
        "tool_name": "name",
        "function.arguments": "arguments",
        "arguments": "arguments",
        "args": "arguments",
        "input": "arguments",
        "tool_input": "arguments",
        "parameters": "arguments",
    }

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            normalized_key = key.strip().lower()
            mapped_key = key_aliases.get(normalized_key)
            if mapped_key:
                current_key = mapped_key
                values[mapped_key].append(value.strip())
                continue
        if current_key:
            values[current_key].append(raw_line)

    if values["name"]:
        name = "\n".join(values["name"]).strip().splitlines()[0].strip().strip('"\'')
    if values["arguments"]:
        arguments = "\n".join(values["arguments"]).strip()

    if not name:
        return []

    return [{
        "name": normalize_tool_name(name, allowed_names),
        "input": _normalize_textkv_arguments(arguments),
    }]
