from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelMode:
    requested_model: str
    base_model: str
    chat_type: str = "t2t"
    force_thinking: bool = False
    mode: str = "chat"


MODEL_MODE_SUFFIXES: tuple[tuple[str, str, bool, str], ...] = (
    ("-deep-research", "deep_research", False, "deep_research"),
    ("-deep_research", "deep_research", False, "deep_research"),
    ("-web-dev", "web_dev", False, "webdev"),
    ("-thinking", "t2t", True, "thinking"),
    ("-search", "t2t", False, "search"),
    ("-webdev", "web_dev", False, "webdev"),
    ("-image", "t2i", False, "image"),
    ("-video", "t2v", False, "video"),
    ("-slides", "slides", False, "slides"),
    ("-t2i", "t2i", False, "image"),
    ("-t2v", "t2v", False, "video"),
)


def parse_model_mode(model_id: str | None, *, default_model: str = "") -> ModelMode:
    requested = str(model_id or default_model or "").strip()
    lowered = requested.lower()
    for suffix, chat_type, force_thinking, mode in MODEL_MODE_SUFFIXES:
        if lowered.endswith(suffix):
            base_model = requested[: -len(suffix)]
            return ModelMode(
                requested_model=requested,
                base_model=base_model,
                chat_type=chat_type,
                force_thinking=force_thinking,
                mode=mode,
            )
    return ModelMode(requested_model=requested, base_model=requested)


def strip_model_mode_suffix(model_id: str | None) -> str:
    return parse_model_mode(model_id).base_model
