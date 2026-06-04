from __future__ import annotations

import time
from typing import Any, Iterable


OPENAI_MODEL_CREATED_FALLBACK = 1_700_000_000

CHAT_TYPE_CAPABILITIES = {
    "deep_research": "deep_research",
    "t2i": "image_gen",
    "image_gen": "image_gen",
    "t2v": "video_gen",
    "web_dev": "web_dev",
    "slides": "slides",
}

MODEL_VARIANTS: tuple[tuple[str, str, str, dict[str, bool]], ...] = (
    ("thinking", "-thinking", "thinking", {"thinking": True}),
    ("search", "-search", "search", {"search": True}),
    ("deep_research", "-deep-research", "deep_research", {"deep_research": True, "search": True}),
    ("image_gen", "-image", "image", {"image_gen": True}),
    ("video_gen", "-video", "video", {"video_gen": True}),
    ("web_dev", "-webdev", "webdev", {"web_dev": True}),
    ("slides", "-slides", "slides", {"slides": True}),
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_chat_types(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Iterable):
        return {str(item) for item in value if item}
    return set()


def _first_text(*values: Any, default: str = "") -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _model_id(item: dict[str, Any]) -> str:
    return _first_text(item.get("id"), item.get("model"), item.get("name"))


def _created_at(item: dict[str, Any]) -> int:
    value = item.get("created_at") or item.get("created") or item.get("createdAt")
    try:
        return int(value)
    except (TypeError, ValueError):
        return OPENAI_MODEL_CREATED_FALLBACK


def _derive_family(model_id: str, item: dict[str, Any], meta: dict[str, Any]) -> str:
    explicit = _first_text(item.get("family"), meta.get("family"), default="")
    if explicit:
        return explicit
    if model_id.startswith("qwen3."):
        parts = model_id.split("-", 1)[0].split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])
    return model_id.split("-", 1)[0] if "-" in model_id else model_id


def extract_model_capabilities(item: dict[str, Any]) -> dict[str, bool]:
    info = _as_dict(item.get("info"))
    meta = _as_dict(info.get("meta") or item.get("meta"))
    raw_caps = _as_dict(meta.get("capabilities") or item.get("capabilities"))
    chat_types = _as_chat_types(meta.get("chat_type") or item.get("chat_type"))

    capabilities = {
        "thinking": bool(raw_caps.get("thinking")),
        "search": bool(raw_caps.get("search")),
        "vision": bool(raw_caps.get("vision")),
        "deep_research": False,
        "image_gen": False,
        "video_gen": False,
        "web_dev": False,
        "slides": False,
    }
    for chat_type in chat_types:
        key = CHAT_TYPE_CAPABILITIES.get(chat_type)
        if key:
            capabilities[key] = True
    return capabilities


def build_model_entry(
    *,
    model_id: str,
    base_model: str | None = None,
    capabilities: dict[str, bool] | None = None,
    mode: str = "chat",
    display_name: str | None = None,
    family: str | None = None,
    created: int | None = None,
    owned_by: str = "qwen",
) -> dict[str, Any]:
    base = base_model or model_id
    return {
        "id": model_id,
        "object": "model",
        "created": int(created or OPENAI_MODEL_CREATED_FALLBACK),
        "owned_by": owned_by or "qwen",
        "capabilities": capabilities or {},
        "base_model": base,
        "mode": mode,
        "display_name": display_name or model_id,
        "family": family or base,
    }


def build_openai_model_list(upstream_models: list[dict[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    data: list[dict[str, Any]] = []

    def add(entry: dict[str, Any]) -> None:
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            return
        seen.add(model_id)
        data.append(entry)

    for raw_item in upstream_models:
        if not isinstance(raw_item, dict):
            continue
        model_id = _model_id(raw_item)
        if not model_id:
            continue
        info = _as_dict(raw_item.get("info"))
        meta = _as_dict(info.get("meta") or raw_item.get("meta"))
        capabilities = extract_model_capabilities(raw_item)
        display_name = _first_text(
            raw_item.get("display_name"),
            raw_item.get("displayName"),
            raw_item.get("name"),
            meta.get("display_name"),
            meta.get("name"),
            default=model_id,
        )
        family = _derive_family(model_id, raw_item, meta)
        created = _created_at(raw_item)
        owned_by = _first_text(raw_item.get("owned_by"), raw_item.get("owner"), default="qwen")

        add(build_model_entry(
            model_id=model_id,
            base_model=model_id,
            capabilities=capabilities,
            mode="chat",
            display_name=display_name,
            family=family,
            created=created,
            owned_by=owned_by,
        ))

        for capability_key, suffix, mode, variant_capabilities in MODEL_VARIANTS:
            if capabilities.get(capability_key) or capability_key == "search":
                add(build_model_entry(
                    model_id=f"{model_id}{suffix}",
                    base_model=model_id,
                    capabilities=variant_capabilities,
                    mode=mode,
                    display_name=f"{display_name} {mode}",
                    family=family,
                    created=created,
                    owned_by=owned_by,
                ))

    return {"object": "list", "data": data}


def build_fallback_model_list(model_map: dict[str, str]) -> dict[str, Any]:
    now = int(time.time())
    seen: set[str] = set()
    data: list[dict[str, Any]] = []
    for model_id, resolved in model_map.items():
        if model_id in seen:
            continue
        seen.add(model_id)
        data.append(build_model_entry(
            model_id=model_id,
            base_model=resolved or model_id,
            capabilities={},
            mode="chat",
            display_name=model_id,
            family=(resolved or model_id).split("-", 1)[0],
            created=now,
            owned_by="qwen2api",
        ))
    return {"object": "list", "data": data}
