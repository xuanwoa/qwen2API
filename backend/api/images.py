"""
图片生成接口 — 兼容 OpenAI /v1/images/generations 规范。

底层通过现有直连 HTTP 聊天能力触发千问“生成图像”模式，
不依赖浏览器运行时。
"""
import re
import time
import json
import logging
from typing import Any
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from backend.services.qwen_client import QwenClient

log = logging.getLogger("qwen2api.images")
router = APIRouter()

DEFAULT_IMAGE_MODEL = "qwen3.6-plus"

IMAGE_MODEL_MAP = {
    "dall-e-3": "qwen3.6-plus",
    "dall-e-2": "qwen3.6-plus",
    "qwen-image": "qwen3.6-plus",
    "qwen-image-plus": "qwen3.6-plus",
    "qwen-image-turbo": "qwen3.6-plus",
    "qwen3.6-plus": "qwen3.6-plus",
}

SUPPORTED_IMAGE_SIZES = {
    "1328x1328": "1:1",
    "1664x928": "16:9",
    "928x1664": "9:16",
    "1472x1140": "4:3",
    "1140x1472": "3:4",
}

IMAGE_RATIO_TO_SIZE = {ratio: size for size, ratio in SUPPORTED_IMAGE_SIZES.items()}


IMAGE_URL_KEYS = {
    "url",
    "image",
    "src",
    "imageUrl",
    "image_url",
    "imageURL",
    "preview_url",
    "previewUrl",
    "download_url",
    "downloadUrl",
    "origin_url",
    "originUrl",
    "oss_url",
    "ossUrl",
    "signed_url",
    "signedUrl",
}


def _looks_like_image_url(value: str) -> bool:
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        return False
    lowered = value.lower()
    image_hosts = ("cdn.qwenlm.ai", "wanx.alicdn.com", "img.alicdn.com", "alicdn.com")
    if any(host in lowered for host in image_hosts):
        return True
    return bool(re.search(r"\.(?:jpg|jpeg|png|webp|gif)(?:[?#][^\s\"'<>]*)?$", lowered))


def _collect_image_urls_from_obj(value: Any, urls: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and (key in IMAGE_URL_KEYS or _looks_like_image_url(item)):
                if _looks_like_image_url(item):
                    urls.append(item)
            else:
                _collect_image_urls_from_obj(item, urls)
        return
    if isinstance(value, list):
        for item in value:
            _collect_image_urls_from_obj(item, urls)


def _extract_image_urls(text: str) -> list[str]:
    urls: list[str] = []

    for u in re.findall(r'!\[.*?\]\((https?://[^\s\)]+)\)', text):
        urls.append(u.rstrip(").,;"))

    for u in re.findall(r'"(?:url|image|src|imageUrl|image_url)"\s*:\s*"(https?://[^"]+)"', text):
        urls.append(u)

    cdn_pattern = r'https?://(?:cdn\.qwenlm\.ai|wanx\.alicdn\.com|img\.alicdn\.com|[^\s"<>]+\.(?:jpg|jpeg|png|webp|gif))(?:[^\s"<>]*)'
    for u in re.findall(cdn_pattern, text, re.IGNORECASE):
        urls.append(u.rstrip(".,;)\"'>"))

    for match in re.finditer(r"[\{\[]", text):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[match.start():])
        except Exception:
            continue
        _collect_image_urls_from_obj(obj, urls)

    seen: set[str] = set()
    result: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _extract_upstream_failure(text: str) -> str | None:
    for match in re.finditer(r"\{", text or ""):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[match.start():])
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        request_id = obj.get("request_id") or obj.get("response_id") or "-"
        if obj.get("success") is False:
            data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
            code = data.get("code") or obj.get("code") or "upstream_error"
            details = data.get("details") or data.get("message") or obj.get("details") or obj.get("message") or ""
            return f"Qwen upstream error code={code} request_id={request_id} details={details}"
        error = obj.get("error")
        if isinstance(error, dict):
            code = error.get("code") or "upstream_error"
            details = error.get("details") or error.get("message") or error.get("type") or ""
            return f"Qwen upstream error code={code} request_id={request_id} details={details}"
        if isinstance(error, str) and error:
            return f"Qwen upstream error request_id={request_id} details={error}"
    return None


def _resolve_image_model(requested: str | None) -> str:
    if not requested:
        return DEFAULT_IMAGE_MODEL
    return IMAGE_MODEL_MAP.get(requested, DEFAULT_IMAGE_MODEL)


def _normalize_image_size(value: str | None) -> tuple[str, str, int, int]:
    requested = (value or "").strip().lower().replace("*", "x").replace("×", "x")
    if requested in IMAGE_RATIO_TO_SIZE:
        size = IMAGE_RATIO_TO_SIZE[requested]
        width, height = (int(part) for part in size.split("x", 1))
        return size, requested, width, height
    if requested in SUPPORTED_IMAGE_SIZES:
        width, height = (int(part) for part in requested.split("x", 1))
        return requested, SUPPORTED_IMAGE_SIZES[requested], width, height
    return "1328x1328", "1:1", 1328, 1328


def _get_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def _build_image_prompt(prompt: str, *, size: str, ratio: str) -> str:
    return (
        "请调用图片生成能力直接生成图片，不要只输出文字描述。"
        "如果可以生成图片，请返回可访问的图片链接或包含图片链接的结果。\n"
        f"强制画布尺寸：{size} 像素。强制宽高比：{ratio}。"
        "必须严格按这个尺寸和比例生成，不要裁切成其它比例，不要改成默认尺寸。\n\n"
        f"用户需求：{prompt}"
    )


@router.post("/v1/images/generations")
@router.post("/images/generations")
async def create_image(request: Request):
    from backend.core.config import API_KEYS, settings

    client: QwenClient = request.app.state.qwen_client

    token = _get_token(request)
    if API_KEYS:
        if token != settings.ADMIN_KEY and token not in API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    prompt: str = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")

    n: int = min(max(int(body.get("n", 1)), 1), 4)
    model = _resolve_image_model(body.get("model"))
    size, ratio, width, height = _normalize_image_size(
        body.get("size") or body.get("ratio") or body.get("aspect_ratio")
    )

    image_options = {"size": size, "ratio": ratio, "width": width, "height": height}

    log.info(f"[T2I] model={model}, n={n}, size={size}, ratio={ratio}, prompt={prompt[:80]!r}")

    acc = None
    chat_id = None
    try:
        prompt_text = _build_image_prompt(prompt, size=size, ratio=ratio)
        event_payloads: list[str] = []
        async for item in client.chat_stream_events_with_retry(
            model,
            prompt_text,
            has_custom_tools=False,
            chat_type="image_gen",
            use_prewarmed=False,
            image_options=image_options,
        ):
            if item.get("type") == "meta":
                acc = item.get("acc")
                chat_id = item.get("chat_id")
                continue
            if item.get("type") != "event":
                continue
            event_payloads.append(json.dumps(item.get("event", {}), ensure_ascii=False))

        if acc is None or chat_id is None:
            raise HTTPException(status_code=500, detail="Image generation session was not created")

        chats = await client.list_chats(acc.token, limit=20)
        current_chat = next((c for c in chats if isinstance(c, dict) and c.get("id") == chat_id), None)
        answer_text = "\n".join(event_payloads)
        if current_chat:
            answer_text += "\n" + json.dumps(current_chat, ensure_ascii=False)

        upstream_failure = _extract_upstream_failure(answer_text)
        if upstream_failure:
            log.warning("[T2I] 上游返回失败 chat_id=%s error=%s", chat_id, upstream_failure)
            raise HTTPException(status_code=502, detail=upstream_failure)

        image_urls = _extract_image_urls(answer_text)
        log.info(
            "[T2I] 提取到 %s 张图片 URL: %s event_count=%s chat_found=%s chat_id=%s answer_tail=%r",
            len(image_urls),
            image_urls,
            len(event_payloads),
            bool(current_chat),
            chat_id,
            answer_text[-500:],
        )

        if not image_urls:
            raise HTTPException(status_code=500, detail=f"Image generation produced no image URL (chat_id={chat_id})")

        data = [{"url": url, "revised_prompt": prompt, "size": size, "ratio": ratio, "width": width, "height": height} for url in image_urls[:n]]
        return JSONResponse({"created": int(time.time()), "data": data})

    except HTTPException:
        raise
    except Exception as e:
        detail = str(e)
        log.error("[T2I] 生成失败: %s", detail)
        if "Qwen upstream error" in detail:
            raise HTTPException(status_code=502, detail=detail)
        raise HTTPException(status_code=500, detail=detail)
    finally:
        if acc is not None:
            client.account_pool.release(acc)
            if chat_id:
                await client.delete_chat_reliable(acc.token, chat_id, source="image_cleanup")
