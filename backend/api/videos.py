"""
视频生成接口 — 兼容 OpenAI 风格的 /v1/videos/generations。
"""
import asyncio
import json
import logging
import re
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.api.images import _extract_upstream_failure, _normalize_image_size
from backend.services.qwen_client import QwenClient

log = logging.getLogger("qwen2api.videos")
router = APIRouter()

DEFAULT_VIDEO_MODEL = "qwen3.6-plus"

VIDEO_MODEL_MAP = {
    "qwen-video": "qwen3.6-plus",
    "qwen-video-plus": "qwen3.6-plus",
    "qwen-video-turbo": "qwen3.6-plus",
    "qwen3.6-plus-video": "qwen3.6-plus",
}

VIDEO_URL_KEYS = {
    "url",
    "video",
    "src",
    "videoUrl",
    "video_url",
    "videoURL",
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

TASK_ID_KEYS = {
    "task_id",
    "taskId",
    "wanx_task_id",
    "wanxTaskId",
}

VIDEO_RUNNING_STATUSES = {"running", "pending", "queued", "processing", "created"}
VIDEO_SUCCESS_STATUSES = {"success", "succeeded", "finished", "completed"}


def _looks_like_video_url(value: str) -> bool:
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        return False
    lowered = value.lower()
    if re.search(r"\.(?:mp4|webm|mov|m3u8)(?:[?#][^\s\"'<>]*)?$", lowered):
        return True
    video_hosts = ("cdn.qwenlm.ai", "wanx.alicdn.com", "alicdn.com")
    return any(host in lowered for host in video_hosts) and any(marker in lowered for marker in ("video", "mp4", "t2v"))


def _collect_video_urls_from_obj(value: Any, urls: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and (key in VIDEO_URL_KEYS or _looks_like_video_url(item)):
                if _looks_like_video_url(item):
                    urls.append(item)
            else:
                _collect_video_urls_from_obj(item, urls)
        return
    if isinstance(value, list):
        for item in value:
            _collect_video_urls_from_obj(item, urls)


def _collect_task_ids_from_obj(value: Any, task_ids: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in TASK_ID_KEYS and isinstance(item, str) and item:
                task_ids.append(item)
                continue
            _collect_task_ids_from_obj(item, task_ids)
        return
    if isinstance(value, list):
        for item in value:
            _collect_task_ids_from_obj(item, task_ids)


def _extract_video_urls(text: str) -> list[str]:
    urls: list[str] = []

    for u in re.findall(r'!\[.*?\]\((https?://[^\s\)]+)\)', text):
        if _looks_like_video_url(u):
            urls.append(u.rstrip(").,;"))

    for u in re.findall(r'"(?:url|video|src|videoUrl|video_url)"\s*:\s*"(https?://[^"]+)"', text):
        if _looks_like_video_url(u):
            urls.append(u)

    video_pattern = r'https?://[^\s"<>]+\.(?:mp4|webm|mov|m3u8)(?:[^\s"<>]*)'
    for u in re.findall(video_pattern, text, re.IGNORECASE):
        urls.append(u.rstrip(".,;)\"'>"))

    for match in re.finditer(r"[\{\[]", text or ""):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[match.start():])
        except Exception:
            continue
        _collect_video_urls_from_obj(obj, urls)

    seen: set[str] = set()
    result: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _extract_task_ids(text: str) -> list[str]:
    task_ids: list[str] = []
    for match in re.finditer(r"[\{\[]", text or ""):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[match.start():])
        except Exception:
            continue
        _collect_task_ids_from_obj(obj, task_ids)

    seen: set[str] = set()
    result: list[str] = []
    for task_id in task_ids:
        if task_id not in seen:
            seen.add(task_id)
            result.append(task_id)
    return result


def _resolve_video_model(requested: str | None) -> str:
    from backend.core.config import resolve_model
    from backend.services.model_modes import parse_model_mode

    if not requested:
        return DEFAULT_VIDEO_MODEL
    aliased = VIDEO_MODEL_MAP.get(str(requested).strip(), str(requested).strip())
    mode = parse_model_mode(aliased, default_model=DEFAULT_VIDEO_MODEL)
    return resolve_model(mode.base_model or DEFAULT_VIDEO_MODEL)


def _get_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key", "").strip()


def _coerce_duration(value: Any) -> int:
    try:
        duration = int(value)
    except (TypeError, ValueError):
        duration = 5
    return min(max(duration, 1), 10)


def _build_video_prompt(prompt: str, *, size: str, ratio: str, duration: int) -> str:
    return (
        f"{prompt}\n\n"
        f"视频要求：生成 {duration} 秒视频，宽高比 {ratio}，参考画面尺寸 {size}。"
    )


async def _poll_video_task(client: QwenClient, token: str, task_id: str, *, timeout_seconds: int = 420) -> str:
    started = time.monotonic()
    interval = 10.0
    snapshots: list[str] = []
    last_status = ""

    while time.monotonic() - started < timeout_seconds:
        res = await client.get_vision_task_status(token, task_id, timeout=30.0)
        body_text = str(res.get("body") or "")
        snapshots.append(body_text)

        if int(res.get("status") or 0) != 200:
            log.warning("[T2V] 任务状态查询 HTTP %s task_id=%s body=%r", res.get("status"), task_id, body_text[:300])
            await asyncio.sleep(interval)
            continue

        try:
            obj = json.loads(body_text)
        except Exception:
            obj = {}
        data = obj.get("data") if isinstance(obj, dict) and isinstance(obj.get("data"), dict) else {}
        status = str(
            (obj.get("task_status") if isinstance(obj, dict) else None)
            or (obj.get("status") if isinstance(obj, dict) else None)
            or data.get("task_status")
            or data.get("status")
            or ""
        ).lower()
        last_status = status or last_status

        if status in VIDEO_SUCCESS_STATUSES:
            log.info("[T2V] 视频任务完成 task_id=%s elapsed=%.1fs", task_id, time.monotonic() - started)
            return "\n".join(snapshots)
        if status and status not in VIDEO_RUNNING_STATUSES:
            raise RuntimeError(f"Video task failed status={status} body={body_text[:500]}")
        if not status:
            log.info("[T2V] 任务状态未识别 task_id=%s body=%r", task_id, body_text[:300])

        await asyncio.sleep(interval)

    raise RuntimeError(f"Video task timed out task_id={task_id} last_status={last_status or '-'}")


async def _collect_chat_detail_text(client: QwenClient, token: str, chat_id: str) -> str:
    res = await client.get_chat_detail(token, chat_id, timeout=30.0)
    if int(res.get("status") or 0) != 200:
        return ""
    return str(res.get("body") or "")


async def _create_video_with_account(
    client: QwenClient,
    token: str,
    *,
    model: str,
    prompt_text: str,
    video_options: dict,
) -> tuple[str, list[str], str]:
    chat_id = await client.create_chat(token, model, chat_type="t2v", use_prewarmed=False)
    payload = client._build_payload(
        chat_id,
        model,
        prompt_text,
        has_custom_tools=False,
        chat_type="t2v",
        image_options=video_options,
    )
    payload["stream"] = False

    res = await client.post_chat_completion_once(token, chat_id, payload, timeout=90.0)
    body_text = str(res.get("body") or "")
    if int(res.get("status") or 0) != 200:
        raise RuntimeError(f"video completion HTTP {res.get('status')}: {body_text[:500]}")

    answer_text = body_text
    upstream_failure = _extract_upstream_failure(answer_text)
    if upstream_failure:
        raise RuntimeError(upstream_failure)

    video_urls = _extract_video_urls(answer_text)
    task_ids = _extract_task_ids(answer_text)
    log.info("[T2V] 非流式响应 chat_id=%s task_ids=%s video_urls=%s body_tail=%r", chat_id, task_ids, len(video_urls), body_text[-500:])

    if not video_urls and task_ids:
        answer_text += "\n" + await _poll_video_task(client, token, task_ids[0])
        video_urls = _extract_video_urls(answer_text)

    if not video_urls:
        detail_text = await _collect_chat_detail_text(client, token, chat_id)
        if detail_text:
            answer_text += "\n" + detail_text
            video_urls = _extract_video_urls(answer_text)

    return chat_id, video_urls, answer_text


@router.post("/v1/videos/generations")
@router.post("/videos/generations")
async def create_video(request: Request):
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

    prompt: str = str(body.get("prompt", "")).strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")

    n = min(max(int(body.get("n", 1)), 1), 2)
    model = _resolve_video_model(body.get("model"))
    duration = _coerce_duration(body.get("duration"))
    size, ratio, width, height = _normalize_image_size(
        body.get("size") or body.get("ratio") or body.get("aspect_ratio")
    )
    video_options = {"size": size, "ratio": ratio, "width": width, "height": height, "duration": duration}

    log.info("[T2V] model=%s n=%s size=%s ratio=%s duration=%ss prompt=%r", model, n, size, ratio, duration, prompt[:80])

    prompt_text = _build_video_prompt(prompt, size=size, ratio=ratio, duration=duration)
    exclude: set[str] = set()
    last_error: str | None = None

    for attempt in range(max(1, int(settings.MAX_RETRIES))):
        acc = None
        chat_id = None
        try:
            acc = await client.account_pool.acquire_wait(timeout=60, exclude=exclude)
            if acc is None:
                raise RuntimeError("No available accounts in pool (all busy or rate limited)")

            log.info("[T2V] 使用账号=%s 第%s/%s次", acc.email, attempt + 1, settings.MAX_RETRIES)
            chat_id, video_urls, answer_text = await _create_video_with_account(
                client,
                acc.token,
                model=model,
                prompt_text=prompt_text,
                video_options=video_options,
            )

            log.info("[T2V] 提取到 %s 个视频 URL chat_id=%s answer_tail=%r", len(video_urls), chat_id, answer_text[-500:])
            if not video_urls:
                raise RuntimeError(f"Video generation produced no video URL (chat_id={chat_id})")

            data = [
                {
                    "url": url,
                    "revised_prompt": prompt,
                    "size": size,
                    "ratio": ratio,
                    "width": width,
                    "height": height,
                    "duration": duration,
                }
                for url in video_urls[:n]
            ]
            return JSONResponse({"created": int(time.time()), "data": data})

        except Exception as e:
            last_error = str(e)
            if acc is not None:
                exclude.add(acc.email)
            log.warning("[T2V] 尝试失败 第%s/%s次 账号=%s 错误=%s", attempt + 1, settings.MAX_RETRIES, getattr(acc, "email", "-"), last_error)
        finally:
            if acc is not None:
                client.account_pool.release(acc)
                if chat_id:
                    client.delete_chat_background(acc.token, chat_id, source="video_cleanup")

    detail = f"All {settings.MAX_RETRIES} attempts failed. Last error: {last_error or 'unknown'}"
    log.error("[T2V] 生成失败: %s", detail)
    if "Qwen upstream error" in detail:
        raise HTTPException(status_code=502, detail=detail)
    raise HTTPException(status_code=500, detail=detail)
