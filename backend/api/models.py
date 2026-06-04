from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.core.config import MODEL_MAP, resolve_model
from backend.services.auth_quota import resolve_auth_context
from backend.services.model_catalog import build_fallback_model_list, build_model_entry, build_openai_model_list
from backend.services.model_modes import parse_model_mode
from backend.services.qwen_client import QwenClient

router = APIRouter()


def _build_model_list_payload() -> dict:
    return build_fallback_model_list(MODEL_MAP)


@router.get("/v1/models")
async def list_models(request: Request):
    app = request.app
    users_db = app.state.users_db
    client: QwenClient = app.state.qwen_client

    # 鉴权（只校验客户端 API KEY，不把它当 Qwen token 用）
    await resolve_auth_context(request, users_db)

    # 从账号池拿合法 Qwen token 调上游 /api/models，带 5min 缓存
    upstream_models = await client.list_models_from_pool()

    if upstream_models:
        return JSONResponse(build_openai_model_list(upstream_models))

    # 上游不可用时才回退到静态 MODEL_MAP（包含 gpt-4o/claude 等别名）
    return JSONResponse(_build_model_list_payload())


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    mode = parse_model_mode(model_id)
    if not mode.base_model:
        raise HTTPException(status_code=404, detail={"error": {"message": f"Model '{model_id}' not found", "type": "invalid_request_error"}})
    resolved = resolve_model(mode.base_model)
    capabilities = {}
    if mode.force_thinking:
        capabilities["thinking"] = True
    if mode.mode == "deep_research":
        capabilities["deep_research"] = True
        capabilities["search"] = True
    if mode.mode == "search":
        capabilities["search"] = True
    if mode.mode == "image":
        capabilities["image_gen"] = True
    if mode.mode == "video":
        capabilities["video_gen"] = True
    if mode.mode == "webdev":
        capabilities["web_dev"] = True
    if mode.mode == "slides":
        capabilities["slides"] = True
    payload = build_model_entry(
        model_id=model_id,
        base_model=mode.base_model,
        capabilities=capabilities,
        mode=mode.mode,
        display_name=model_id,
        family=resolved.split("-", 1)[0],
        owned_by="qwen2api",
    )
    payload["resolved_model"] = resolved
    return JSONResponse(payload)
