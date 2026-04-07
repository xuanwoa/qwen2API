from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import json
import logging
import asyncio
import uuid
from backend.services.qwen_client import QwenClient
from backend.services.token_calc import calculate_usage
from backend.services.prompt_builder import build_prompt_with_tools
from backend.core.config import resolve_model

log = logging.getLogger("qwen2api.anthropic")
router = APIRouter()

@router.post("/messages")
@router.post("/v1/messages")
@router.post("/anthropic/v1/messages")
async def anthropic_messages(request: Request):
    """
    Claude API 协议转换层 -> 转入 OpenAI/Qwen 统一处理内核
    """
    app = request.app
    users_db = app.state.users_db
    client: QwenClient = app.state.qwen_client

    # 鉴权 (完全复原单文件逻辑)
    token = request.headers.get("x-api-key", "").strip()

    # Anthropic 请求可能没有传 x-api-key 而是使用 Bearer Token
    if not token:
        bearer = request.headers.get("Authorization", "")
        if bearer.startswith("Bearer "):
            token = bearer[7:].strip()

    # 有些工具可能会传在 querystring 中
    if not token:
        token = request.query_params.get("key", "").strip() or request.query_params.get("api_key", "").strip()

    from backend.core.config import API_KEYS, settings
    admin_k = settings.ADMIN_KEY

    if API_KEYS:
        if token != admin_k and token not in API_KEYS and not token:
            raise HTTPException(status_code=401, detail="Invalid API Key")

    # 获取下游用户处理配额
    users = await users_db.get()
    user = next((u for u in users if u["id"] == token), None)
    if user and user.get("quota", 0) <= user.get("used_tokens", 0):
        raise HTTPException(status_code=402, detail="Quota Exceeded")
        
    body = await request.json()
    model = resolve_model(body.get("model", "claude-3-5-sonnet"))
    messages = body.get("messages", [])
    tools = body.get("tools", [])
    
    # 构造兼容 OpenAI 的消息格式给 Prompt builder
    system_text = body.get("system", "")
    oai_msgs = []
    if system_text:
        oai_msgs.append({"role": "system", "content": system_text})
    for m in messages:
        oai_msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})

    content = build_prompt_with_tools(oai_msgs, tools)
            
    log.info(f"[Anthropic] model={model}, stream=True, tools={[t.get('name') for t in tools]}, prompt_len={len(content)}")

    # try:
    #     events, chat_id, acc = await client.chat_stream_events_with_retry(model, content)
    # except Exception as e:
    #     log.error(f"Anthropic proxy failed: {e}")
    #     raise HTTPException(status_code=500, detail=str(e))
        
    async def generate():
        current_prompt = content

        for stream_attempt in range(5): # MAX_RETRIES
            try:
                chat_id = None
                acc = None
                
                msg_id = f"msg_{uuid.uuid4().hex[:12]}"
                input_usage = len(current_prompt) # simple char len approx or precise
                
                yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'usage': {'input_tokens': input_usage, 'output_tokens': 0}}})}\n\n"
                
                answer_chunks = []
                native_tc_chunks = {}
                block_idx = 0
                in_thinking_block = False
                
                async for item in client.chat_stream_events_with_retry(model, current_prompt):
                    if item["type"] == "meta":
                        chat_id = item["chat_id"]
                        acc = item["acc"]
                        continue
                    
                    if item["type"] == "event":
                        evt = item["event"]
                        if evt.get("type") != "delta":
                            continue
                        
                        phase = evt.get("phase", "")
                        cont = evt.get("content", "")
                        
                        if phase in ("think", "thinking_summary") and cont:
                            if not in_thinking_block:
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_idx, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                                in_thinking_block = True
                            
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_idx, 'delta': {'type': 'thinking_delta', 'thinking': cont}})}\n\n"
                            
                        elif phase == "answer" and cont:
                            if in_thinking_block:
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"
                                in_thinking_block = False
                                block_idx += 1
                                
                            answer_chunks.append(cont)
                            
                        elif phase == "tool_call" and cont:
                            if in_thinking_block:
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"
                                in_thinking_block = False
                                block_idx += 1
                                
                            tc_id = evt.get("extra", {}).get("tool_call_id", "tc_0")
                            if tc_id not in native_tc_chunks:
                                native_tc_chunks[tc_id] = {"name": "", "args": ""}
                            try:
                                chunk_json = json.loads(cont)
                                if "name" in chunk_json:
                                    native_tc_chunks[tc_id]["name"] = chunk_json["name"]
                                if "arguments" in chunk_json:
                                    native_tc_chunks[tc_id]["args"] += chunk_json["arguments"]
                            except (json.JSONDecodeError, ValueError):
                                native_tc_chunks[tc_id]["args"] += cont

                if in_thinking_block:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"
                    in_thinking_block = False
                    block_idx += 1

                answer_text = "".join(answer_chunks)

                if native_tc_chunks:
                    log.info(f"[Native-TC] 收到原生工具调用事件: {list(native_tc_chunks.keys())}")
                    tc_parts = []
                    for tc_id, tc in native_tc_chunks.items():
                        name = tc["name"]
                        try:
                            inp = json.loads(tc["args"]) if tc["args"] else {}
                        except (json.JSONDecodeError, ValueError):
                            inp = {"raw": tc["args"]}
                        tc_parts.append(f'✿ACTION✿\n{{"action": {json.dumps(name)}, "args": {json.dumps(inp, ensure_ascii=False)}}}\n✿END_ACTION✿')
                    
                    if not answer_text:
                        answer_text = "\n\n".join(tc_parts)
                    else:
                        answer_text += "\n\n" + "\n\n".join(tc_parts)

                # Parse tools
                from backend.services.tool_parser import parse_tool_calls
                if tools:
                    blocks, stop_reason = parse_tool_calls(answer_text, tools)
                else:
                    blocks = [{"type": "text", "text": answer_text}]
                    stop_reason = "end_turn"

                # Text + tool_use blocks
                for blk in blocks:
                    if blk["type"] == "text" and blk.get("text"):
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_idx, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_idx, 'delta': {'type': 'text_delta', 'text': blk['text']}})}\n\n"
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"
                        block_idx += 1
                    elif blk["type"] == "tool_use":
                        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': block_idx, 'content_block': {'type': 'tool_use', 'id': blk['id'], 'name': blk['name'], 'input': {}}})}\n\n"
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': block_idx, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(blk.get('input', {}), ensure_ascii=False)}})}\n\n"
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"
                        block_idx += 1

                yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason}, 'usage': {'output_tokens': len(answer_text)}})}\n\n"
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

                users = await users_db.get()
                for u in users:
                    if u["id"] == token:
                        u["used_tokens"] += len(answer_text) + input_usage
                        break
                await users_db.save(users)

                if acc:
                    client.account_pool.release(acc)
                    if chat_id:
                        import asyncio
                        asyncio.create_task(client.delete_chat(acc.token, chat_id))
                return

            except Exception as e:
                log.error(f"Anthropic stream error: {e}")
                return

    return StreamingResponse(generate(), media_type="text/event-stream")
