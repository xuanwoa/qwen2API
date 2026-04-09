from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import logging
import uuid
import time
import re
from backend.services.qwen_client import QwenClient
from backend.services.token_calc import calculate_usage
from backend.services.prompt_builder import messages_to_prompt
from backend.services.tool_parser import parse_tool_calls, inject_format_reminder
from backend.core.config import resolve_model, settings

log = logging.getLogger("qwen2api.chat")
router = APIRouter()

async def _stream_items_with_keepalive(client, model: str, prompt: str, has_custom_tools: bool, exclude_accounts=None):
    queue: asyncio.Queue = asyncio.Queue()

    async def _producer():
        try:
            async for item in client.chat_stream_events_with_retry(model, prompt, has_custom_tools=has_custom_tools, exclude_accounts=exclude_accounts):
                await queue.put(("item", item))
        except Exception as e:
            await queue.put(("error", e))
        finally:
            await queue.put(("done", None))

    producer_task = asyncio.create_task(_producer())
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=max(1, settings.STREAM_KEEPALIVE_INTERVAL))
            except asyncio.TimeoutError:
                yield {"type": "keepalive"}
                continue

            if kind == "item":
                yield payload
            elif kind == "error":
                raise payload
            elif kind == "done":
                break
    finally:
        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except asyncio.CancelledError:
                pass

def _extract_blocked_tool_names(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"Tool\s+([A-Za-z0-9_.:-]+)\s+does not exists?\.?", text)

def _has_recent_unchanged_read_result(messages) -> bool:
    checked = 0
    for msg in reversed(messages or []):
        checked += 1
        content = msg.get("content", "")
        texts = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    t = part.get("type")
                    if t == "text":
                        texts.append(part.get("text", ""))
                    elif t == "tool_result":
                        inner = part.get("content", "")
                        if isinstance(inner, str):
                            texts.append(inner)
                        elif isinstance(inner, list):
                            for p in inner:
                                if isinstance(p, dict) and p.get("type") == "text":
                                    texts.append(p.get("text", ""))
                elif isinstance(part, str):
                    texts.append(part)
        merged = "\n".join(t for t in texts if t)
        if "Unchanged since last read" in merged:
            return True
        if checked >= 10:
            break
    return False

@router.post("/completions")
@router.post("/chat/completions")
@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    app = request.app
    users_db = app.state.users_db
    client: QwenClient = app.state.qwen_client

    # 鉴权
    auth_header = request.headers.get("Authorization", "")
    token = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""

    if not token:
        token = request.headers.get("x-api-key", "").strip()
    if not token:
        token = request.query_params.get("key", "").strip() or request.query_params.get("api_key", "").strip()

    from backend.core.config import API_KEYS
    admin_k = settings.ADMIN_KEY

    if API_KEYS:
        if token != admin_k and token not in API_KEYS and not token:
            raise HTTPException(status_code=401, detail="Invalid API Key")

    # 获取下游用户并处理配额
    users = await users_db.get()
    user = next((u for u in users if u["id"] == token), None)
    if user and user.get("quota", 0) <= user.get("used_tokens", 0):
        raise HTTPException(status_code=402, detail="Quota Exceeded")
        
    try:
        req_data = await request.json()
    except Exception:
        raise HTTPException(400, {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}})
        
    model_name = req_data.get("model", "gpt-3.5-turbo")
    qwen_model = resolve_model(model_name)
    stream = req_data.get("stream", False)
    
    prompt, tools = messages_to_prompt(req_data)
    log.info(f"[OAI] model={qwen_model}, stream={stream}, tools={[t.get('name') for t in tools]}, prompt_len={len(prompt)}")
    history_messages = req_data.get("messages", [])

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if stream:
        async def generate():
            current_prompt = prompt  # local copy so we can modify for native-block retries
            excluded_accounts = set()
            for stream_attempt in range(settings.MAX_RETRIES):
              try:
                # We need to simulate `_stream_with_retry` behavior using `client.chat_stream_events_with_retry`
                events = []
                chat_id = None
                acc = None
                
                # Fetch all events (buffered)
                async for item in _stream_items_with_keepalive(client, qwen_model, current_prompt, has_custom_tools=bool(tools), exclude_accounts=excluded_accounts):
                    if item["type"] == "keepalive":
                        yield ": keepalive\n\n"
                        continue
                    if item["type"] == "meta":
                        chat_id = item["chat_id"]
                        acc = item["acc"]
                        yield ": upstream-connected\n\n"
                        continue
                    if item["type"] == "event":
                        events.append(item["event"])

                # Buffer all text first (Qwen fetches full SSE at once anyway)
                answer_text = ""
                reasoning_text = ""
                native_tc_chunks: dict = {}
                for evt in events:
                    if evt["type"] != "delta":
                        continue
                    phase = evt.get("phase", "")
                    content = evt.get("content", "")
                    if phase in ("think", "thinking_summary") and content:
                        reasoning_text += content
                    elif phase == "answer" and content:
                        answer_text += content
                    elif phase == "tool_call" and content:
                        tc_id = evt.get("extra", {}).get("tool_call_id", "tc_0")
                        if tc_id not in native_tc_chunks:
                            native_tc_chunks[tc_id] = {"name": "", "args": ""}
                        try:
                            chunk = json.loads(content)
                            if "name" in chunk:
                                native_tc_chunks[tc_id]["name"] = chunk["name"]
                            if "arguments" in chunk:
                                native_tc_chunks[tc_id]["args"] += chunk["arguments"]
                        except (json.JSONDecodeError, ValueError):
                            native_tc_chunks[tc_id]["args"] += content
                    if evt.get("status") == "finished" and phase == "answer":
                        break
                if native_tc_chunks and not answer_text:
                    log.info(f"[SSE-stream] 检测到 Qwen 原生 tool_call 事件: {list(native_tc_chunks.keys())}")
                    tc_parts = []
                    for tc_id, tc in native_tc_chunks.items():
                        name = tc["name"]
                        try:
                            inp = json.loads(tc["args"]) if tc["args"] else {}
                        except (json.JSONDecodeError, ValueError):
                            inp = {"raw": tc["args"]}
                        tc_parts.append(f'<tool_call>{{"name": {json.dumps(name)}, "input": {json.dumps(inp, ensure_ascii=False)}}}</tool_call>')
                    answer_text = "\n".join(tc_parts)
                elif answer_text:
                    log.debug(f"[SSE-stream] 收到 answer 文本({len(answer_text)}字): {answer_text[:120]!r}")

                # Detect Qwen native tool call interception before yielding
                blocked_names = _extract_blocked_tool_names(answer_text.strip())
                if blocked_names and tools and stream_attempt < settings.MAX_RETRIES - 1:
                    blocked_name = blocked_names[0]
                    if acc:
                        client.account_pool.release(acc)
                        if chat_id:
                            import asyncio
                            asyncio.create_task(client.delete_chat(acc.token, chat_id))
                    log.warning(f"[NativeBlock-Stream] Qwen拦截原生工具调用 '{blocked_name}'，注入格式纠正后重试 (attempt {stream_attempt+1}/{settings.MAX_RETRIES})")
                    if acc: excluded_accounts.add(acc.email)
                    current_prompt = inject_format_reminder(current_prompt, blocked_name)
                    await asyncio.sleep(0.15)
                    continue  # retry the stream call

                # Detect tool calls BEFORE yielding any content
                tool_blocks, stop = parse_tool_calls(answer_text, tools)
                has_tool_call = stop == "tool_use"
                if has_tool_call:
                    first_tool = next((b for b in tool_blocks if b.get("type") == "tool_use"), None)
                    if (first_tool and first_tool.get("name") == "Read"
                            and _has_recent_unchanged_read_result(history_messages)
                            and stream_attempt < settings.MAX_RETRIES - 1):
                        if acc:
                            client.account_pool.release(acc)
                            if chat_id:
                                import asyncio
                                asyncio.create_task(client.delete_chat(acc.token, chat_id))
                        current_prompt = current_prompt.rstrip()
                        force_text = (
                            "[MANDATORY NEXT STEP]: You just received 'Unchanged since last read'. "
                            "Do NOT call Read again on the same target. "
                            "Choose another tool now."
                        )
                        if current_prompt.endswith("Assistant:"):
                            current_prompt = current_prompt[:-len("Assistant:")] + force_text + "\nAssistant:"
                        else:
                            current_prompt += "\n\n" + force_text + "\nAssistant:"
                        log.warning(f"[ToolLoop-OAI] 检测到 Unchanged since last read，立即阻止重复 Read (attempt {stream_attempt+1}/{settings.MAX_RETRIES})")
                        await asyncio.sleep(0.15)
                        continue

                mk = lambda delta, finish=None: json.dumps({
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
                }, ensure_ascii=False)

                # Role chunk
                yield f"data: {mk({'role': 'assistant'})}\n\n"

                if has_tool_call:
                    # Emit tool_calls chunks (OpenAI streaming format)
                    tc_list = [b for b in tool_blocks if b["type"] == "tool_use"]
                    for idx, tc in enumerate(tc_list):
                        # Function name chunk
                        yield f"data: {mk({'tool_calls': [{'index': idx, 'id': tc['id'], 'type': 'function', 'function': {'name': tc['name'], 'arguments': ''}}]})}\n\n"
                        # Arguments chunk
                        yield f"data: {mk({'tool_calls': [{'index': idx, 'function': {'arguments': json.dumps(tc.get('input', {}), ensure_ascii=False)}}]})}\n\n"
                    yield f"data: {mk({}, 'tool_calls')}\n\n"
                else:
                    # Thinking chunks
                    if reasoning_text:
                        yield f"data: {mk({'reasoning_content': reasoning_text})}\n\n"
                    # Content chunks
                    if answer_text:
                        yield f"data: {mk({'content': answer_text})}\n\n"
                    yield f"data: {mk({}, 'stop')}\n\n"

                yield "data: [DONE]\n\n"
                
                users = await users_db.get()
                for u in users:
                    if u["id"] == token:
                        u["used_tokens"] += len(answer_text) + len(prompt)
                        break
                await users_db.save(users)
                
                if acc:
                    client.account_pool.release(acc)
                    if chat_id:
                        import asyncio
                        asyncio.create_task(client.delete_chat(acc.token, chat_id))
                return  # success — exit the retry loop
              except HTTPException as he:
                yield f"data: {json.dumps({'error': he.detail})}\n\n"
                return
              except Exception as e:
                if acc and acc.inflight > 0:
                    client.account_pool.release(acc)
                    if chat_id:
                        import asyncio
                        asyncio.create_task(client.delete_chat(acc.token, chat_id))
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return

        return StreamingResponse(generate(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    else:
        current_prompt = prompt
        excluded_accounts = set()
        for stream_attempt in range(settings.MAX_RETRIES):
            try:
                events = []
                chat_id = None
                acc = None
                
                async for item in client.chat_stream_events_with_retry(qwen_model, current_prompt, has_custom_tools=bool(tools), exclude_accounts=excluded_accounts):
                    if item["type"] == "meta":
                        chat_id = item["chat_id"]
                        acc = item["acc"]
                        continue
                    if item["type"] == "event":
                        events.append(item["event"])

                answer_text = ""
                reasoning_text = ""
                native_tc_chunks: dict = {}
                for evt in events:
                    if evt["type"] != "delta":
                        continue
                    phase = evt.get("phase", "")
                    content = evt.get("content", "")
                    if phase in ("think", "thinking_summary") and content:
                        reasoning_text += content
                    elif phase == "answer" and content:
                        answer_text += content
                    elif phase == "tool_call" and content:
                        tc_id = evt.get("extra", {}).get("tool_call_id", "tc_0")
                        if tc_id not in native_tc_chunks:
                            native_tc_chunks[tc_id] = {"name": "", "args": ""}
                        try:
                            chunk = json.loads(content)
                            if "name" in chunk:
                                native_tc_chunks[tc_id]["name"] = chunk["name"]
                            if "arguments" in chunk:
                                native_tc_chunks[tc_id]["args"] += chunk["arguments"]
                        except (json.JSONDecodeError, ValueError):
                            native_tc_chunks[tc_id]["args"] += content
                    if evt.get("status") == "finished" and phase == "answer":
                        break
                        
                if native_tc_chunks and not answer_text:
                    tc_parts = []
                    for tc_id, tc in native_tc_chunks.items():
                        name = tc["name"]
                        try:
                            inp = json.loads(tc["args"]) if tc["args"] else {}
                        except (json.JSONDecodeError, ValueError):
                            inp = {"raw": tc["args"]}
                        tc_parts.append(f'<tool_call>{{"name": {json.dumps(name)}, "input": {json.dumps(inp, ensure_ascii=False)}}}</tool_call>')
                    answer_text = "\n".join(tc_parts)

                blocked_names = _extract_blocked_tool_names(answer_text.strip())
                if blocked_names and tools and stream_attempt < settings.MAX_RETRIES - 1:
                    blocked_name = blocked_names[0]
                    if acc:
                        client.account_pool.release(acc)
                        if chat_id:
                            import asyncio
                            asyncio.create_task(client.delete_chat(acc.token, chat_id))
                    current_prompt = inject_format_reminder(current_prompt, blocked_name)
                    await asyncio.sleep(0.15)
                    continue

                tool_blocks, stop = parse_tool_calls(answer_text, tools)
                has_tool_call = stop == "tool_use"
                if has_tool_call:
                    first_tool = next((b for b in tool_blocks if b.get("type") == "tool_use"), None)
                    if (first_tool and first_tool.get("name") == "Read"
                            and _has_recent_unchanged_read_result(history_messages)
                            and stream_attempt < settings.MAX_RETRIES - 1):
                        if acc:
                            client.account_pool.release(acc)
                            if chat_id:
                                import asyncio
                                asyncio.create_task(client.delete_chat(acc.token, chat_id))
                        current_prompt = current_prompt.rstrip()
                        force_text = (
                            "[MANDATORY NEXT STEP]: You just received 'Unchanged since last read'. "
                            "Do NOT call Read again on the same target. "
                            "Choose another tool now."
                        )
                        if current_prompt.endswith("Assistant:"):
                            current_prompt = current_prompt[:-len("Assistant:")] + force_text + "\nAssistant:"
                        else:
                            current_prompt += "\n\n" + force_text + "\nAssistant:"
                        log.warning(f"[ToolLoop-OAI] 检测到 Unchanged since last read，立即阻止重复 Read (attempt {stream_attempt+1}/{settings.MAX_RETRIES})")
                        await asyncio.sleep(0.15)
                        continue

                if has_tool_call:
                    tc_list = [b for b in tool_blocks if b["type"] == "tool_use"]
                    oai_tool_calls = [{
                        "id": tc["id"], "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("input", {}), ensure_ascii=False)
                        }
                    } for tc in tc_list]
                    msg = {"role": "assistant", "content": None, "tool_calls": oai_tool_calls}
                    finish_reason = "tool_calls"
                else:
                    msg = {"role": "assistant", "content": answer_text}
                    if reasoning_text:
                        msg["reasoning_content"] = reasoning_text
                    finish_reason = "stop"

                users = await users_db.get()
                for u in users:
                    if u["id"] == token:
                        u["used_tokens"] += len(answer_text) + len(prompt)
                        break
                await users_db.save(users)

                if acc:
                    client.account_pool.release(acc)
                    if chat_id:
                        import asyncio
                        asyncio.create_task(client.delete_chat(acc.token, chat_id))

                from fastapi.responses import JSONResponse
                return JSONResponse({
                    "id": completion_id, "object": "chat.completion", "created": created, "model": model_name,
                    "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(answer_text),
                              "total_tokens": len(prompt) + len(answer_text)}
                })
            except Exception as e:
                if acc and acc.inflight > 0:
                    client.account_pool.release(acc)
                    if chat_id:
                        import asyncio
                        asyncio.create_task(client.delete_chat(acc.token, chat_id))
                if stream_attempt == settings.MAX_RETRIES - 1:
                    raise HTTPException(status_code=500, detail=str(e))
                await asyncio.sleep(1)
