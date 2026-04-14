from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
import json
import logging
import uuid

from backend.adapter.standard_request import StandardRequest
from backend.core.config import resolve_model, settings
from backend.core.request_logging import new_request_id, request_context, update_request_context
from backend.runtime import stream_presenter
from backend.runtime.execution import (
    build_tool_directive,
    cleanup_runtime_resources,
    collect_completion_run,
    evaluate_retry_directive,
)
from backend.services.auth_quota import resolve_auth_context
from backend.services.prompt_builder import CLAUDE_CODE_OPENAI_PROFILE, messages_to_prompt
from backend.services.qwen_client import QwenClient
from backend.toolcall.normalize import build_tool_name_registry

log = logging.getLogger("qwen2api.anthropic")
router = APIRouter()


class _AnthropicStreamState:
    def __init__(self, *, msg_id: str, model_name: str, prompt: str):
        self.msg_id = msg_id
        self.model_name = model_name
        self.prompt = prompt
        self.pending_chunks: list[str] = []
        self.answer_text_buffer: list[tuple[int, str]] = []
        self.block_index = 0
        self.current_block: dict[str, object] = {"type": None, "index": None, "tool_call_id": None}
        self.opened_tool_calls: set[str] = set()

    def ensure_message_start(self) -> None:
        if not self.pending_chunks:
            self.pending_chunks.append(_message_start_event(self.msg_id, self.model_name, self.prompt, ""))

    def close_current_block(self) -> None:
        index = self.current_block.get("index")
        if index is None:
            return
        self.pending_chunks.append(stream_presenter.anthropic_content_block_stop(index))
        self.current_block = {"type": None, "index": None, "tool_call_id": None}

    def open_textual_block(self, block_type: str) -> int:
        current_type = self.current_block.get("type")
        current_index = self.current_block.get("index")
        if current_type == block_type and isinstance(current_index, int):
            return current_index
        self.close_current_block()
        index = self.block_index
        self.block_index += 1
        if block_type == "thinking":
            content_block = {"type": "thinking", "thinking": ""}
        else:
            content_block = {"type": "text", "text": ""}
        self.pending_chunks.append(stream_presenter.anthropic_content_block_start(index, content_block))
        self.current_block = {"type": block_type, "index": index, "tool_call_id": None}
        return index

    def open_tool_block(self, tool_call_id: str, tool_name: str) -> int:
        current_index = self.current_block.get("index")
        if (
            self.current_block.get("type") == "tool_use"
            and self.current_block.get("tool_call_id") == tool_call_id
            and isinstance(current_index, int)
        ):
            return current_index
        self.close_current_block()
        index = self.block_index
        self.block_index += 1
        self.pending_chunks.append(
            f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': index, 'content_block': {'type': 'tool_use', 'id': tool_call_id, 'name': tool_name, 'input': {}}}, ensure_ascii=False)}\n\n"
        )
        self.current_block = {"type": "tool_use", "index": index, "tool_call_id": tool_call_id}
        self.opened_tool_calls.add(tool_call_id)
        return index

    def append_thinking_delta(self, text_chunk: str) -> None:
        index = self.open_textual_block("thinking")
        self.pending_chunks.append(
            stream_presenter.anthropic_content_block_delta(index, {"type": "thinking_delta", "thinking": text_chunk})
        )

    def buffer_answer_text(self, text_chunk: str) -> None:
        index = self.open_textual_block("text")
        self.answer_text_buffer.append((index, text_chunk))

    def append_tool_delta(self, *, tool_call_id: str, tool_name: str, partial_json: str) -> None:
        index = self.open_tool_block(tool_call_id, tool_name)
        if partial_json:
            self.pending_chunks.append(
                stream_presenter.anthropic_content_block_delta(index, {"type": "input_json_delta", "partial_json": partial_json})
            )

    def flush_answer_text(self) -> None:
        if not self.answer_text_buffer:
            return
        for index, text_chunk in self.answer_text_buffer:
            self.pending_chunks.append(
                stream_presenter.anthropic_content_block_delta(index, {"type": "text_delta", "text": text_chunk})
            )
        self.answer_text_buffer = []

    def clear_answer_text(self) -> None:
        self.answer_text_buffer = []


def _build_standard_request(req_data: dict) -> StandardRequest:
    model_name = req_data.get("model", "claude-3-5-sonnet")
    prompt_result = messages_to_prompt(req_data, client_profile=CLAUDE_CODE_OPENAI_PROFILE)
    prompt = prompt_result.prompt
    tools = prompt_result.tools
    tool_names = [tool_name for tool_name in (tool.get("name") for tool in tools) if isinstance(tool_name, str) and tool_name]
    return StandardRequest(
        prompt=prompt,
        response_model=model_name,
        resolved_model=resolve_model(model_name),
        surface="anthropic",
        client_profile=CLAUDE_CODE_OPENAI_PROFILE,
        stream=req_data.get("stream", False),
        tools=tools,
        tool_names=tool_names,
        tool_name_registry=build_tool_name_registry(tool_names),
        tool_enabled=prompt_result.tool_enabled,
    )


def _anthropic_usage(prompt: str, answer_text: str) -> dict[str, int]:
    return {"input_tokens": len(prompt), "output_tokens": len(answer_text)}


def _message_start_event(msg_id: str, model_name: str, prompt: str, answer_text: str) -> str:
    return stream_presenter.anthropic_message_start(msg_id, model_name, _anthropic_usage(prompt, answer_text))


async def _run_anthropic_attempt(
    *,
    client: QwenClient,
    standard_request: StandardRequest,
    current_prompt: str,
    history_messages: list[dict],
    stream_attempt: int,
    max_attempts: int,
):
    update_request_context(stream_attempt=stream_attempt + 1)
    execution = await collect_completion_run(client, standard_request, current_prompt)
    retry = evaluate_retry_directive(
        request=standard_request,
        current_prompt=current_prompt,
        history_messages=history_messages,
        attempt_index=stream_attempt,
        max_attempts=max_attempts,
        state=execution.state,
        allow_after_visible_output=False,
    )
    return execution, retry


def _visible_answer_text_length(*, directive, execution, stream_state: _AnthropicStreamState | None = None) -> int:
    if directive.stop_reason == "tool_use":
        return 0
    if stream_state is not None:
        return sum(len(text_chunk) for _, text_chunk in stream_state.answer_text_buffer)
    return len(execution.state.answer_text)


async def _add_used_tokens_for_prompt(*, users_db, token: str, prompt_text: str, answer_text_length: int) -> None:
    users = await users_db.get()
    for user in users:
        if user["id"] == token:
            user["used_tokens"] += answer_text_length + len(prompt_text)
            break
    await users_db.save(users)


@router.post("/messages")
@router.post("/v1/messages")
@router.post("/anthropic/v1/messages")
async def anthropic_messages(request: Request):
    app = request.app
    users_db = app.state.users_db
    client: QwenClient = app.state.qwen_client

    auth = await resolve_auth_context(request, users_db)
    token = auth.token

    try:
        req_data = await request.json()
    except Exception:
        raise HTTPException(400, {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}})

    standard_request = _build_standard_request(req_data)
    model_name = standard_request.response_model
    qwen_model = standard_request.resolved_model
    prompt = standard_request.prompt
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    history_messages = req_data.get("messages", [])

    with request_context(req_id=new_request_id(), surface="anthropic", requested_model=model_name, resolved_model=qwen_model):
        log.info(f"[ANT] model={qwen_model}, stream={standard_request.stream}, tool_enabled={standard_request.tool_enabled}, tools={[t.get('name') for t in standard_request.tools]}, prompt_len={len(prompt)}")

        if standard_request.stream:
            async def generate():
                current_prompt = prompt
                max_attempts = settings.MAX_RETRIES + (1 if standard_request.tools else 0)
                for stream_attempt in range(max_attempts):
                    stream_state = _AnthropicStreamState(msg_id=msg_id, model_name=model_name, prompt=current_prompt)
                    try:
                        update_request_context(stream_attempt=stream_attempt + 1)

                        async def on_delta(evt, text_chunk, _):
                            stream_state.ensure_message_start()
                            phase = evt.get("phase")

                            if text_chunk and phase in ("think", "thinking_summary"):
                                stream_state.append_thinking_delta(text_chunk)
                                return

                            if text_chunk and phase == "answer":
                                stream_state.buffer_answer_text(text_chunk)
                                return

                            if phase == "tool_call":
                                extra = evt.get("extra", {}) or {}
                                tool_call_id = extra.get("tool_call_id")
                                if tool_call_id is None:
                                    tool_call_id = f"tc_idx_{extra.get('index', 0)}"
                                tool_name = extra.get("tool_name")
                                if not tool_name:
                                    return
                                stream_state.append_tool_delta(
                                    tool_call_id=str(tool_call_id),
                                    tool_name=str(tool_name),
                                    partial_json=evt.get("content", ""),
                                )

                        execution = await collect_completion_run(
                            client,
                            standard_request,
                            current_prompt,
                            capture_events=False,
                            on_delta=on_delta,
                        )
                        retry = evaluate_retry_directive(
                            request=standard_request,
                            current_prompt=current_prompt,
                            history_messages=history_messages,
                            attempt_index=stream_attempt,
                            max_attempts=max_attempts,
                            state=execution.state,
                            allow_after_visible_output=False,
                        )
                        if retry.retry:
                            await cleanup_runtime_resources(client, execution.acc, execution.chat_id)
                            current_prompt = retry.next_prompt
                            continue

                        if not stream_state.pending_chunks:
                            stream_state.pending_chunks.append(_message_start_event(msg_id, model_name, current_prompt, execution.state.answer_text))

                        stream_state.close_current_block()

                        directive = build_tool_directive(standard_request, execution.state)
                        if directive.stop_reason == "tool_use":
                            stream_state.clear_answer_text()
                            stream_state.current_block = {"type": None, "index": None, "tool_call_id": None}
                        else:
                            stream_state.flush_answer_text()
                        expected_tool_ids = {
                            block.get("id")
                            for block in directive.tool_blocks
                            if block.get("type") == "tool_use" and block.get("id")
                        }
                        for block in directive.tool_blocks:
                            if block.get("type") != "tool_use":
                                continue
                            tool_id = block.get("id")
                            if tool_id in stream_state.opened_tool_calls:
                                continue
                            index = stream_state.open_tool_block(str(tool_id), str(block.get("name", "")))
                            stream_state.pending_chunks.append(
                                stream_presenter.anthropic_content_block_delta(index, {'type': 'input_json_delta', 'partial_json': json.dumps(block.get('input', {}), ensure_ascii=False)})
                            )
                            stream_state.close_current_block()

                        visible_answer_length = _visible_answer_text_length(
                            directive=directive,
                            execution=execution,
                            stream_state=stream_state,
                        )
                        stop_reason = "tool_use" if expected_tool_ids else "end_turn"
                        stream_state.pending_chunks.append(stream_presenter.anthropic_message_delta(stop_reason, visible_answer_length))
                        stream_state.pending_chunks.append(stream_presenter.anthropic_message_stop())

                        await _add_used_tokens_for_prompt(
                            users_db=users_db,
                            token=token,
                            prompt_text=current_prompt,
                            answer_text_length=len(execution.state.answer_text),
                        )
                        await cleanup_runtime_resources(client, execution.acc, execution.chat_id)

                        for chunk in stream_state.pending_chunks:
                            yield chunk
                        return
                    except HTTPException as he:
                        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': he.detail}}, ensure_ascii=False)}\n\n"
                        return
                    except Exception as e:
                        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}}, ensure_ascii=False)}\n\n"
                        return

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        current_prompt = prompt
        max_attempts = settings.MAX_RETRIES + (1 if standard_request.tools else 0)
        for stream_attempt in range(max_attempts):
            try:
                execution, retry = await _run_anthropic_attempt(
                    client=client,
                    standard_request=standard_request,
                    current_prompt=current_prompt,
                    history_messages=history_messages,
                    stream_attempt=stream_attempt,
                    max_attempts=max_attempts,
                )
                if retry.retry:
                    await cleanup_runtime_resources(client, execution.acc, execution.chat_id)
                    current_prompt = retry.next_prompt
                    continue

                directive = build_tool_directive(standard_request, execution.state)
                content_blocks: list[dict] = []
                if execution.state.reasoning_text:
                    content_blocks.append({"type": "thinking", "thinking": execution.state.reasoning_text})
                content_blocks.extend(directive.tool_blocks)

                await _add_used_tokens_for_prompt(
                    users_db=users_db,
                    token=token,
                    prompt_text=current_prompt,
                    answer_text_length=len(execution.state.answer_text),
                )
                await cleanup_runtime_resources(client, execution.acc, execution.chat_id)

                return JSONResponse(
                    {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model_name,
                        "content": content_blocks,
                        "stop_reason": directive.stop_reason,
                        "stop_sequence": None,
                        "usage": _anthropic_usage(current_prompt, execution.state.answer_text),
                    }
                )
            except Exception as e:
                if stream_attempt == max_attempts - 1:
                    raise HTTPException(status_code=500, detail=str(e))
