from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend.adapter.standard_request import StandardRequest
from backend.core.request_logging import new_request_id, request_context, update_request_context
from backend.core.request_trace import log_test_prompt, prompt_tail
from backend.runtime.execution import build_tool_directive, build_usage_delta_factory, request_max_attempts, tool_directive_visible_text
from backend.runtime.visible_text import VisibleTextSanitizer, sanitize_visible_text
from backend.services.attachment_preprocessor import preprocess_attachments
from backend.services.auth_quota import resolve_auth_context
from backend.services.client_profiles import detect_openai_client_profile
from backend.services.completion_bridge import run_retryable_completion_bridge
from backend.services.context_attachment_manager import derive_session_key, prepare_context_attachments
from backend.services.qwen_client import QwenClient
from backend.services.standard_request_builder import build_chat_standard_request
from backend.services.task_session import (
    build_openai_assistant_history_message,
    clear_invalidated_session_chat,
    log_session_plan_reuse_cancelled,
    persist_session_turn,
    plan_persistent_session_turn,
)


log = logging.getLogger("qwen2api.responses")
router = APIRouter()


def _stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_stringify_content(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("output"), str):
            return value["output"]
        if isinstance(value.get("content"), (str, list, dict)):
            return _stringify_content(value.get("content"))
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _convert_content_part(part: Any, *, role: str | None = None) -> Any:
    if isinstance(part, str):
        return {"type": "text", "text": part}
    if not isinstance(part, dict):
        return {"type": "text", "text": str(part)}

    part_type = part.get("type")
    if part_type in ("input_text", "output_text", "text"):
        return {"type": "text", "text": str(part.get("text") or "")}
    if part_type in ("input_image", "image_url"):
        if isinstance(part.get("image_url"), dict):
            return {"type": "image_url", "image_url": part["image_url"]}
        if isinstance(part.get("image_url"), str):
            return {"type": "image_url", "image_url": {"url": part["image_url"]}}
        if isinstance(part.get("file_id"), str):
            return {
                "type": "input_image",
                "file_id": part["file_id"],
                "mime_type": part.get("mime_type") or part.get("media_type") or "image/*",
            }
        if isinstance(part.get("data"), str):
            return {
                "type": "input_image",
                "data": part["data"],
                "mime_type": part.get("mime_type") or part.get("media_type") or "image/*",
            }
        return dict(part)
    if part_type in ("input_file", "file"):
        converted = dict(part)
        converted["type"] = "input_file"
        if "filename" not in converted and isinstance(converted.get("name"), str):
            converted["filename"] = converted["name"]
        if "data_base64" not in converted and isinstance(converted.get("file_data"), str):
            converted["data_base64"] = converted["file_data"]
        return converted
    if part_type in ("tool_result", "function_call_output"):
        return {
            "type": "tool_result",
            "tool_use_id": part.get("tool_use_id") or part.get("call_id") or part.get("id") or "",
            "content": _stringify_content(part.get("content") if "content" in part else part.get("output")),
        }
    if part_type == "refusal":
        return {"type": "text", "text": str(part.get("refusal") or part.get("text") or "")}
    if role == "assistant" and isinstance(part.get("text"), str):
        return {"type": "text", "text": part["text"]}
    if isinstance(part.get("content"), str):
        return {"type": "text", "text": part["content"]}
    return {"type": "text", "text": json.dumps(part, ensure_ascii=False)}


def _convert_message_content(content: Any, *, role: str | None = None) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        converted = [_convert_content_part(part, role=role) for part in content]
        text_parts = [part["text"] for part in converted if isinstance(part, dict) and part.get("type") == "text" and part.get("text")]
        non_text_parts = [part for part in converted if not (isinstance(part, dict) and part.get("type") == "text")]
        return converted if non_text_parts else "\n".join(text_parts)
    if isinstance(content, dict):
        converted = _convert_content_part(content, role=role)
        if isinstance(converted, dict) and converted.get("type") == "text":
            return converted.get("text", "")
        return [converted]
    return str(content) if content is not None else ""


def _convert_response_message(item: dict[str, Any]) -> dict[str, Any] | None:
    role = item.get("role") or "user"
    if role not in {"system", "user", "assistant", "tool", "developer"}:
        role = "user"
    if role == "developer":
        role = "system"
    content = item.get("content") if "content" in item else item.get("input")
    message: dict[str, Any] = {"role": role, "content": _convert_message_content(content, role=role)}
    if isinstance(item.get("tool_calls"), list):
        message["tool_calls"] = item["tool_calls"]
    if isinstance(item.get("tool_call_id"), str):
        message["tool_call_id"] = item["tool_call_id"]
    return message


def _tool_item_name(item: dict[str, Any]) -> str:
    item_type = str(item.get("type") or "")
    if item_type in {"local_shell_call", "shell_call"}:
        return "local_shell" if item_type == "local_shell_call" else "shell"
    if item_type == "custom_tool_call":
        return str(item.get("name") or "custom_tool")
    return str(item.get("name") or "")


def _tool_item_arguments(item: dict[str, Any]) -> Any:
    item_type = str(item.get("type") or "")
    if item_type in {"local_shell_call", "shell_call"}:
        action = item.get("action") if isinstance(item.get("action"), dict) else {}
        return {
            "command": action.get("command") or item.get("command") or item.get("input") or "",
            "timeout_ms": action.get("timeout_ms") or item.get("timeout_ms"),
            "working_directory": action.get("working_directory") or item.get("working_directory"),
            "env": action.get("env") or item.get("env"),
        }
    if item_type == "custom_tool_call":
        return item.get("input") if "input" in item else item.get("arguments", "")
    return item.get("arguments") if "arguments" in item else item.get("input", {})


def _convert_tool_call_item(item: dict[str, Any]) -> dict[str, Any]:
    call_id = str(item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:12]}")
    name = _tool_item_name(item)
    arguments = _tool_item_arguments(item)
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, ensure_ascii=False)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }],
    }


def _convert_tool_call_output_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": str(item.get("call_id") or item.get("id") or ""),
        "content": _stringify_content(item.get("output") if "output" in item else item.get("content")),
    }


def _responses_input_to_messages(req_data: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = req_data.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})
    elif isinstance(instructions, list):
        text = _stringify_content(instructions).strip()
        if text:
            messages.append({"role": "system", "content": text})

    input_value = req_data.get("input", "")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
        return messages

    if isinstance(input_value, list):
        for item in input_value:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                messages.append({"role": "user", "content": str(item)})
                continue

            item_type = item.get("type")
            if item_type == "message" or "role" in item:
                message = _convert_response_message(item)
                if message is not None:
                    messages.append(message)
                continue
            if item_type in {"function_call", "custom_tool_call", "local_shell_call", "shell_call"}:
                messages.append(_convert_tool_call_item(item))
                continue
            if item_type in {"function_call_output", "custom_tool_call_output", "local_shell_call_output", "shell_call_output"}:
                messages.append(_convert_tool_call_output_item(item))
                continue
            messages.append({"role": "user", "content": _convert_message_content([item], role="user")})

    if not any(message.get("role") == "user" for message in messages):
        messages.append({"role": "user", "content": ""})
    return messages


def _normalize_responses_tool(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    tool_type = str(tool.get("type") or "")
    if tool_type in {"shell", "local_shell"}:
        return {
            "type": "function",
            "function": {
                "name": tool_type,
                "description": str(tool.get("description") or f"Run a {tool_type} command."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout_ms": {"type": "integer"},
                        "working_directory": {"type": "string"},
                        "env": {"type": "object"},
                    },
                    "required": ["command"],
                },
            },
        }
    if tool_type in {"custom", "custom_tool"}:
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip():
            name = "custom_tool"
        return {
            "type": "function",
            "function": {
                "name": name.strip(),
                "description": str(tool.get("description") or "Custom text tool."),
                "parameters": tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                },
            },
        }
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        fn = tool["function"]
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            return None
        return {
            "type": "function",
            "function": {
                "name": name.strip(),
                "description": str(fn.get("description") or tool.get("description") or ""),
                "parameters": fn.get("parameters") or fn.get("input_schema") or tool.get("parameters") or {},
            },
        }

    name = tool.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    parameters = tool.get("parameters") or tool.get("input_schema") or {}
    if not isinstance(parameters, dict):
        parameters = {}
    return {
        "type": "function",
        "function": {
            "name": name.strip(),
            "description": str(tool.get("description") or ""),
            "parameters": parameters,
        },
    }


def _responses_to_chat_payload(req_data: dict[str, Any]) -> dict[str, Any]:
    metadata = req_data.get("metadata") if isinstance(req_data.get("metadata"), dict) else {}
    chat_payload: dict[str, Any] = {
        "model": req_data.get("model") or "gpt-5",
        "messages": _responses_input_to_messages(req_data),
        "stream": bool(req_data.get("stream", False)),
        "metadata": metadata,
    }
    tools = [_normalize_responses_tool(tool) for tool in (req_data.get("tools") or [])]
    chat_payload["tools"] = [tool for tool in tools if tool is not None]

    for key in ("session_key", "conversation_id", "_workspace_root", "upstream_files"):
        if key in req_data:
            chat_payload[key] = req_data[key]
    if req_data.get("previous_response_id") and not chat_payload.get("session_key"):
        chat_payload["session_key"] = req_data["previous_response_id"]
    if req_data.get("store") is not None:
        chat_payload["store"] = req_data.get("store")
    return chat_payload


def _build_standard_request(req_data: dict[str, Any], *, client_profile: str) -> StandardRequest:
    standard_request = build_chat_standard_request(
        req_data,
        default_model="gpt-5",
        surface="responses",
        client_profile=client_profile,
    )
    standard_request.requested_model = req_data.get("model")
    log.info("[Responses] normalized tools=%s profile=%s", standard_request.tool_names, client_profile)
    return standard_request


def _response_base(*, response_id: str, created: int, model_name: str) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "in_progress",
        "model": model_name,
        "output": [],
        "parallel_tool_calls": True,
        "error": None,
        "incomplete_details": None,
    }


def _response_usage(prompt: str, output_text: str) -> dict[str, Any]:
    input_tokens = len(prompt or "")
    output_tokens = len(output_text or "")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


def _tool_block_output_type(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in {"local_shell", "local_shell_call"}:
        return "local_shell_call"
    if lowered in {"shell", "shell_call"}:
        return "shell_call"
    if lowered in {"custom_tool", "custom_tool_call"}:
        return "custom_tool_call"
    return "function_call"


def _tool_block_to_response_item(block: dict[str, Any]) -> dict[str, Any]:
    arguments = block.get("input", {})
    name = str(block.get("name") or "")
    output_type = _tool_block_output_type(name)
    call_id = str(block.get("id") or f"call_{uuid.uuid4().hex[:12]}")

    if output_type in {"local_shell_call", "shell_call"}:
        if not isinstance(arguments, dict):
            arguments = {"command": str(arguments)}
        action = {
            "type": "exec",
            "command": str(arguments.get("command") or arguments.get("cmd") or arguments.get("input") or ""),
        }
        for key in ("timeout_ms", "working_directory", "env"):
            if arguments.get(key) is not None:
                action[key] = arguments[key]
        return {
            "id": f"lsc_{uuid.uuid4().hex[:12]}" if output_type == "local_shell_call" else f"sc_{uuid.uuid4().hex[:12]}",
            "type": output_type,
            "status": "completed",
            "call_id": call_id,
            "action": action,
        }

    if output_type == "custom_tool_call":
        input_value = arguments
        if not isinstance(input_value, str):
            input_value = json.dumps(input_value or {}, ensure_ascii=False)
        return {
            "id": f"ctc_{uuid.uuid4().hex[:12]}",
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": call_id,
            "name": name or "custom_tool",
            "input": input_value,
        }

    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, ensure_ascii=False)
    return {
        "id": f"fc_{uuid.uuid4().hex[:12]}",
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _tool_item_arguments_text(item: dict[str, Any]) -> str:
    item_type = item.get("type")
    if item_type in {"local_shell_call", "shell_call"}:
        return json.dumps(item.get("action") or {}, ensure_ascii=False)
    if item_type == "custom_tool_call":
        return str(item.get("input") or "")
    return str(item.get("arguments") or "")


def _tool_argument_event_names(item_type: str) -> tuple[str, str] | None:
    if item_type == "function_call":
        return "response.function_call_arguments.delta", "response.function_call_arguments.done"
    if item_type == "custom_tool_call":
        return "response.custom_tool_call_input.delta", "response.custom_tool_call_input.done"
    return None


def build_responses_payload(
    *,
    response_id: str,
    created: int,
    model_name: str,
    prompt: str,
    execution,
    standard_request: StandardRequest,
    directive=None,
) -> dict[str, Any]:
    directive = directive or build_tool_directive(standard_request, execution.state)
    visible_text = tool_directive_visible_text(directive, execution.state.answer_text)
    payload = _response_base(response_id=response_id, created=created, model_name=model_name)
    payload["status"] = "completed"
    payload["usage"] = _response_usage(prompt, visible_text)

    if directive.stop_reason == "tool_use":
        payload["output"] = [
            _tool_block_to_response_item(block)
            for block in directive.tool_blocks
            if block.get("type") == "tool_use"
        ]
        payload["output_text"] = ""
    else:
        output_text = visible_text
        content: list[dict[str, Any]] = []
        if execution.state.reasoning_text:
            content.append({"type": "reasoning_text", "text": sanitize_visible_text(execution.state.reasoning_text)})
        content.append({"type": "output_text", "text": output_text, "annotations": []})
        payload["output"] = [{
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": content,
        }]
        payload["output_text"] = output_text

    return payload


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = dict(data)
    payload.setdefault("type", event)
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class ResponsesStreamTranslator:
    def __init__(self, *, response_id: str, created: int, model_name: str, prompt: str, standard_request: StandardRequest) -> None:
        self.response_id = response_id
        self.created = created
        self.model_name = model_name
        self.prompt = prompt
        self.standard_request = standard_request
        self.message_id = f"msg_{uuid.uuid4().hex[:12]}"
        self.output_index = 0
        self.content_index = 0
        self.started_text = False
        self.pending_chunks: list[str] = []
        self.answer_fragments: list[str] = []
        self.visible_answer_fragments: list[str] = []
        self.reasoning_fragments: list[str] = []
        self.answer_sanitizer = VisibleTextSanitizer()
        self.reasoning_sanitizer = VisibleTextSanitizer()
        self.tool_calls_emitted = False

    def initial_chunks(self) -> list[str]:
        response = _response_base(response_id=self.response_id, created=self.created, model_name=self.model_name)
        return [_sse("response.created", {"response": response})]

    def _ensure_text_item(self) -> None:
        if self.started_text:
            return
        self.started_text = True
        item = {
            "id": self.message_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        self.pending_chunks.append(_sse("response.output_item.added", {
            "response_id": self.response_id,
            "output_index": self.output_index,
            "item": item,
        }))
        self.pending_chunks.append(_sse("response.content_part.added", {
            "response_id": self.response_id,
            "item_id": self.message_id,
            "output_index": self.output_index,
            "content_index": self.content_index,
            "part": {"type": "output_text", "text": "", "annotations": []},
        }))

    def on_delta(self, evt: dict[str, Any], text_chunk: str | None, tool_calls: list[dict[str, Any]] | None) -> None:
        if text_chunk and evt.get("phase") in ("think", "thinking_summary"):
            text_chunk = self.reasoning_sanitizer.feed(text_chunk)
            if not text_chunk:
                return
            self.reasoning_fragments.append(text_chunk)
            self.pending_chunks.append(_sse("response.reasoning_text.delta", {
                "response_id": self.response_id,
                "item_id": self.message_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "delta": text_chunk,
            }))
            return

        if text_chunk and evt.get("phase") == "answer":
            self.answer_fragments.append(text_chunk)
            text_chunk = self.answer_sanitizer.feed(text_chunk)
            if not text_chunk:
                return
            self._ensure_text_item()
            self.visible_answer_fragments.append(text_chunk)
            self.pending_chunks.append(_sse("response.output_text.delta", {
                "response_id": self.response_id,
                "item_id": self.message_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "delta": text_chunk,
            }))
            return

        if tool_calls:
            self.emit_tool_calls(tool_calls)

    def drain_pending(self) -> list[str]:
        chunks = self.pending_chunks
        self.pending_chunks = []
        return chunks

    def _flush_visible_sanitizers(self) -> None:
        reasoning_tail = self.reasoning_sanitizer.flush()
        if reasoning_tail:
            self.reasoning_fragments.append(reasoning_tail)
            self.pending_chunks.append(_sse("response.reasoning_text.delta", {
                "response_id": self.response_id,
                "item_id": self.message_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "delta": reasoning_tail,
            }))
        answer_tail = self.answer_sanitizer.flush()
        if answer_tail:
            self._ensure_text_item()
            self.visible_answer_fragments.append(answer_tail)
            self.pending_chunks.append(_sse("response.output_text.delta", {
                "response_id": self.response_id,
                "item_id": self.message_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "delta": answer_tail,
            }))

    def emit_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        for tool_call in tool_calls:
            block = {
                "type": "tool_use",
                "id": tool_call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "name": tool_call.get("name") or "",
                "input": tool_call.get("input") or {},
            }
            item = _tool_block_to_response_item(block)
            output_index = self.output_index
            self.output_index += 1
            self.pending_chunks.append(_sse("response.output_item.added", {
                "response_id": self.response_id,
                "output_index": output_index,
                "item": item,
            }))
            argument_events = _tool_argument_event_names(str(item.get("type") or ""))
            if argument_events is not None:
                delta_event, done_event = argument_events
                argument_text = _tool_item_arguments_text(item)
                self.pending_chunks.append(_sse(delta_event, {
                    "response_id": self.response_id,
                    "item_id": item["id"],
                    "output_index": output_index,
                    "delta": argument_text,
                }))
                done_payload_key = "input" if item.get("type") == "custom_tool_call" else "arguments"
                self.pending_chunks.append(_sse(done_event, {
                    "response_id": self.response_id,
                    "item_id": item["id"],
                    "output_index": output_index,
                    done_payload_key: argument_text,
                }))
            self.pending_chunks.append(_sse("response.output_item.done", {
                "response_id": self.response_id,
                "output_index": output_index,
                "item": item,
            }))
        if tool_calls:
            self.tool_calls_emitted = True

    def finalize(self, execution, directive) -> list[str]:
        self._flush_visible_sanitizers()
        chunks = self.drain_pending()
        final_text = tool_directive_visible_text(
            directive,
            execution.state.answer_text or "".join(self.answer_fragments),
        )

        if directive.stop_reason == "tool_use" and not self.tool_calls_emitted:
            tool_calls = [
                {"id": block["id"], "name": block["name"], "input": block.get("input", {})}
                for block in directive.tool_blocks
                if block.get("type") == "tool_use"
            ]
            self.emit_tool_calls(tool_calls)
            chunks.extend(self.drain_pending())
        elif not self.started_text:
            self._ensure_text_item()
            chunks.extend(self.drain_pending())
            if final_text:
                chunks.append(_sse("response.output_text.delta", {
                    "response_id": self.response_id,
                    "item_id": self.message_id,
                    "output_index": self.output_index,
                    "content_index": self.content_index,
                    "delta": final_text,
                }))
                self.answer_fragments.append(final_text)
                self.visible_answer_fragments.append(final_text)
            chunks.extend(self.drain_pending())
        elif directive.stop_reason != "tool_use":
            streamed_text = "".join(self.visible_answer_fragments)
            if final_text.startswith(streamed_text):
                missing_tail = final_text[len(streamed_text):]
                if missing_tail:
                    chunks.append(_sse("response.output_text.delta", {
                        "response_id": self.response_id,
                        "item_id": self.message_id,
                        "output_index": self.output_index,
                        "content_index": self.content_index,
                        "delta": missing_tail,
                    }))
                    self.answer_fragments.append(missing_tail)
                    self.visible_answer_fragments.append(missing_tail)

        if self.started_text:
            chunks.append(_sse("response.output_text.done", {
                "response_id": self.response_id,
                "item_id": self.message_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "text": final_text,
            }))
            chunks.append(_sse("response.content_part.done", {
                "response_id": self.response_id,
                "item_id": self.message_id,
                "output_index": self.output_index,
                "content_index": self.content_index,
                "part": {"type": "output_text", "text": final_text, "annotations": []},
            }))
            chunks.append(_sse("response.output_item.done", {
                "response_id": self.response_id,
                "output_index": self.output_index,
                "item": {
                    "id": self.message_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": final_text, "annotations": []}],
                },
            }))

        final_response = build_responses_payload(
            response_id=self.response_id,
            created=self.created,
            model_name=self.model_name,
            prompt=self.prompt,
            execution=execution,
            standard_request=self.standard_request,
            directive=directive,
        )
        chunks.append(_sse("response.completed", {"response": final_response}))
        chunks.append("data: [DONE]\n\n")
        return chunks


def _session_payload(req_data: dict[str, Any], original_req_data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(req_data)
    if original_req_data.get("previous_response_id") and not payload.get("session_key"):
        payload["session_key"] = original_req_data["previous_response_id"]
    return payload


@router.post("/responses")
@router.post("/v1/responses")
async def responses_create(request: Request):
    app = request.app
    users_db = app.state.users_db
    client: QwenClient = app.state.qwen_client

    auth = await resolve_auth_context(request, users_db)
    token = auth.token

    try:
        original_req_data = await request.json()
    except Exception:
        raise HTTPException(400, {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}})
    if not isinstance(original_req_data, dict):
        raise HTTPException(400, {"error": {"message": "JSON body must be an object", "type": "invalid_request_error"}})

    req_data = _responses_to_chat_payload(original_req_data)
    client_profile = detect_openai_client_profile(request.headers, req_data)
    session_key = derive_session_key("responses", token, req_data)
    original_history_messages = req_data.get("messages", [])

    file_store = getattr(app.state, "file_store", None)
    preprocessed = None
    if file_store is not None:
        preprocessed = await preprocess_attachments(req_data, file_store, owner_token=token)
        req_data = preprocessed.payload

    context_prepared = await prepare_context_attachments(
        app=app,
        payload=req_data,
        surface="responses",
        auth_token=token,
        client_profile=client_profile,
        existing_attachments=(preprocessed.attachments if preprocessed is not None else None),
    )
    req_data = context_prepared["payload"]
    standard_request = _build_standard_request(req_data, client_profile=client_profile)
    if preprocessed is not None:
        standard_request.attachments = preprocessed.attachments
        standard_request.uploaded_file_ids = preprocessed.uploaded_file_ids
    standard_request.upstream_files = context_prepared["upstream_files"]
    standard_request.session_key = context_prepared["session_key"]
    standard_request.context_mode = context_prepared["context_mode"]
    standard_request.bound_account_email = context_prepared["bound_account_email"]
    standard_request.bound_account = context_prepared["bound_account"]

    session_plan = await plan_persistent_session_turn(
        app=app,
        request=standard_request,
        payload=_session_payload(req_data, original_req_data),
        surface="responses",
    )
    if session_plan.enabled:
        standard_request.persistent_session = True
        standard_request.full_prompt = session_plan.full_prompt
        standard_request.prompt = session_plan.prompt
        standard_request.session_message_hashes = session_plan.current_hashes
        standard_request.upstream_chat_id = session_plan.existing_chat_id if session_plan.reuse_chat else None
        if standard_request.bound_account is None and session_plan.account_email:
            standard_request.bound_account = await app.state.account_pool.acquire_wait_preferred(session_plan.account_email, timeout=60)
            if standard_request.bound_account is not None:
                standard_request.bound_account_email = standard_request.bound_account.email
        elif standard_request.bound_account is not None and not standard_request.bound_account_email:
            standard_request.bound_account_email = standard_request.bound_account.email
        if standard_request.upstream_chat_id and standard_request.bound_account is None:
            log_session_plan_reuse_cancelled(
                request=standard_request,
                planned_chat_id=session_plan.existing_chat_id,
                reason="missing_bound_account",
            )
            standard_request.upstream_chat_id = None
            standard_request.prompt = standard_request.full_prompt or standard_request.prompt

    model_name = standard_request.response_model
    qwen_model = standard_request.resolved_model
    prompt = standard_request.prompt
    history_messages = original_history_messages
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    with request_context(req_id=new_request_id(), surface="responses", requested_model=model_name, resolved_model=qwen_model):
        test_markers = log_test_prompt(
            log,
            surface="responses",
            model=qwen_model,
            stream=standard_request.stream,
            tools=[tool.get("name") for tool in standard_request.tools],
            prompt=prompt,
        )
        log.info(
            "[Responses] model=%s stream=%s tool_enabled=%s profile=%s tools=%s prompt_len=%s prompt_tail=%r",
            qwen_model,
            standard_request.stream,
            standard_request.tool_enabled,
            standard_request.client_profile,
            [tool.get("name") for tool in standard_request.tools],
            len(prompt),
            prompt_tail(prompt),
        )

        if standard_request.stream:
            async def generate():
                async with app.state.session_locks.hold(session_key):
                    queue: asyncio.Queue[str | None] = asyncio.Queue()
                    translator = ResponsesStreamTranslator(
                        response_id=response_id,
                        created=created,
                        model_name=model_name,
                        prompt=prompt,
                        standard_request=standard_request,
                    )
                    for chunk in translator.initial_chunks():
                        await queue.put(chunk)

                    async def _emit_pending() -> None:
                        for pending in translator.drain_pending():
                            await queue.put(pending)

                    async def on_delta(evt: dict[str, Any], text_chunk: str | None, tool_calls: list[dict[str, Any]] | None) -> None:
                        translator.on_delta(evt, text_chunk, tool_calls)
                        await _emit_pending()

                    async def _run_request() -> None:
                        try:
                            update_request_context(stream_attempt=1)
                            result = await run_retryable_completion_bridge(
                                client=client,
                                standard_request=standard_request,
                                prompt=prompt,
                                users_db=users_db,
                                token=token,
                                history_messages=history_messages,
                                max_attempts=request_max_attempts(standard_request),
                                usage_delta_factory=build_usage_delta_factory(prompt),
                                allow_after_visible_output=False,
                                capture_events=False,
                                on_delta=on_delta,
                            )
                            execution = result.execution
                            directive = result.directive or build_tool_directive(standard_request, execution.state, history_messages=history_messages)
                            assistant_message = build_openai_assistant_history_message(
                                execution=execution,
                                request=standard_request,
                                directive=directive,
                            )
                            await persist_session_turn(
                                app=app,
                                request=standard_request,
                                surface="responses",
                                execution=execution,
                                assistant_message=assistant_message,
                            )
                            for chunk in translator.finalize(execution, directive):
                                await queue.put(chunk)
                        except HTTPException as he:
                            await clear_invalidated_session_chat(app=app, request=standard_request)
                            await queue.put(_sse("response.failed", {
                                "response": {
                                    **_response_base(response_id=response_id, created=created, model_name=model_name),
                                    "status": "failed",
                                    "error": he.detail,
                                }
                            }))
                        except Exception as e:
                            await clear_invalidated_session_chat(app=app, request=standard_request)
                            await queue.put(_sse("response.failed", {
                                "response": {
                                    **_response_base(response_id=response_id, created=created, model_name=model_name),
                                    "status": "failed",
                                    "error": {"message": str(e), "type": "server_error"},
                                }
                            }))
                        finally:
                            await queue.put(None)

                    task = asyncio.create_task(_run_request())
                    try:
                        while True:
                            chunk = await queue.get()
                            if chunk is None:
                                break
                            yield chunk
                        await task
                    finally:
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            async with app.state.session_locks.hold(session_key):
                update_request_context(stream_attempt=1)
                result = await run_retryable_completion_bridge(
                    client=client,
                    standard_request=standard_request,
                    prompt=prompt,
                    users_db=users_db,
                    token=token,
                    history_messages=history_messages,
                    max_attempts=request_max_attempts(standard_request),
                    usage_delta_factory=build_usage_delta_factory(prompt),
                    allow_after_visible_output=True,
                )
                execution = result.execution
                directive = result.directive or build_tool_directive(standard_request, execution.state, history_messages=history_messages)
                assistant_message = build_openai_assistant_history_message(
                    execution=execution,
                    request=standard_request,
                    directive=directive,
                )
                await persist_session_turn(
                    app=app,
                    request=standard_request,
                    surface="responses",
                    execution=execution,
                    assistant_message=assistant_message,
                )
                return JSONResponse(build_responses_payload(
                    response_id=response_id,
                    created=created,
                    model_name=model_name,
                    prompt=result.prompt,
                    execution=execution,
                    standard_request=standard_request,
                    directive=directive,
                ))
        except Exception as e:
            await clear_invalidated_session_chat(app=app, request=standard_request)
            raise HTTPException(status_code=500, detail=str(e))
