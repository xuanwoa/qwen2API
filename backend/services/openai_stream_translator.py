from __future__ import annotations

import json
import re
from typing import Any, Callable

from backend.adapter.standard_request import CLAUDE_CODE_OPENAI_PROFILE, OPENCLAW_OPENAI_PROFILE
from backend.runtime.execution import RuntimeToolDirective, tool_directive_visible_text
from backend.runtime.visible_text import VisibleTextSanitizer
from backend.toolcall.parser import parse_tool_calls_detailed


STRICT_TOOL_TEXT_PREFIXES = ("{", "[", "`", "<")
BUFFERED_TOOL_CALLS_ONLY = "buffered_tool_calls_only"
DIRECTIVE_DRIVEN_TOOL_CALLS = "directive_driven_tool_calls"


class OpenAIStreamTranslator:
    def __init__(
        self,
        *,
        completion_id: str,
        created: int,
        model_name: str,
        client_profile: str,
        build_final_directive: Callable[[str], RuntimeToolDirective] | None = None,
        allowed_tool_names: list[str] | None = None,
    ):
        self.completion_id = completion_id
        self.created = created
        self.model_name = model_name
        self.client_profile = client_profile
        self.build_final_directive = build_final_directive
        self.allowed_tool_names = {name for name in (allowed_tool_names or []) if isinstance(name, str) and name}
        self.pending_chunks: list[str] = []
        self.role_chunk_sent = False
        self.emitted_tool_index = 0
        self.answer_fragments: list[str] = []
        self.buffered_toolish_fragments: list[str] = []
        self.pending_content_chunks: list[str] = []
        self.visible_answer_fragments: list[str] = []
        self.content_sanitizer = VisibleTextSanitizer()
        self.reasoning_sanitizer = VisibleTextSanitizer()
        self.tool_calls_emitted = False
        self.tool_text_detection_mode = self._resolve_tool_text_detection_mode(client_profile)
        self.tool_call_finalize_mode = self._resolve_tool_call_finalize_mode(client_profile)
        self.in_think_block = False
        self.pending_think_text = ""

    @staticmethod
    def _partial_tag_suffix_len(text: str, tag_prefixes: tuple[str, ...]) -> int:
        lowered = text.lower()
        best = 0
        for prefix in tag_prefixes:
            limit = min(len(prefix) - 1, len(lowered))
            for length in range(1, limit + 1):
                if lowered.endswith(prefix[:length]):
                    best = max(best, length)
        return best

    def _may_contain_think_marker(self, text_chunk: str) -> bool:
        if self.in_think_block or self.pending_think_text:
            return True
        lowered = text_chunk.lower()
        if "<think" in lowered or "</think" in lowered:
            return True
        return self._partial_tag_suffix_len(text_chunk, ("<think", "<thinking")) > 0

    def _emit_split_think_content(self, text_chunk: str) -> bool:
        """Split streamed <think>...</think> blocks even when tags cross chunks."""
        if not self._may_contain_think_marker(text_chunk):
            return False

        text = self.pending_think_text + text_chunk
        self.pending_think_text = ""
        emitted = False

        while text:
            if self.in_think_block:
                close_match = re.search(r"</think(?:ing)?>", text, flags=re.IGNORECASE)
                if close_match:
                    reasoning = text[:close_match.start()]
                    if reasoning:
                        self._emit_reasoning_chunk(reasoning)
                        emitted = True
                    text = text[close_match.end():]
                    self.in_think_block = False
                    emitted = True
                    continue

                hold = self._partial_tag_suffix_len(text, ("</think", "</thinking"))
                reasoning = text[:-hold] if hold else text
                if reasoning:
                    self._emit_reasoning_chunk(reasoning)
                    emitted = True
                self.pending_think_text = text[-hold:] if hold else ""
                break

            open_match = re.search(r"<think(?:ing)?[^>]*>", text, flags=re.IGNORECASE)
            if open_match:
                before = text[:open_match.start()]
                if before:
                    self._emit_content_chunk(before)
                    emitted = True
                text = text[open_match.end():]
                self.in_think_block = True
                emitted = True
                continue

            hold = self._partial_tag_suffix_len(text, ("<think", "<thinking"))
            before = text[:-hold] if hold else text
            if before:
                self._emit_content_chunk(before)
                emitted = True
            self.pending_think_text = text[-hold:] if hold else ""
            break

        return emitted or bool(self.pending_think_text) or self.in_think_block

    def _flush_pending_think_text(self) -> None:
        if not self.pending_think_text:
            return
        if self.in_think_block:
            self._emit_reasoning_chunk(self.pending_think_text)
        else:
            self._emit_content_chunk(self.pending_think_text)
        self.pending_think_text = ""
        self.in_think_block = False

    @staticmethod
    def _resolve_tool_text_detection_mode(client_profile: str) -> str:
        if client_profile == OPENCLAW_OPENAI_PROFILE:
            return "strict_prefix"
        return "accept_any_tool_syntax"

    @staticmethod
    def _resolve_tool_call_finalize_mode(client_profile: str) -> str:
        if client_profile == CLAUDE_CODE_OPENAI_PROFILE:
            return BUFFERED_TOOL_CALLS_ONLY
        return DIRECTIVE_DRIVEN_TOOL_CALLS

    def _looks_like_tool_output(self, text_chunk: str) -> bool:
        if not text_chunk:
            return False
        lowered = text_chunk.lower()
        common_markers = (
            "tool does not exists",
            "</think>",
            "<|qnml|tool_calls",
            "</|qnml|tool_calls",
            "<|qnml|invoke",
            "<tool_calls",
            "</tool_calls",
            "<invoke",
            "function.name:",
            "##tool_call##",
            "##end_call##",
            '"tool_calls"',
            '"function":',
        )
        if any(marker in lowered for marker in common_markers):
            return True
        if self.allowed_tool_names:
            detailed = parse_tool_calls_detailed(text_chunk, self.allowed_tool_names)
            if detailed.get("saw_tool_syntax"):
                if self.tool_text_detection_mode == "strict_prefix":
                    stripped = text_chunk.lstrip()
                    return stripped.startswith(STRICT_TOOL_TEXT_PREFIXES)
                return True
        return False

    def _should_finalize_tool_calls(self, directive: RuntimeToolDirective) -> bool:
        if directive.stop_reason != "tool_use":
            return False
        if self.tool_call_finalize_mode == BUFFERED_TOOL_CALLS_ONLY:
            return bool(self.buffered_toolish_fragments)
        return True

    def _ensure_role_chunk(self) -> None:
        if self.role_chunk_sent:
            return
        yield_payload = {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        self.pending_chunks.append(f"data: {json.dumps(yield_payload, ensure_ascii=False)}\n\n")
        self.role_chunk_sent = True

    def _append_content_chunk(self, text_chunk: str) -> None:
        if not text_chunk:
            return
        chunk = (
            f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'content': text_chunk}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
        )
        self.pending_chunks.append(chunk)
        self.pending_content_chunks.append(chunk)
        self.visible_answer_fragments.append(text_chunk)

    def _emit_content_chunk(self, text_chunk: str) -> str:
        sanitized = self.content_sanitizer.feed(text_chunk)
        self._append_content_chunk(sanitized)
        return sanitized

    def _flush_content_sanitizer(self) -> str:
        sanitized = self.content_sanitizer.flush()
        self._append_content_chunk(sanitized)
        return sanitized

    def _append_reasoning_chunk(self, text_chunk: str) -> None:
        if not text_chunk:
            return
        chunk = (
            f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'reasoning_content': text_chunk}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
        )
        self.pending_chunks.append(chunk)

    def _emit_reasoning_chunk(self, text_chunk: str) -> str:
        """把 Qwen 的思考内容以 DeepSeek R1 风格 reasoning_content 发出去，
        让网页端/客户端能显示推理过程。"""
        sanitized = self.reasoning_sanitizer.feed(text_chunk)
        self._append_reasoning_chunk(sanitized)
        return sanitized

    def _flush_reasoning_sanitizer(self) -> str:
        sanitized = self.reasoning_sanitizer.flush()
        self._append_reasoning_chunk(sanitized)
        return sanitized

    def _discard_pending_content_chunks(self) -> None:
        if not self.pending_content_chunks:
            return
        pending_content_ids = {id(chunk) for chunk in self.pending_content_chunks}
        self.pending_chunks = [chunk for chunk in self.pending_chunks if id(chunk) not in pending_content_ids]
        self.pending_content_chunks = []
        self.visible_answer_fragments = []
        self.content_sanitizer.reset()

    def drain_pending(self) -> list[str]:
        """Return and clear chunks that are safe to send immediately.

        The OpenAI route uses this during live SSE streaming so parsed upstream
        deltas are flushed to the client instead of waiting for finalization.
        """
        if not self.pending_chunks:
            return []
        chunks = self.pending_chunks
        self.pending_chunks = []
        self.pending_content_chunks = []
        return chunks

    def on_delta(self, evt: dict[str, Any], text_chunk: str | None, tool_calls: list[dict[str, Any]] | None) -> None:
        self._ensure_role_chunk()

        if text_chunk and evt.get("phase") in ("think", "thinking_summary"):
            # 把思考内容作为 reasoning_content 发给客户端（DeepSeek R1 风格）
            # 网页端 TestPage 会单独显示这段"推理过程"
            self._emit_reasoning_chunk(text_chunk)
            return

        if text_chunk and evt.get("phase") == "answer":
            self.answer_fragments.append(text_chunk)
            if self._emit_split_think_content(text_chunk):
                return
            if self._looks_like_tool_output(text_chunk):
                self.buffered_toolish_fragments.append(text_chunk)
            elif self.buffered_toolish_fragments:
                self.buffered_toolish_fragments.append(text_chunk)
            else:
                self._emit_content_chunk(text_chunk)
            return

        if tool_calls:
            self.emit_tool_calls(tool_calls)

    def emit_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        self._ensure_role_chunk()
        for tool_call in tool_calls:
            idx = self.emitted_tool_index
            self.emitted_tool_index += 1
            self.pending_chunks.append(
                f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': idx, 'id': tool_call['id'], 'type': 'function', 'function': {'name': tool_call['name'], 'arguments': json.dumps(tool_call['input'], ensure_ascii=False)}}]}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
            )
        if tool_calls:
            self.tool_calls_emitted = True

    def _emit_missing_safe_text(self, safe_text: str) -> bool:
        if not safe_text:
            return False
        streamed_text = "".join(self.visible_answer_fragments)
        if safe_text.startswith(streamed_text):
            missing_tail = safe_text[len(streamed_text):]
            if missing_tail:
                emitted = self._emit_content_chunk(missing_tail)
                self.answer_fragments.append(missing_tail)
                return bool(emitted)
            return False
        self._discard_pending_content_chunks()
        emitted = self._emit_content_chunk(safe_text)
        emitted += self._flush_content_sanitizer()
        self.answer_fragments = [safe_text]
        return bool(emitted)

    def finalize(self, finish_reason: str, directive: RuntimeToolDirective | None = None) -> list[str]:
        self._flush_pending_think_text()
        self._flush_content_sanitizer()
        self._flush_reasoning_sanitizer()
        final_finish_reason = finish_reason
        buffered_text = "".join(self.buffered_toolish_fragments)
        if directive is None and self.build_final_directive is not None and not self.tool_calls_emitted:
            directive = self.build_final_directive("".join(self.answer_fragments))
        if directive is not None and not self.tool_calls_emitted:
            if self._should_finalize_tool_calls(directive):
                self._discard_pending_content_chunks()
                tool_calls = [
                    {
                        "id": block["id"],
                        "name": block["name"],
                        "input": block.get("input", {}),
                    }
                    for block in directive.tool_blocks
                    if block.get("type") == "tool_use"
                ]
                if tool_calls:
                    self.emit_tool_calls(tool_calls)
                    final_finish_reason = "tool_calls"
            elif safe_text := tool_directive_visible_text(directive, ""):
                self._emit_missing_safe_text(safe_text)
                final_finish_reason = "stop"
            elif buffered_text:
                self._emit_content_chunk(buffered_text)
        elif buffered_text and not self.tool_calls_emitted:
            self._emit_content_chunk(buffered_text)

        self._flush_content_sanitizer()
        self._flush_reasoning_sanitizer()
        chunks = list(self.pending_chunks)
        chunks.append(
            f"data: {json.dumps({'id': self.completion_id, 'object': 'chat.completion.chunk', 'created': self.created, 'model': self.model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': final_finish_reason}]}, ensure_ascii=False)}\n\n"
        )
        chunks.append("data: [DONE]\n\n")
        return chunks
