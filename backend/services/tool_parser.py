import json
import logging
import re
import uuid
from typing import Any, cast

from backend.adapter.standard_request import CLAUDE_CODE_OPENAI_PROFILE, OPENCLAW_OPENAI_PROFILE
from backend.core.request_logging import get_request_context
from backend.services.tool_arg_fixer import fix_tool_call_arguments
from backend.services.tool_name_obfuscation import from_qwen_name, to_qwen_name
from backend.toolcall.normalize import build_tool_name_registry, normalize_tool_name
from backend.toolcall.parser import parse_tool_calls_detailed
from backend.toolcall.formats_qnml import canonicalize_qnml_markup

__all__ = [
    "parse_tool_calls",
    "parse_tool_calls_detailed",
    "inject_format_reminder",
    "parse_tool_calls_silent",
    "extract_attempted_tool_name",
    "ToolSieve",
]

log = logging.getLogger("qwen2api.tool_parser")

QNML_TOOL_MARKERS = ("<|QNML|tool_calls", "</|QNML|tool_calls", "<|QNML|invoke")
LEGACY_XML_TOOL_MARKERS = ("<tool_calls", "</tool_calls", "<invoke", "<tool_call>", "</tool_call>")

_QWEN_SAFE_REQUIRED_ALIASES: dict[str, tuple[str, ...]] = {
    "fs_open_file": ("file_path",),
    "fs_put_file": ("file_path", "content"),
    "fs_patch_file": ("file_path",),
    "shell_run": ("command",),
}


CASE_SENSITIVE_TOOL_NAMES = {"Bash", "Edit", "Write", "Read", "Grep", "Glob", "WebFetch", "WebSearch"}


def _normalize_tool_name_case(name: str, tool_names: set[str]) -> str:
    if not isinstance(name, str) or not name:
        return name
    if name in tool_names:
        return name
    lowered = name.lower()
    for candidate in tool_names:
        if candidate.lower() == lowered:
            if candidate in CASE_SENSITIVE_TOOL_NAMES:
                return candidate
            return candidate
    return name


def _find_tool_use_json(text: str, tool_names: set[str]):
    i = 0
    while i < len(text):
        pos = text.find('{', i)
        if pos == -1:
            break
        depth = 0
        for j in range(pos, len(text)):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[pos:j + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and obj.get("type") == "tool_use" and obj.get("name"):
                            normalized_name = normalize_tool_name(obj.get("name", ""), tool_names)
                            if normalized_name in tool_names:
                                obj = dict(obj)
                                obj["name"] = normalized_name
                                return pos, obj

                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
        i = pos + 1

    return None


def _extract_first_xml_tool_call(text: str) -> str | None:
    wrapped_match = re.search(r"<tool_calls>\s*(<tool_call>[\s\S]*?</tool_call>)\s*</tool_calls>", text, re.IGNORECASE)
    if wrapped_match:
        return wrapped_match.group(1)

    tool_call_match = re.search(r"<tool_call>\s*(\{[\s\S]*?\}|[\s\S]*?)\s*</tool_call>", text, re.IGNORECASE)
    if tool_call_match:
        return tool_call_match.group(0)
    return None


def _extract_first_json_tool_call(text: str) -> str | None:
    normalized = text.strip()

    # 优先查找完整的 JSON 对象
    # markers 按优先级：Qwen 官方 tool_calls 外层包装 > 单对象 > 松散片段
    markers = [
        '<|QNML|tool_calls',
        '<|QNML|invoke',
        '<tool_calls',
        '<invoke',
        '<tool_call>{"name"',
        '<tool_calls><tool_call>{"name"',
        '{"tool_calls"',
        '{"name"',
        '"name":',
        '"name="',
        'function.name:',
    ]
    start_positions = [normalized.find(marker) for marker in markers if normalized.find(marker) != -1]
    if not start_positions:
        return None
    start = min(start_positions)
    candidate = normalized[start:]

    wrapped_match = re.search(r"<tool_calls>\s*(<tool_call>[\s\S]*?</tool_call>)\s*</tool_calls>", candidate, re.IGNORECASE)
    if wrapped_match:
        return wrapped_match.group(1)

    tool_call_match = re.search(r"<tool_call>\s*(\{[\s\S]*?\}|[\s\S]*?)\s*</tool_call>", candidate, re.IGNORECASE)
    if tool_call_match:
        return tool_call_match.group(0)

    json_start = candidate.find("{")
    if json_start == -1:
        return None
    depth = 0
    for idx in range(json_start, len(candidate)):
        ch = candidate[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                json_str = candidate[json_start:idx + 1]
                # 验证是否是有效的工具调用 JSON
                try:
                    obj = json.loads(json_str)
                    if isinstance(obj, dict) and "name" in obj:
                        return json_str
                except (json.JSONDecodeError, ValueError):
                    pass
                return json_str
    return candidate[json_start:]


def _normalize_fragmented_tool_call(answer: str) -> str:
    text = answer.strip()
    if re.search(r"function\.name\s*:", text, flags=re.IGNORECASE):
        return text
    if ("##TOOL_CALL##" in text and "##END_CALL##" in text) or ("<|QNML|tool_calls" in text and "</|QNML|tool_calls" in text) or ("<tool_calls" in text and "</tool_calls" in text):
        return text

    extracted_tool_call = _extract_first_xml_tool_call(text) or _extract_first_json_tool_call(text)
    if extracted_tool_call:
        return extracted_tool_call

    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Tool\s+[A-Za-z0-9_.:-]*\s*does not exists?\\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```[\s\S]*?```", "", text)

    extracted_tool_call = _extract_first_xml_tool_call(text) or _extract_first_json_tool_call(text)
    if extracted_tool_call:
        return extracted_tool_call

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[•●·\-*]+\s*", "", line)
        line = line.replace("END_CALL##", "##END_CALL##")
        if line:
            lines.append(line)

    normalized = "\n".join(lines)
    if "TOOL_CALL##" in normalized and "##TOOL_CALL##" not in normalized:
        normalized = normalized.replace("TOOL_CALL##", "##TOOL_CALL##")
    if "##END_CALL##" in normalized and "##TOOL_CALL##" not in normalized and '"name"' in normalized:
        normalized = f"##TOOL_CALL##\n{normalized}"
    return normalized


def _coerce_tool_input(name: str, input_data: Any, tools: list[dict[str, Any]]) -> Any:
    if not isinstance(input_data, dict):
        return input_data

    input_data = _coerce_tool_input_by_schema(name, input_data, tools)

    # 修正 AskUserQuestion 工具参数
    if name == "AskUserQuestion":
        fixed = dict(input_data)

        # 如果只有 question 字段，转换为 questions 数组
        if "question" in fixed and "questions" not in fixed:
            question_text = fixed.pop("question")
            fixed["questions"] = [{
                "question": question_text,
                "header": "Question",
                "options": [
                    {"label": "Yes", "description": "Confirm"},
                    {"label": "No", "description": "Decline"}
                ],
                "multiSelect": False
            }]
            log.info(f"[ToolCoerce] Fixed AskUserQuestion: converted 'question' to 'questions' array")

        # 确保 questions 是数组
        if "questions" in fixed:
            if not isinstance(fixed["questions"], list):
                fixed["questions"] = [fixed["questions"]]

            # 验证每个问题的格式
            for i, q in enumerate(fixed["questions"]):
                if not isinstance(q, dict):
                    continue

                # 确保有必需字段
                if "question" not in q:
                    q["question"] = "Please provide your input"
                if "header" not in q:
                    q["header"] = "Question"
                if "multiSelect" not in q:
                    q["multiSelect"] = False

                # 确保 options 格式正确
                if "options" not in q:
                    q["options"] = [
                        {"label": "Continue", "description": "Proceed"},
                        {"label": "Cancel", "description": "Stop"}
                    ]
                elif isinstance(q.get("options"), list):
                    for j, opt in enumerate(q["options"]):
                        if isinstance(opt, str):
                            q["options"][j] = {"label": opt, "description": opt}
                        elif isinstance(opt, dict):
                            if "label" not in opt:
                                opt["label"] = opt.get("description", f"Option {j+1}")
                            if "description" not in opt:
                                opt["description"] = opt.get("label", "")

        return fixed

    # 修正 Agent 工具参数
    if name == "Agent":
        fixed = dict(input_data)
        if "description" not in fixed:
            fixed["description"] = "Execute sub-task"
        if "prompt" not in fixed:
            fixed["prompt"] = fixed.get("description", "Execute the task")
        return fixed

    # 修正 Read 工具参数
    if name == "Read":
        fixed = dict(input_data)
        if "file_path" not in fixed:
            if "path" in fixed:
                fixed["file_path"] = fixed.pop("path")
            elif "filename" in fixed:
                fixed["file_path"] = fixed.pop("filename")
        return fixed

    # 修正 Bash 工具参数
    if name in {"Write", "Edit"}:
        fixed = dict(input_data)
        if "file_path" not in fixed:
            for alias in ("path", "target_file", "filename", "file"):
                if alias in fixed:
                    fixed["file_path"] = fixed.pop(alias)
                    break
        if name == "Write" and "content" not in fixed:
            for alias in ("text", "body", "data", "file_content", "contents", "value"):
                if alias in fixed:
                    fixed["content"] = fixed.pop(alias)
                    break
        return fixed

    if name == "Bash":
        fixed = dict(input_data)
        if "command" not in fixed:
            if "cmd" in fixed:
                fixed["command"] = fixed.pop("cmd")
            elif "script" in fixed:
                fixed["command"] = fixed.pop("script")
        return fixed

    # 原有的 query/queries 转换逻辑
    query_value = input_data.get("query")
    queries = input_data.get("queries")
    if query_value or "queries" not in input_data:
        return input_data
    if not any(isinstance(tool, dict) and isinstance(tool.get("parameters"), dict) and isinstance(tool["parameters"].get("properties"), dict) and "query" in tool["parameters"]["properties"] for tool in tools):
        return input_data

    if isinstance(queries, list):
        merged = "\n".join(str(item).strip() for item in queries if str(item).strip())
        if merged:
            coerced = dict(input_data)
            coerced.pop("queries", None)
            coerced["query"] = merged
            return coerced
    if isinstance(queries, str) and queries.strip():
        coerced = dict(input_data)
        coerced.pop("queries", None)
        coerced["query"] = queries.strip()
        return coerced

    return input_data


def _tool_schema(tool_name: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        schema = tool.get("parameters") or tool.get("input_schema")
        function_payload = tool.get("function")
        if isinstance(function_payload, dict):
            name = name or function_payload.get("name")
            schema = schema or function_payload.get("parameters")
        if name == tool_name and isinstance(schema, dict):
            return schema
    return {}


def _schema_types(schema: Any) -> set[str]:
    if not isinstance(schema, dict):
        return set()
    raw_type = schema.get("type")
    types: set[str] = set()
    if isinstance(raw_type, str):
        types.add(raw_type)
    elif isinstance(raw_type, list):
        types.update(item for item in raw_type if isinstance(item, str))
    if "properties" in schema:
        types.add("object")
    if "items" in schema:
        types.add("array")
    for variant_key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(variant_key)
        if isinstance(variants, list):
            for variant in variants:
                types.update(_schema_types(variant))
    return types


def _parse_json_string_for_schema(value: str, *, want_array: bool, want_object: bool) -> tuple[Any, bool]:
    stripped = value.strip()
    if not stripped:
        return value, False

    candidates = [stripped]
    # Qwen sometimes emits array fields as: {"id":"a"}, {"id":"b"}
    # Wrap that form only when the schema explicitly asks for an array.
    if want_array and not stripped.startswith("["):
        candidates.append(f"[{stripped}]")

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

        if want_array:
            if isinstance(parsed, list):
                return parsed, True
            if isinstance(parsed, dict):
                return [parsed], True
        if want_object and isinstance(parsed, dict):
            return parsed, True

    return value, False


def _coerce_value_by_schema(tool_name: str, key: str, value: Any, schema: Any) -> Any:
    if not isinstance(schema, dict):
        return value

    types = _schema_types(schema)
    want_array = "array" in types
    want_object = "object" in types

    if isinstance(value, str) and (want_array or want_object):
        parsed, changed = _parse_json_string_for_schema(value, want_array=want_array, want_object=want_object)
        if changed:
            log.info(
                "[ToolCoerce] schema decoded JSON string: tool=%s field=%s expected=%s",
                tool_name,
                key,
                ",".join(sorted(types)),
            )
            value = parsed

    if want_array:
        if isinstance(value, dict):
            value = [value]
        if isinstance(value, list):
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                return [_coerce_value_by_schema(tool_name, f"{key}[]", item, item_schema) for item in value]
        return value

    if want_object and isinstance(value, dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return value
        fixed = dict(value)
        for child_key, child_schema in properties.items():
            if child_key in fixed:
                fixed[child_key] = _coerce_value_by_schema(tool_name, f"{key}.{child_key}", fixed[child_key], child_schema)
        return fixed

    return value


def _coerce_tool_input_by_schema(tool_name: str, input_data: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    schema = _tool_schema(tool_name, tools)
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return input_data

    fixed = dict(input_data)
    for key, value in list(fixed.items()):
        property_schema = properties.get(key)
        if isinstance(property_schema, dict):
            fixed[key] = _coerce_value_by_schema(tool_name, key, value, property_schema)
    return fixed


def _tool_schema_required(tool_name: str, tools: list[dict[str, Any]]) -> tuple[str, ...]:
    required: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("name") != tool_name:
            continue
        schema = tool.get("parameters") or tool.get("input_schema") or {}
        if not isinstance(schema, dict):
            continue
        raw_required = schema.get("required") or []
        if isinstance(raw_required, list):
            required.extend(str(item) for item in raw_required if isinstance(item, str) and item)
        break
    required.extend(_QWEN_SAFE_REQUIRED_ALIASES.get(tool_name, ()))

    # Alias-normalized names are what the client receives after from_qwen_name().
    canonical_defaults = {
        "Read": ("file_path",),
        "Write": ("file_path", "content"),
        "Edit": ("file_path",),
        "Bash": ("command",),
    }
    required.extend(canonical_defaults.get(tool_name, ()))

    deduped: list[str] = []
    for key in required:
        if key not in deduped:
            deduped.append(key)
    return tuple(deduped)


def _missing_required_args(tool_name: str, input_data: Any, tools: list[dict[str, Any]]) -> list[str]:
    if not isinstance(input_data, dict):
        return []
    missing: list[str] = []
    for key in _tool_schema_required(tool_name, tools):
        value = input_data.get(key)
        if value is None:
            missing.append(key)
        elif isinstance(value, str) and not value.strip():
            missing.append(key)
    return missing


def extract_attempted_tool_name(text: str, tool_names: list[str] | set[str]) -> str | None:
    """Best-effort recovery of the intended tool from malformed QNML/JSON text."""
    if not text:
        return None
    allowed = {name for name in tool_names if isinstance(name, str) and name}
    if not allowed:
        return None

    aliases: dict[str, str] = {}
    for name in allowed:
        aliases[name] = name
        aliases[to_qwen_name(name)] = name
        aliases[from_qwen_name(name)] = name

    patterns = (
        r"<\s*(?:\|\s*)?QNML(?:\s*\|\s*|\s+)?invoke\b[^>]*?name\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s>|/]+))",
        r"<\s*invoke\b[^>]*?name\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s>/]+))",
        r'"name"\s*:\s*"([^"]+)"',
        r"function\.name\s*:\s*([^\s\n]+)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, canonicalize_qnml_markup(text), flags=re.IGNORECASE):
            raw = next((group for group in match.groups() if group), "")
            if not raw:
                continue
            candidate = raw.strip().strip("'\"")
            mapped = aliases.get(candidate) or from_qwen_name(candidate)
            normalized = normalize_tool_name(mapped, allowed)
            cased = _normalize_tool_name_case(normalized, allowed)
            if cased in allowed:
                return cased
    return None


def parse_tool_calls(answer: str, tools: list):
    return _parse_tool_calls(answer, tools, emit_logs=True)


def parse_tool_calls_silent(answer: str, tools: list):
    return _parse_tool_calls(answer, tools, emit_logs=False)


def _parse_tool_calls(answer: str, tools: list, *, emit_logs: bool):
    answer = _normalize_fragmented_tool_call(answer)
    ctx = get_request_context()
    req_tag = f"req={ctx.get('req_id', '-')} chat={ctx.get('chat_id', '-')}"
    if not tools:
        return [{"type": "text", "text": answer}], "end_turn"
    tool_names = {t.get("name") for t in tools if t.get("name")}
    tool_registry = build_tool_name_registry(tool_names)

    def _log_debug(message: str) -> None:
        if emit_logs:
            log.debug(message)

    def _log_info(message: str) -> None:
        if emit_logs:
            log.info(message)

    def _log_warning(message: str) -> None:
        if emit_logs:
            log.warning(message)

    # 强制记录原始输入用于调试（但遵守 emit_logs 开关：ToolSieve 流式解析每 chunk 都调一次，
    # 若无条件记录会刷 1000+ 行 [ToolParse]——只在 finalize/诊断场景打印）
    if emit_logs:
        log.info(f"[ToolParse] [{req_tag}] 原始回复({len(answer)}字): {answer[:500]!r}")

    def _make_tool_blocks(calls, prefix=""):
        blocks = []
        if prefix:
            blocks.append({"type": "text", "text": prefix})

        for call in calls:
            raw_name = call.get("name", "") if isinstance(call, dict) else ""
            input_data = call.get("input", {}) if isinstance(call, dict) else {}
            # 入站反混淆：Qwen 返回的别名（ReadX）→ 客户端原名（Read）。
            # 未知别名原样返回，不影响 Qwen 直接返回原名的兼容路径。
            name = from_qwen_name(raw_name)
            normalized_name = normalize_tool_name(name, tool_registry.values())
            cased_name = _normalize_tool_name_case(normalized_name, tool_names)
            if cased_name not in tool_names:
                _log_warning(f"[ToolParse] 工具名不匹配，回退为普通文本: name={name!r}, normalized={normalized_name!r}, cased={cased_name!r}, tools={tool_names}")
                return [{"type": "text", "text": answer}], "end_turn"
            coerced_input = _coerce_tool_input(cased_name, input_data, tools)
            # 智能引号修复 + Edit/StrReplace 的 old_string fuzzy 修复
            coerced_input = fix_tool_call_arguments(cased_name, coerced_input)
            missing_args = _missing_required_args(cased_name, coerced_input, tools)
            if missing_args:
                _log_warning(
                    f"[ToolParse] invalid tool args: tool={cased_name!r}, missing={missing_args}, "
                    f"input={json.dumps(coerced_input, ensure_ascii=False)[:200]}"
                )
                return [{"type": "text", "text": answer}], "end_turn"
            tool_id = f"toolu_{uuid.uuid4().hex[:8]}"
            blocks.append({"type": "tool_use", "id": tool_id, "name": cased_name, "input": coerced_input})
            _log_info(f"[ToolParse] 返回工具块: original={name!r}, normalized={normalized_name!r}, final={cased_name!r}, input={json.dumps(coerced_input, ensure_ascii=False)[:200]}")

        return blocks, "tool_use" if any(block.get("type") == "tool_use" for block in blocks) else "end_turn"

    def _make_tool_block(name, input_data, prefix=""):
        return _make_tool_blocks([{"name": name, "input": input_data}], prefix)

    detailed = parse_tool_calls_detailed(answer, tool_names)
    detailed_calls = cast(list[dict[str, Any]], detailed["calls"])
    if detailed_calls:
        _log_info(
            f"[ToolParse] ✓ 详细解析格式: source={detailed['source']}, "
            f"calls={len(detailed_calls)}, tools={[c.get('name') for c in detailed_calls]}"
        )
        return _make_tool_blocks(detailed_calls)

    tc_m = re.search(r'##TOOL_CALL##\s*(.*?)\s*##END_CALL##', answer, re.DOTALL | re.IGNORECASE)
    if tc_m:
        try:
            obj = json.loads(tc_m.group(1))
            name = obj.get("name", "")
            inp = obj.get("input", obj.get("args", obj.get("arguments", obj.get("parameters", {}))))
            if isinstance(inp, str):
                try:
                    inp = json.loads(inp)
                except Exception:
                    inp = {"value": inp}
            prefix = answer[:tc_m.start()].strip()
            _log_info(f"[ToolParse] ✓ ##TOOL_CALL## 格式: name={name!r}, input={str(inp)[:120]}")
            return _make_tool_block(name, inp, prefix)
        except (json.JSONDecodeError, ValueError) as e:
            _log_warning(f"[ToolParse] ##TOOL_CALL## 格式解析失败: {e}, content={tc_m.group(1)[:100]!r}")

    xml_m = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', answer, re.DOTALL | re.IGNORECASE)
    if xml_m:
        try:
            obj = json.loads(xml_m.group(1))
            name = obj.get("name", "")
            inp = obj.get("input", obj.get("args", obj.get("arguments", obj.get("parameters", {}))))
            if isinstance(inp, str):
                try:
                    inp = json.loads(inp)
                except Exception:
                    inp = {"value": inp}
            prefix = answer[:xml_m.start()].strip()
            _log_info(f"[ToolParse] ✓ XML格式 <tool_call>: name={name!r}, input={str(inp)[:120]}")
            return _make_tool_block(name, inp, prefix)
        except (json.JSONDecodeError, ValueError) as e:
            _log_warning(f"[ToolParse] XML格式解析失败: {e}, content={xml_m.group(1)[:100]!r}")

    cb_m = re.search(r'```tool_call\s*\n(.*?)\n```', answer, re.DOTALL)
    if cb_m:
        try:
            obj = json.loads(cb_m.group(1).strip())
            name = obj.get("name", "")
            inp = obj.get("input", obj.get("args", {}))
            if isinstance(inp, str):
                try:
                    inp = json.loads(inp)
                except Exception:
                    inp = {"value": inp}
            prefix = answer[:cb_m.start()].strip()
            _log_info(f"[ToolParse] ✓ 代码块格式 tool_call: name={name!r}, input={str(inp)[:120]}")
            return _make_tool_block(name, inp, prefix)
        except (json.JSONDecodeError, ValueError) as e:
            _log_warning(f"[ToolParse] 代码块格式解析失败: {e}")

    stripped = re.sub(r'```json\s*\n?', '', answer)
    stripped = re.sub(r'\n?```', '', stripped)
    result = _find_tool_use_json(stripped, tool_names)
    if result:
        pos, tool_call = result
        prefix = stripped[:pos].strip()
        tool_id = tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:8]}"
        _log_info(f"[ToolParse] ✓ 旧JSON格式 tool_call: name={tool_call['name']!r}")
        blocks = []
        if prefix:
            blocks.append({"type": "text", "text": prefix})
        blocks.append({
            "type": "tool_use",
            "id": tool_id,
            "name": tool_call["name"],
            "input": _coerce_tool_input(tool_call["name"], tool_call.get("input", {}), tools),
        })
        return blocks, "tool_use"

    # 尝试解析纯 JSON 格式: {"name": "...", "input": {...}}
    stripped_clean = stripped.strip()
    try:
        if stripped_clean.startswith('{') and stripped_clean.endswith('}'):
            obj = json.loads(stripped_clean)
            if isinstance(obj, dict) and "name" in obj:
                name = obj.get("name", "")
                inp = obj.get("input", obj.get("args", obj.get("arguments", obj.get("parameters", {}))))
                if isinstance(inp, str):
                    try:
                        inp = json.loads(inp)
                    except Exception:
                        inp = {"value": inp}
                _log_info(f"[ToolParse] ✓ 纯JSON格式: name={name!r}, input={str(inp)[:120]}")
                return _make_tool_block(name, inp)
    except (json.JSONDecodeError, ValueError) as e:
        _log_debug(f"[ToolParse] 纯JSON格式解析失败: {e}, content={stripped_clean[:200]!r}")

    _log_warning(f"[ToolParse] ✗ 未检测到工具调用，作为普通文本返回。工具列表: {tool_names}")
    return [{"type": "text", "text": answer}], "end_turn"


class ToolSieve:
    """工具调用流式检测器 - 实时检测并分离工具调用"""

    def __init__(self, tool_names: list[str]):
        self.tool_names = set(tool_names) if tool_names else set()
        self.pending = ""
        self.capture = ""
        self.capturing = False
        self.pending_tool_calls = []
        self.tool_calls_detected = False
        self.in_markdown_fence = False
        self.fence_char = ""
        self.fence_len = 0

    def process_chunk(self, chunk: str) -> list[dict]:
        """
        处理一个chunk，返回事件列表
        事件类型：
        - {"type": "content", "text": "..."}  # 普通文本
        - {"type": "tool_calls", "calls": [...]}  # 工具调用
        """
        if not chunk:
            return []

        self.pending += chunk
        events = []

        # 如果正在捕获工具调用
        if self.capturing:
            self.capture += self.pending
            self.pending = ""

            # 尝试解析
            prefix, calls, suffix, ready = self._consume_tool_capture()

            if ready and calls:
                # Parsed successfully; emit tool calls immediately instead of waiting for flush.
                if prefix:
                    events.append({"type": "content", "text": prefix})

                self.pending_tool_calls = calls
                self.tool_calls_detected = True
                events.append({"type": "tool_calls", "calls": calls})
                self.pending_tool_calls = []
                self.pending = suffix
                self.capture = ""
                self.capturing = False

            return events

        # 检测工具调用开始
        start = self._find_tool_start(self.pending)

        if start >= 0:
            # 找到工具调用开始
            prefix = self.pending[:start]
            if prefix:
                events.append({"type": "content", "text": prefix})
                self._advance_markdown_fence_state(prefix)

            self.capture = self.pending[start:]
            self.pending = ""
            self.capturing = True

            # The current chunk may already contain a complete tool call; parse immediately.
            prefix2, calls2, suffix2, ready2 = self._consume_tool_capture()
            if ready2 and calls2:
                if prefix2:
                    events.append({"type": "content", "text": prefix2})
                self.pending_tool_calls = calls2
                self.tool_calls_detected = True
                events.append({"type": "tool_calls", "calls": calls2})
                self.pending_tool_calls = []
                self.pending = suffix2
                self.capture = ""
                self.capturing = False
        else:
            # 没找到，输出安全部分
            safe, hold = self._split_safe_content(self.pending)
            if safe:
                events.append({"type": "content", "text": safe})
                self._advance_markdown_fence_state(safe)
            self.pending = hold

        return events

    def _find_tool_start(self, text: str) -> int:
        """Find the earliest likely textual tool-call marker outside markdown examples."""
        if not text:
            return -1

        in_fence = self.in_markdown_fence
        fence_char = self.fence_char
        fence_len = self.fence_len
        line_start = 0
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "\n":
                line_start = i + 1
                i += 1
                continue

            at_line_indent = text[line_start:i].strip(" \t\r") == ""
            if at_line_indent and (text.startswith("```", i) or text.startswith("~~~", i)):
                run_char = text[i]
                run_len = 0
                while i + run_len < len(text) and text[i + run_len] == run_char:
                    run_len += 1
                if run_len >= 3:
                    if not in_fence:
                        in_fence = True
                        fence_char = run_char
                        fence_len = run_len
                    elif run_char == fence_char and run_len >= fence_len:
                        tail_end = text.find("\n", i)
                        tail = text[i + run_len:] if tail_end < 0 else text[i + run_len:tail_end]
                        if tail.strip() == "":
                            in_fence = False
                            fence_char = ""
                            fence_len = 0
                    next_newline = text.find("\n", i)
                    if next_newline < 0:
                        return -1
                    i = next_newline + 1
                    line_start = i
                    continue

            if in_fence:
                i += 1
                continue

            if ch == "`":
                run_len = 0
                while i + run_len < len(text) and text[i + run_len] == "`":
                    run_len += 1
                close = text.find("`" * run_len, i + run_len)
                newline = text.find("\n", i + run_len)
                if close >= 0 and (newline < 0 or close < newline):
                    i = close + run_len
                    continue
                # If a marker starts immediately after an unmatched backtick,
                # keep buffering; it is probably an inline-code example split
                # across chunks. A stray backtick followed by prose still allows
                # a later real tool call to be detected.
                immediate_tail = text[i + run_len:]
                if self._marker_match_start(immediate_tail) == 0:
                    return -1
                i += run_len
                continue

            if self._marker_match_start(text[i:]) == 0:
                return i
            i += 1

        return -1

    @staticmethod
    def _marker_match_start(text: str) -> int:
        patterns = (
            r"<\s*(?:\|\s*)?QNML(?:\s*\|\s*|\s+)?(?:tool_calls|tool-calls|toolcalls|invoke|parameter)?",
            r"＜\s*(?:\|\s*)?QNML",
            r"<\s*tool_calls\b",
            r"<\s*invoke\b",
            r"<\s*tool_call\b",
            r"\{\s*\"tool_calls\"",
            r"\{\s*\"name\"\s*:",
            r"##\s*TOOL_CALL##",
            r"function\.name\s*:",
        )
        for pat in patterns:
            m = re.match(pat, text, flags=re.IGNORECASE)
            if m:
                return m.start()
        return -1

    def _advance_markdown_fence_state(self, text: str) -> None:
        if not text:
            return
        for line in text.splitlines(keepends=True):
            stripped = line.lstrip(" \t\r")
            if not (stripped.startswith("```") or stripped.startswith("~~~")):
                continue
            run_char = stripped[0]
            run_len = len(stripped) - len(stripped.lstrip(run_char))
            if run_len < 3:
                continue
            if not self.in_markdown_fence:
                self.in_markdown_fence = True
                self.fence_char = run_char
                self.fence_len = run_len
                continue
            if run_char == self.fence_char and run_len >= self.fence_len:
                tail = stripped[run_len:].strip()
                if tail == "":
                    self.in_markdown_fence = False
                    self.fence_char = ""
                    self.fence_len = 0

    def _consume_tool_capture(self, *, force: bool = False) -> tuple[str, list, str, bool]:
        """尝试解析捕获的工具调用"""
        if not self.capture:
            return "", [], "", False

        if (
            not force
            and re.search(r"function\.name\s*:", self.capture, flags=re.IGNORECASE)
            and re.search(r"function\.arguments\s*:", self.capture, flags=re.IGNORECASE)
            and not re.search(r"(?m)\n\s*[\]}]\s*$", self.capture)
        ):
            return "", [], "", False

        # 尝试解析工具调用
        try:
            # 使用现有的解析逻辑
            blocks, stop_reason = parse_tool_calls_silent(self.capture,
                [{"name": name} for name in self.tool_names])

            if stop_reason == "tool_use":
                # 找到工具��用
                tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
                if tool_blocks:
                    # 转换为标准格式
                    calls = [{
                        "name": tb["name"],
                        "input": tb["input"]
                    } for tb in tool_blocks]

                    # 提取前缀文本
                    text_blocks = [b for b in blocks if b.get("type") == "text"]
                    prefix = text_blocks[0]["text"] if text_blocks else ""

                    return prefix, calls, "", True
        except Exception as e:
            log.debug(f"[ToolSieve] 解析失败: {e}")

        # 还不完整或解析失败
        return "", [], "", False

    def _split_safe_content(self, text: str) -> tuple[str, str]:
        """Split safe content while holding enough tail for split QNML markers."""
        hold_start = self._inline_tool_example_hold_start(text)
        if hold_start >= 0:
            return text[:hold_start], text[hold_start:]

        hold_len = 64
        if len(text) <= hold_len:
            return "", text

        return text[:-hold_len], text[-hold_len:]

    @classmethod
    def _inline_tool_example_hold_start(cls, text: str) -> int:
        """Hold an inline-code tool example until its closing backtick arrives."""
        i = 0
        while i < len(text):
            pos = text.find("`", i)
            if pos < 0:
                return -1
            run_len = 0
            while pos + run_len < len(text) and text[pos + run_len] == "`":
                run_len += 1
            if run_len <= 0:
                return -1
            tail_start = pos + run_len
            if cls._marker_match_start(text[tail_start:]) == 0:
                close = text.find("`" * run_len, tail_start)
                newline = text.find("\n", tail_start)
                if close < 0 or (newline >= 0 and newline < close):
                    return pos
            i = tail_start
        return -1

    def flush(self) -> list[dict]:
        """刷新剩余内容"""
        events = []

        if self.pending_tool_calls:
            events.append({"type": "tool_calls", "calls": self.pending_tool_calls})
            self.pending_tool_calls = []

        if self.capturing and self.capture:
            # 尝试最后一次解析
            prefix, calls, suffix, ready = self._consume_tool_capture(force=True)
            if ready and calls:
                if prefix:
                    events.append({"type": "content", "text": prefix})
                events.append({"type": "tool_calls", "calls": calls})
                self.tool_calls_detected = True
                if suffix:
                    events.append({"type": "content", "text": suffix})
            else:
                # 解析失败，检查是否看起来像工具调用
                log.warning("[ToolSieve] dropped unparsed captured tool markup len=%d", len(self.capture))

        if self.pending:
            events.append({"type": "content", "text": self.pending})
            self._advance_markdown_fence_state(self.pending)

        self.pending = ""
        self.capture = ""
        self.capturing = False
        return events

    def _looks_like_incomplete_tool_call(self, text: str) -> bool:
        """Return true when text still resembles an incomplete tool call."""
        if self._find_tool_start(text) >= 0:
            return True
        canonical = canonicalize_qnml_markup(text).lower()
        return any(marker in canonical for marker in ('<|qnml', '<qnml', '<tool_calls', '</tool_calls', '<invoke', '{"tool_calls"', '{"name":', '<tool_call', '##tool_call##', 'function.name:'))

    def has_tool_calls(self) -> bool:
        """是否检测到工具调用"""
        return self.tool_calls_detected or bool(self.pending_tool_calls)


def inject_format_reminder(prompt: str, tool_name: str, *, client_profile: str = OPENCLAW_OPENAI_PROFILE) -> str:
    """Inject a QNML format correction reminder before the final Assistant tag."""
    reminder = (
        "[CORRECTION / 纠正]: Your previous output contained a forbidden hallucinated error phrase.\n"
        f"要调用 {tool_name}，只输出这个精确 QNML 格式，不要有其他文本：\n"
        f"To invoke {tool_name}, output ONLY this exact QNML format with NO other text:\n"
        "<|QNML|tool_calls>\n"
        f"  <|QNML|invoke name={json.dumps(tool_name)}>\n"
        "    <|QNML|parameter name=\"arg1\"><![CDATA[value1]]></|QNML|parameter>\n"
        "    <|QNML|parameter name=\"arg2\"><![CDATA[value2]]></|QNML|parameter>\n"
        "  </|QNML|invoke>\n"
        "</|QNML|tool_calls>\n\n"
        "ABSOLUTELY FORBIDDEN in your next output:\n"
        "- Any disclaimer about a tool being unavailable, missing, or unregistered\n"
        "- Any sentence claiming you are unable to run a function\n"
        "- Any apology for failing to invoke something\n"
        "These QNML blocks are plain TEXT MARKERS the proxy parses — not native function calls.\n"
    )
    prompt = prompt.rstrip()
    if prompt.endswith("Assistant:"):
        return prompt[: -len("Assistant:")] + reminder + "\nAssistant:"
    return prompt + "\n\n" + reminder + "\nAssistant:"


