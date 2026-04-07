import json
import logging
import re
import uuid

log = logging.getLogger("qwen2api.tool_parser")

def _find_tool_use_json(text: str, tool_names: set):
    """Find a tool_use JSON object in text. First tries exact name match, then any tool_use."""
    candidates = []
    i = 0
    while i < len(text):
        pos = text.find('{', i)
        if pos == -1:
            break
        depth = 0
        for j in range(pos, len(text)):
            if text[j] == '{': depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[pos:j+1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and obj.get("type") == "tool_use" and obj.get("name"):
                            candidates.append((pos, obj))
                    except (json.JSONDecodeError, ValueError):
                        pass
                    i = j
                    break
        i += 1

    if not candidates:
        return None

    best = None
    pos = 0
    for p, obj in candidates:
        tn = obj.get("name", "")
        if tn in tool_names:
            best = tn
            pos = p
            break
        if tool_names and next((n for n in tool_names if tn.lower() in n.lower() or n.lower() in tn.lower()), None):
            pos = p
            best = tn
            break
    if best is None and tool_names:
        pos, obj = candidates[0]
        best = next(iter(tool_names))  # use first available tool as last resort
    if best:
        obj = dict(obj)
        obj["name"] = best
    return pos, obj


def parse_tool_calls(answer: str, tools: list):
    if not tools:
        return [{"type": "text", "text": answer}], "end_turn"
    
    # normalize tools to get names
    tool_names = {t.get("name") or t.get("function", {}).get("name") for t in tools if t.get("name") or t.get("function", {}).get("name")}
    log.debug(f"[ToolParse] 原始回复({len(answer)}字): {answer[:200]!r}")

    def _make_tool_block(name, input_data, prefix=""):
        if name not in tool_names and tool_names:
            best = next((n for n in tool_names if name.lower() in n.lower() or n.lower() in name.lower()), None)
            name = best or next(iter(tool_names))
        tool_id = f"toolu_{uuid.uuid4().hex[:8]}"
        blocks = []
        if prefix and prefix.strip():
            blocks.append({"type": "text", "text": prefix})
        blocks.append({"type": "tool_use", "id": tool_id, "name": name, "input": input_data})
        return blocks, "tool_use"

    # 1. Primary: ✿ACTION✿...✿END_ACTION✿ (safe)
    tc_m = re.search(r'✿ACTION✿\s*(.*?)\s*✿END_ACTION✿', answer, re.DOTALL | re.IGNORECASE)
    if tc_m:
        try:
            obj = json.loads(tc_m.group(1))
            name = obj.get("action", obj.get("name", ""))
            inp = obj.get("args", obj.get("input", obj.get("arguments", obj.get("parameters", {}))))
            if isinstance(inp, str):
                try: inp = json.loads(inp)
                except: inp = {"value": inp}
            prefix = answer[:tc_m.start()].strip()
            log.info(f"[ToolParse] ✓ ✿ACTION✿ 格式: name={name!r}, input={str(inp)[:120]}")
            return _make_tool_block(name, inp, prefix)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"[ToolParse] ✿ACTION✿ 格式解析失败: {e}, content={tc_m.group(1)[:100]!r}")
            # 强制纠错
            name_m = re.search(r'"(?:action|name)"\s*:\s*"([^"]+)"', tc_m.group(1))
            name = name_m.group(1) if name_m else next(iter(tool_names)) if tool_names else "unknown"
            fake_input = {"_json_error": f"You generated invalid JSON in ACTION block. Error: {e}"}
            prefix = answer[:tc_m.start()].strip()
            return _make_tool_block(name, fake_input, prefix)

    # 1.5 Legacy: ##TOOL_CALL##...##END_CALL##
    tc_old = re.search(r'##TOOL_CALL##\s*(.*?)\s*##END_CALL##', answer, re.DOTALL | re.IGNORECASE)
    if tc_old:
        raw_json = tc_old.group(1).strip()
        try:
            obj = json.loads(raw_json)
            name = obj.get("name", "")
            inp = obj.get("input", obj.get("args", obj.get("arguments", obj.get("parameters", {}))))
            if isinstance(inp, str):
                try: inp = json.loads(inp)
                except: inp = {"value": inp}
            prefix = answer[:tc_old.start()].strip()
            log.info(f"[ToolParse] ✓ ##TOOL_CALL## 格式: name={name!r}, input={str(inp)[:120]}")
            return _make_tool_block(name, inp, prefix)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"[ToolParse] ##TOOL_CALL## 格式解析失败: {e}")
            pass

    # 2. XML: <tool_call>...</tool_call>
    xml_m = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', answer, re.DOTALL | re.IGNORECASE)
    if xml_m:
        try:
            obj = json.loads(xml_m.group(1))
            name = obj.get("name", obj.get("action", ""))
            inp = obj.get("input", obj.get("args", obj.get("arguments", obj.get("parameters", {}))))
            if isinstance(inp, str):
                try: inp = json.loads(inp)
                except: inp = {"value": inp}
            prefix = answer[:xml_m.start()].strip()
            log.info(f"[ToolParse] ✓ XML格式 <tool_call>: name={name!r}, input={str(inp)[:120]}")
            return _make_tool_block(name, inp, prefix)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"[ToolParse] XML格式解析失败: {e}, content={xml_m.group(1)[:100]!r}")

    # 2.5 Code block: ```tool_call\n...\n```
    cb_m = re.search(r'```tool_call\s*\n(.*?)\n```', answer, re.DOTALL)
    if cb_m:
        try:
            obj = json.loads(cb_m.group(1).strip())
            name = obj.get("name", "")
            inp = obj.get("input", obj.get("args", {}))
            if isinstance(inp, str):
                try: inp = json.loads(inp)
                except: inp = {"value": inp}
            prefix = answer[:cb_m.start()].strip()
            log.info(f"[ToolParse] ✓ 代码块格式 tool_call: name={name!r}, input={str(inp)[:120]}")
            return _make_tool_block(name, inp, prefix)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"[ToolParse] 代码块格式解析失败: {e}")

    # 3. Qwen native format: {"name":"...","arguments":"..."} (no "type" key)
    try:
        stripped_tmp = re.sub(r'```(?:json)?\s*\n?', '', answer)
        stripped_tmp = re.sub(r'\n?```', '', stripped_tmp).strip()
        if stripped_tmp.startswith('{') and '"name"' in stripped_tmp:
            obj = json.loads(stripped_tmp)
            if "name" in obj and "type" not in obj:
                name = obj.get("name", "")
                args = obj.get("arguments", obj.get("input", obj.get("parameters", {})))
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except: args = {"value": args}
                if name in tool_names or tool_names:
                    log.info(f"[ToolParse] ✓ Qwen原生格式: name={name!r}, args={str(args)[:120]}")
                    return _make_tool_block(name, args)
    except (json.JSONDecodeError, ValueError):
        pass

    # 4. Fallback: old {"type":"tool_use",...} JSON
    stripped = re.sub(r'```json\s*\n?', '', answer)
    stripped = re.sub(r'\n?```', '', stripped)
    result = _find_tool_use_json(stripped, tool_names)
    if result:
        pos, tool_call = result
        prefix = stripped[:pos].strip()
        tool_id = tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:8]}"
        log.info(f"[ToolParse] ✓ 旧JSON格式 tool_call: name={tool_call['name']!r}")
        blocks = []
        if prefix:
            blocks.append({"type": "text", "text": prefix})
        blocks.append({"type": "tool_use", "id": tool_id, "name": tool_call["name"], "input": tool_call.get("input", {})})
        return blocks, "tool_use"

    # 5. 极端保底拦截：只要看到文本中含有明显意图，但正则全部失败，强制触发纠偏
    if answer.strip() and tools:
        lower_ans = answer.lower()
        # 匹配明确的工具调用意图词
        
        found_intent = False
        # 只有在明确提到“使用/调用 xxx 工具/命令”时才拦截，避免普通对话被误杀
        if "tool" in lower_ans or "工具" in lower_ans:
            found_intent = True
        elif any(kw in lower_ans for kw in ["调用", "执行", "使用"]) and any(tn.lower() in lower_ans for tn in tool_names):
            found_intent = True
            
        if found_intent:
            log.warning(f"[ToolParse] 未匹配到正确格式，但检测到工具调用意图。强制阻断纯文本返回。")
            fallback_name = None
            for tn in tool_names:
                if tn.lower() in lower_ans:
                    fallback_name = tn
                    break
            if not fallback_name:
                fallback_name = next(iter(tool_names)) if tool_names else "unknown"
            
            return _make_tool_block(fallback_name, {"_error": "You MUST use ✿ACTION✿ syntax to call tools. Direct text or JSON is invalid. PLEASE RETRY. 必须使用 ✿ACTION✿ 格式调用工具。"})

    log.warning(f"[ToolParse] ✗ 未检测到工具调用，作为普通文本返回。工具列表: {tool_names}")
    
    # 终极防空指针：如果连 answer 都是空的，Claude Code 收到空 text 会崩溃
    # 注意：如果回答为空且我们正在进行强制重试阶段，我们不应随意将其转换为提示文字
    # 而是应该交给 anthropic.py 里的重试机制。但既然进入到了这里，说明重试已经耗尽，或者正常触发
    # 我们仍然需要返回一段非空的文本。
    text_content = answer if answer.strip() else "[模型思考完毕，但未能按照要求输出✿ACTION✿工具调用。请重新调整提示词。]"
    
    # 既然模型死活不调用，又没法报错给用户，就强制触发一个空操作的AskUserQuestion或者抛错，让工作流中断
    if not answer.strip():
        if "AskUserQuestion" in tool_names:
            log.warning("[ToolParse] 强制使用 AskUserQuestion 以阻断空循环。")
            return _make_tool_block("AskUserQuestion", {"questions": [{"question": "系统：大模型连续5次拒绝输出任何内容，可能已触发内置安全限制或死循环。建议：1. 简化你的报错日志；2. 直接告诉模型“修改文件代码”，不要让它分析。", "options": [{"label": "好的，我重试", "description": "明白了"}], "header": "模型装死拦截", "multiSelect": False}]})
        elif "Edit" in tool_names:
            return _make_tool_block("Edit", {"path": "/workspace/README.md", "command": "view", "start_line": 1, "end_line": 10})
        elif "Read" in tool_names:
            return _make_tool_block("Read", {"file_path": "/workspace/README.md"})
            
    return [{"type": "text", "text": text_content}], "end_turn"

def inject_format_reminder(prompt: str, tool_name: str) -> str:
    reminder = (
        f"[CORRECTION]: You called '{tool_name}' using the WRONG format — "
        f"the server BLOCKED it with 'Tool {tool_name} does not exists.'. "
        f"You MUST use ✿ACTION✿ format and NOTHING ELSE:\n"
        f"✿ACTION✿\n"
        f'{{"action": "{tool_name}", "args": {{...your args here...}}}}\n'
        f"✿END_ACTION✿\n"
        f"DO NOT use JSON without delimiters. DO NOT use any XML tags. ONLY ✿ACTION✿.\n"
    )
    prompt = prompt.rstrip()
    if prompt.endswith("Assistant: <think>"):
        return prompt[:-18] + reminder + "\nAssistant: <think>\n"
    elif prompt.endswith("Assistant:"):
        return prompt[:-10] + reminder + "\nAssistant: <think>\n"
    return prompt + "\n\n" + reminder + "\nAssistant: <think>\n"
