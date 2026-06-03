from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.adapter.standard_request import StandardRequest
from backend.runtime.execution import (
    RuntimeAttemptState,
    build_tool_directive,
    evaluate_retry_directive,
    _extract_final_phrase,
    tool_directive_visible_text,
)


REPORT_PATH = r"C:\workspace\loop_guard_long_task_report.md"
FINAL_PHRASE = "LONG_TASK_VERIFICATION_DONE"
CHECKPOINTS = ("[CHECKPOINT-ONE]", "[CHECKPOINT-TWO]")


def _prompt() -> str:
    return (
        f"请写报告文件 {REPORT_PATH}，必须包含 "
        f"{CHECKPOINTS[0]} 和 {CHECKPOINTS[1]}。"
        f"最终回答最后一句必须是 {FINAL_PHRASE}"
    )


def _request() -> StandardRequest:
    return StandardRequest(
        prompt=_prompt(),
        response_model="test-model",
        resolved_model="test-model",
        surface="selftest",
        tools=[{"name": "Write"}],
        tool_names=["Write"],
        tool_enabled=True,
    )


def _history_with_written_report(content: str, *, call_id: str = "call_write_1") -> list[dict]:
    return [
        {"role": "user", "content": _prompt()},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": "Write",
                    "input": {"file_path": REPORT_PATH, "content": content},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": "File written successfully",
                }
            ],
        },
    ]


def check_final_phrase_is_appended_without_retry_when_artifact_markers_are_complete() -> None:
    state = RuntimeAttemptState(answer_text="Report written.", finish_reason="stop")
    retry = evaluate_retry_directive(
        request=_request(),
        current_prompt=_prompt(),
        history_messages=_history_with_written_report("\n".join(CHECKPOINTS)),
        attempt_index=0,
        max_attempts=4,
        state=state,
        allow_after_visible_output=True,
    )

    assert retry.retry is False
    assert retry.reason == "final_phrase_appended"
    assert state.answer_text.endswith(FINAL_PHRASE)

    directive = build_tool_directive(_request(), state, _history_with_written_report("\n".join(CHECKPOINTS)))
    assert tool_directive_visible_text(directive, state.answer_text).endswith(FINAL_PHRASE)


def check_missing_artifact_marker_retries_instead_of_pretending_complete() -> None:
    state = RuntimeAttemptState(answer_text=f"Report written.\n{FINAL_PHRASE}", finish_reason="stop")
    retry = evaluate_retry_directive(
        request=_request(),
        current_prompt=_prompt(),
        history_messages=_history_with_written_report(CHECKPOINTS[0]),
        attempt_index=0,
        max_attempts=4,
        state=state,
        allow_after_visible_output=True,
    )

    assert retry.retry is True
    assert retry.reason == "final_delivery_incomplete"
    assert CHECKPOINTS[1] in retry.next_prompt


def check_final_attempt_still_appends_missing_final_phrase() -> None:
    state = RuntimeAttemptState(answer_text="Report written.", finish_reason="stop")
    retry = evaluate_retry_directive(
        request=_request(),
        current_prompt=_prompt(),
        history_messages=_history_with_written_report("\n".join(CHECKPOINTS)),
        attempt_index=3,
        max_attempts=4,
        state=state,
        allow_after_visible_output=True,
    )

    assert retry.retry is False
    assert retry.reason == "final_phrase_appended"
    assert state.answer_text.endswith(FINAL_PHRASE)


def check_chinese_final_phrase_prefers_marker_over_report_filename() -> None:
    prompt = (
        "日志放在了E:\\千问2api\\qwen2API\\日志.log\n"
        "对话放在了E:\\千问2api\\qwen2API\\对话.log\n"
        "然后输出的文件在E:\\千问2api\\qwen2API\\loop_guard_long_task_report.md\n\n"
        "请完成长任务并写报告，报告必须包含 "
        "[CHECKPOINT-NO-INFINITE-LOOP] 和 [CHECKPOINT-NO-TRUNCATION]。\n"
        "最终回答最后一句必须是：LONG_TASK_VERIFICATION_DONE"
    )
    request = StandardRequest(
        prompt=prompt,
        response_model="test-model",
        resolved_model="test-model",
        surface="selftest",
        tools=[{"name": "Write"}],
        tool_names=["Write"],
        tool_enabled=True,
    )
    report_content = "\n".join(("[CHECKPOINT-NO-INFINITE-LOOP]", "[CHECKPOINT-NO-TRUNCATION]"))
    assert _extract_final_phrase(prompt) == FINAL_PHRASE
    assert _extract_final_phrase(
        "然后输出的文件在E:\\千问2api\\qwen2API\\loop_guard_long_task_report.md"
    ) == ""
    history = [
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_write_real",
                    "name": "Write",
                    "input": {
                        "file_path": r"E:\千问2api\qwen2API\loop_guard_long_task_report.md",
                        "content": report_content,
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_write_real",
                    "content": "已写入",
                }
            ],
        },
    ]
    state = RuntimeAttemptState(
        answer_text="所有测试结果均已汇总至 E:\\千问2api\\qwen2API\\loop_guard_long_task_report.md，任务完成。",
        finish_reason="stop",
    )

    retry = evaluate_retry_directive(
        request=request,
        current_prompt=prompt,
        history_messages=history,
        attempt_index=0,
        max_attempts=4,
        state=state,
        allow_after_visible_output=True,
    )

    assert retry.retry is False
    assert retry.reason == "final_phrase_appended"
    assert state.answer_text.endswith(FINAL_PHRASE)
    assert not state.answer_text.rstrip().endswith("loop_guard_long_task_report.md")


def check_same_path_write_guard_blocks_only_identical_payload() -> None:
    history = _history_with_written_report("first version")

    duplicate_state = RuntimeAttemptState(
        tool_calls=[
            {
                "id": "call_write_2",
                "name": "Write",
                "input": {"file_path": REPORT_PATH, "content": "first version"},
            }
        ],
        finish_reason="tool_use",
    )
    duplicate_directive = build_tool_directive(_request(), duplicate_state, history)
    assert duplicate_directive.stop_reason == "end_turn"
    assert tool_directive_visible_text(duplicate_directive, "").endswith("first version")

    next_write_state = RuntimeAttemptState(
        tool_calls=[
            {
                "id": "call_write_3",
                "name": "Write",
                "input": {"file_path": REPORT_PATH, "content": "second version"},
            }
        ],
        finish_reason="tool_use",
    )
    next_write_directive = build_tool_directive(_request(), next_write_state, history)
    assert next_write_directive.stop_reason == "tool_use"


def check_same_path_write_guard_survives_mixed_tool_results() -> None:
    history = [
        {"role": "user", "content": _prompt()},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_write_1",
                    "name": "Write",
                    "input": {"file_path": REPORT_PATH, "content": "first version"},
                },
                {
                    "type": "tool_use",
                    "id": "call_bash_1",
                    "name": "Bash",
                    "input": {"command": "python -m compileall -q backend"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_write_1",
                    "content": "File written successfully",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call_bash_1",
                    "content": "compile ok",
                },
            ],
        },
    ]

    duplicate_state = RuntimeAttemptState(
        tool_calls=[
            {
                "id": "call_write_2",
                "name": "Write",
                "input": {"file_path": REPORT_PATH, "content": "first version"},
            }
        ],
        finish_reason="tool_use",
    )
    duplicate_directive = build_tool_directive(_request(), duplicate_state, history)
    assert duplicate_directive.stop_reason == "end_turn"

    next_write_state = RuntimeAttemptState(
        tool_calls=[
            {
                "id": "call_write_3",
                "name": "Write",
                "input": {"file_path": REPORT_PATH, "content": "second version"},
            }
        ],
        finish_reason="tool_use",
    )
    next_write_directive = build_tool_directive(_request(), next_write_state, history)
    assert next_write_directive.stop_reason == "tool_use"


CHECKS = (
    check_final_phrase_is_appended_without_retry_when_artifact_markers_are_complete,
    check_missing_artifact_marker_retries_instead_of_pretending_complete,
    check_final_attempt_still_appends_missing_final_phrase,
    check_chinese_final_phrase_prefers_marker_over_report_filename,
    check_same_path_write_guard_blocks_only_identical_payload,
    check_same_path_write_guard_survives_mixed_tool_results,
)


def run_checks() -> None:
    for check in CHECKS:
        check()
        print(f"{check.__name__}: ok")


if __name__ == "__main__":
    run_checks()
