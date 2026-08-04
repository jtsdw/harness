"""End-to-end validation for 缺口A (prefill diff) and 缺口B (segment tokens).

Driven through a real `eval()` run against `mockllm/model` so the assertions exercise the actual
Hooks entry-point wiring, not just the detector classes in isolation.
"""

import json
from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import includes
from inspect_ai.solver import basic_agent
from inspect_ai.tool import ToolCall, tool


@tool
def tool_a():
    async def execute(value: str):
        """Echo a value.

        Args:
            value (str): Value to echo.

        Returns:
            The same value.
        """
        return value

    return execute


def _assistant_output(*, reasoning: str, tool_call_id: str, text: str) -> ModelOutput:
    return ModelOutput(
        model="mockllm/model",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=[
                        ContentReasoning(reasoning=reasoning),
                        ContentText(text=text),
                    ],
                    model="mockllm/model",
                    source="generate",
                    tool_calls=[
                        ToolCall(
                            id=tool_call_id, function="tool_a", arguments={"value": "x"}
                        )
                    ],
                ),
                stop_reason="tool_calls",
            )
        ],
    )


def test_prefill_diff_and_segment_tokens(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INSPECT_TRACE_DIR", str(tmp_path))

    task = Task(
        dataset=[Sample(input="call tool_a twice then submit", target="done")],
        solver=basic_agent(tools=[tool_a()]),
        scorer=includes(),
        message_limit=10,
    )

    model = get_model(
        "mockllm/model",
        custom_outputs=[
            _assistant_output(
                reasoning="I should call tool_a first.",
                tool_call_id="call-1",
                text="calling tool_a",
            ),
            _assistant_output(
                reasoning="Let me call tool_a again.",
                tool_call_id="call-2",
                text="calling tool_a again",
            ),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "done"},
            ),
        ],
    )

    logs = eval(task, model=model)
    assert logs[0].status == "success"

    jsonl_files = list(tmp_path.rglob("sample-*.jsonl"))
    assert len(jsonl_files) == 1
    records = [json.loads(line) for line in jsonl_files[0].read_text().splitlines()]

    prefill_records = [r for r in records if r["kind"] == "prefill_diff"]
    segment_records = [r for r in records if r["kind"] == "segment_tokens"]

    assert len(prefill_records) == 3
    assert [r["step_index"] for r in prefill_records] == [1, 2, 3]

    # Step 1: nothing has been seen before, everything is new.
    assert prefill_records[0]["new_messages"] == prefill_records[0]["total_messages"]
    assert prefill_records[0]["reused_messages"] == 0

    # Step 3: step-1's tool_a result (already present in step 2's input) has now re-entered
    # context for a second time. Step 2's own tool_a result appears in step 3's input for the
    # first time (a step's own output is never part of its own input), so it counts as "new"
    # here, not "reused" -- reuse only accrues starting the step *after* a message first appears.
    step3_breakdown = {
        b["function"]: b for b in prefill_records[2]["tool_reuse_breakdown"]
    }
    assert step3_breakdown["tool_a"]["reused_count"] == 1

    # Tool *schema* tracking (separate from message tracking, added after a real BFCL benchmark
    # run showed the fixed tool-definition list -- resent unchanged on every call -- was invisible
    # to message-only diffing). basic_agent's tool list (tool_a + submit) never changes across
    # steps, so it should be all-new at step 1 and all-reused from step 2 onward.
    tools_total = prefill_records[0]["tools_total"]
    assert tools_total >= 2  # tool_a + the basic_agent-injected submit tool
    assert prefill_records[0]["tools_new"] == tools_total
    assert prefill_records[0]["tools_reused"] == 0
    for record in prefill_records[1:]:
        assert record["tools_total"] == tools_total
        assert record["tools_new"] == 0
        assert record["tools_reused"] == tools_total
    assert prefill_records[0]["new_tool_tokens_estimate"] > 0
    assert prefill_records[1]["reused_tool_tokens_estimate"] > 0

    # content_category (需求一/需求二): basic_agent() injects a default system_message() step when
    # no `init` is given, so message index 0 of every step's input is that system prompt.
    for record in prefill_records:
        system_messages = [m for m in record["messages"] if m["role"] == "system"]
        assert len(system_messages) == 1
        assert system_messages[0]["content_category"] == "system_template"
        assert record["system_template_messages"] == 1
        non_system_categories = {
            m["content_category"] for m in record["messages"] if m["role"] != "system"
        }
        assert non_system_categories == {"conversation"}
    assert prefill_records[0]["system_template_new_tokens_estimate"] > 0
    assert prefill_records[0]["system_template_reused_tokens_estimate"] == 0
    for record in prefill_records[1:]:
        assert record["system_template_new_tokens_estimate"] == 0
        assert record["system_template_reused_tokens_estimate"] > 0

    # tool_call_id (需求四's join key): a ChatMessageTool detail's tool_call_id must match the
    # ToolCall.id that produced it -- step 2's input contains step 1's tool_a result.
    step2_tool_messages = [
        m for m in prefill_records[1]["messages"] if m["function"] == "tool_a"
    ]
    assert len(step2_tool_messages) == 1
    assert step2_tool_messages[0]["tool_call_id"] == "call-1"
    non_tool_messages = [
        m for m in prefill_records[1]["messages"] if m["function"] is None
    ]
    assert all(m["tool_call_id"] is None for m in non_tool_messages)

    assert len(segment_records) == 3
    # Step 1 had a reasoning block and a tool call, no plain-text-only final answer.
    assert segment_records[0]["reasoning_estimated_tokens"] > 0
    assert segment_records[0]["tool_call_estimated_tokens"] > 0
    # Step 3 (submit) has a tool call but no reasoning block.
    assert segment_records[2]["reasoning_estimated_tokens"] == 0
    assert segment_records[2]["tool_call_estimated_tokens"] > 0
    # Real billed usage must be present and never overwritten by the estimate.
    assert segment_records[0]["billed_output_tokens"] is not None
