"""End-to-end validation for 需求一 (token_attribution).

A pure recombination of prefill_diff and segment_tokens, so the assertions are exact equalities
against those two records' own fields, not independent re-derivations.
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


def test_token_attribution_is_a_pure_recombination(tmp_path: Path, monkeypatch) -> None:
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

    prefill_records = {
        r["model_event_uuid"]: r for r in records if r["kind"] == "prefill_diff"
    }
    segment_records = {
        r["model_event_uuid"]: r for r in records if r["kind"] == "segment_tokens"
    }
    attribution_records = [r for r in records if r["kind"] == "token_attribution"]

    assert len(attribution_records) == 3
    for record in attribution_records:
        prefill = prefill_records[record["model_event_uuid"]]
        segment = segment_records[record["model_event_uuid"]]

        # Input side: system_template + tool_schema + conversation must equal prefill_diff's own
        # message-side + tool-side totals -- no information should be lost or double-counted by
        # the recombination.
        input_total = (
            record["system_template_tokens_estimate"]
            + record["tool_schema_tokens_estimate"]
            + record["conversation_tokens_estimate"]
        )
        expected_input_total = (
            prefill["new_tokens_estimate"]
            + prefill["reused_tokens_estimate"]
            + prefill["new_tool_tokens_estimate"]
            + prefill["reused_tool_tokens_estimate"]
        )
        assert input_total == expected_input_total
        assert record["tool_schema_tokens_estimate"] == (
            prefill["new_tool_tokens_estimate"] + prefill["reused_tool_tokens_estimate"]
        )
        assert record["system_template_tokens_estimate"] == (
            prefill["system_template_new_tokens_estimate"]
            + prefill["system_template_reused_tokens_estimate"]
        )

        # Output side: must equal segment_tokens' own fields exactly (pure recombination).
        assert (
            record["reasoning_tokens_estimate"] == segment["reasoning_estimated_tokens"]
        )
        assert record["tool_calling_tokens_estimate"] == (
            segment["tool_call_estimated_tokens"]
            + segment["server_tool_use_estimated_tokens"]
        )
        assert (
            record["final_response_tokens_estimate"] == segment["text_estimated_tokens"]
        )

        assert record["billed_output_tokens"] == segment["billed_output_tokens"]
        assert record["billed_reasoning_tokens"] == segment["billed_reasoning_tokens"]
