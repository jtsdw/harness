"""Derived-fact record schemas.

Every record shares `TraceEnvelope`'s identity fields so it can be joined back to the original
`.eval` log via `sample_uuid` (`== EvalSample.uuid`) and, where applicable, `model_event_uuid`
(`== ModelEvent.uuid`). Records never copy raw message/tool-result content from the original log —
only hashes, counts, and estimates derived from it.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field


class TraceEnvelope(BaseModel):
    schema_version: int = 1
    run_id: str | None = None
    eval_set_id: str | None = None
    eval_id: str
    task_name: str | None = None
    sample_id: int | str | None = None
    sample_uuid: str
    epoch: int | None = None
    sample_attempt: int | None = None
    recorded_at: str


class PrefillDiffMessageDetail(BaseModel):
    index: int
    role: str
    message_id: str | None
    tool_call_id: str | None
    """`ChatMessageTool.tool_call_id` if this message is a tool result, else `None`. Stable join key
    back to the originating `ToolCall` and to the `action_parsing` record for the `ToolEvent` that
    produced this message's content."""
    content_hash: str
    status: Literal["new", "reused", "dup_in_step"]
    content_category: Literal["system_template", "conversation"]
    """Whether this message is the (typically static, always-resent) system prompt vs. ordinary
    conversation/tool-result content. Needed to attribute repeated-prefill cost to "system template"
    separately from organically growing history."""
    function: str | None
    estimated_tokens: int
    first_seen_model_event_uuid: str | None
    first_seen_step_index: int | None


class ToolReuseBreakdown(BaseModel):
    function: str
    reused_count: int
    reused_tokens_estimate: int


class PrefillDiffToolDetail(BaseModel):
    index: int
    tool_name: str
    content_hash: str
    status: Literal["new", "reused", "dup_in_step"]
    estimated_tokens: int
    first_seen_model_event_uuid: str | None
    first_seen_step_index: int | None


class PrefillDiffRecord(TraceEnvelope):
    kind: Literal["prefill_diff"] = "prefill_diff"
    model_event_uuid: str
    model_name: str
    step_index: int
    """1-based ordinal count of ModelEvents seen for this sample (includes failed retry attempts)."""
    total_messages: int
    new_messages: int
    reused_messages: int
    dup_in_step_messages: int
    new_tokens_estimate: int
    reused_tokens_estimate: int
    cumulative_distinct_messages: int
    tool_reuse_breakdown: list[ToolReuseBreakdown] = Field(default_factory=list)
    messages: list[PrefillDiffMessageDetail] = Field(default_factory=list)
    """The `event.tools` schema payload (tool name/description/parameters resent on every call) is
    tracked separately from messages, not folded into the fields above: it is a structurally
    different kind of repeated content (a static/semi-static schema blob, not growing conversation
    history), and conflating the two would make it impossible to tell which source dominates the
    repeated-prefill cost."""
    tools_total: int = 0
    tools_new: int = 0
    tools_reused: int = 0
    dup_in_step_tools: int = 0
    new_tool_tokens_estimate: int = 0
    reused_tool_tokens_estimate: int = 0
    tools: list[PrefillDiffToolDetail] = Field(default_factory=list)
    system_template_messages: int = 0
    system_template_new_tokens_estimate: int = 0
    system_template_reused_tokens_estimate: int = 0


class SegmentTokenRecord(TraceEnvelope):
    kind: Literal["segment_tokens"] = "segment_tokens"
    model_event_uuid: str
    model_name: str
    stop_reason: str | None
    cache: Literal["read", "write"] | None
    reasoning_estimated_tokens: int
    reasoning_block_count: int
    tool_call_estimated_tokens: int
    tool_call_count: int
    server_tool_use_estimated_tokens: int
    server_tool_use_count: int
    text_estimated_tokens: int
    text_block_count: int
    other_estimated_tokens: int
    estimated_output_tokens_total: int
    billed_input_tokens: int | None
    billed_output_tokens: int | None
    billed_reasoning_tokens: int | None
    billed_total_tokens: int | None
    retries: int | None


class TokenAttributionRecord(TraceEnvelope):
    kind: Literal["token_attribution"] = "token_attribution"
    """需求一: joins `prefill_diff` (input side) and `segment_tokens` (output side) into one row per
    `ModelEvent`. `billed_*` are real usage from the provider; every `*_tokens_estimate` field is a
    tiktoken estimate -- there is no native real per-segment split from any provider."""
    model_event_uuid: str
    model_name: str
    step_index: int
    billed_input_tokens: int | None
    billed_output_tokens: int | None
    billed_reasoning_tokens: int | None
    billed_total_tokens: int | None
    system_template_tokens_estimate: int
    tool_schema_tokens_estimate: int
    conversation_tokens_estimate: int
    reasoning_tokens_estimate: int
    tool_calling_tokens_estimate: int
    final_response_tokens_estimate: int


class AttemptDetail(BaseModel):
    attempt_number: int
    outcome: Literal["success", "error"]
    model_event_uuid: str | None = None
    error_type: str | None = None
    status_code: int | None = None
    wait_time_after: float | None = None


class AttemptGroupRecord(TraceEnvelope):
    kind: Literal["attempt_group"] = "attempt_group"
    model_name: str
    resolution: Literal["success", "error", "abandoned", "orphan_no_open_match"]
    total_attempts: int
    total_wasted_wait_time: float
    final_model_event_uuid: str | None
    attempts: list[AttemptDetail]
    first_attempt_started_at: str
    resolved_at: str | None


class ToolCallStage(BaseModel):
    stage_index: int
    parent_model_event_uuid: str
    tool_call_ids: list[str]
    tool_functions: list[str]
    tool_count: int
    stage_started_at: str
    stage_ended_at: str | None
    observed_parallel: bool
    """True iff two or more tool calls in this stage had genuinely overlapping [timestamp, completed]
    windows -- observed concurrency, not inspect_ai's declared `ToolDef.parallel` eligibility."""
    max_observed_concurrency: int
    tool_waiting_for_model_seconds: list[float]
    """Per tool call: gap between the parent ModelEvent completing and this ToolEvent starting."""
    tool_semaphore_wait_seconds: list[float | None]
    """Real value from inspect_ai (`ToolEvent.completed - ToolEvent.timestamp - ToolEvent.working_time`)
    -- time this tool call spent queued on a concurrency semaphore, not an inferred gap like the other
    two waiting fields."""
    model_waiting_for_tool_seconds: float | None
    """Gap between this stage's last tool completing and the next ModelEvent starting. `None` if this
    is the trajectory's last stage (no following ModelEvent)."""


class ExecutionTopologyRecord(TraceEnvelope):
    kind: Literal["execution_topology"] = "execution_topology"
    """需求三: one record per sample (emitted at on_sample_end, not per-event). No rollback field --
    inspect_ai has no mechanism that reverts TaskState and discards a branch (see module docstring in
    execution_topology.py for why); the closest real analog is a retry, already in `attempt_group`."""
    total_model_calls: int
    total_tool_calls: int
    total_stages: int
    linear: bool
    stages: list[ToolCallStage] = Field(default_factory=list)
    unmatched_tool_event_uuids: list[str] = Field(default_factory=list)


class ActionParsingRecord(TraceEnvelope):
    kind: Literal["action_parsing"] = "action_parsing"
    """需求四: one record per ToolEvent, capturing tool-call parsing/validation failures and the
    write-back join key (`tool_call_id`) into the next step's `prefill_diff` message."""
    tool_event_uuid: str | None
    tool_call_id: str
    function: str
    parent_model_event_uuid: str | None
    error_present: bool
    error_type: (
        Literal[
            "parsing",
            "timeout",
            "unicode_decode",
            "permission",
            "file_not_found",
            "is_a_directory",
            "limit",
            "approval",
            "cancelled",
            "unknown",
            "output_limit",
        ]
        | None
    )
    error_message: str | None
    is_parse_or_validation_error: bool
    observation_message_id: str | None
    working_time_seconds: float | None
    started_at: str
    completed_at: str | None


class VLLMMetricsRecord(TraceEnvelope):
    kind: Literal["vllm_metrics"] = "vllm_metrics"
    """目标二 model invocation 层: real per-request TTFT/ITL/e2e-latency delta-scraped from
    vLLM's own `/metrics` (see vllm_metrics.py's module docstring for the correlation
    methodology). Only emitted when the model being called is actually backed by a reachable
    vLLM `/metrics` endpoint -- absent entirely for hosted providers, not present-with-nulls."""
    model_event_uuid: str
    attribution_confidence: Literal["exact", "ambiguous", "no_new_observation"]
    """"exact": exactly one new histogram observation between the before/after scrape (requires
    serialized execution, i.e. --max-connections 1). "ambiguous": more than one observation
    landed in the window (concurrent calls), so the delta can't be attributed to this call alone.
    "no_new_observation": this call produced no new vLLM request at all (e.g. inspect_ai's own
    local cache hit)."""
    ttft_seconds: float | None
    itl_seconds_avg: float | None
    e2e_latency_seconds: float | None
    queue_depth_running_at_start: float | None
    queue_depth_waiting_at_start: float | None
    gpu_cache_usage_perc_at_end: float | None
    preemptions_delta: float | None


class ManifestRecord(BaseModel):
    run_id: str | None
    eval_set_id: str | None
    eval_id: str
    task_name: str | None
    log_location: str | None
    status: str


TraceRecord = Union[
    PrefillDiffRecord,
    SegmentTokenRecord,
    AttemptGroupRecord,
    TokenAttributionRecord,
    ExecutionTopologyRecord,
    ActionParsingRecord,
    VLLMMetricsRecord,
]
