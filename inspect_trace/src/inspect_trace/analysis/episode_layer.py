"""目标二 episode 层: per-episode timing/success/cost.

Joins inspect_trace's `attempt_group` records with the real `.eval` log (for raw event
timestamps and `Score` -- inspect_trace itself never copies either, by design; see schema.py's
module docstring).

`model_tool_window_seconds` spans the first model/tool event start through the last event end.
`total_busy_seconds` is the deduplicated union of those intervals. The naive model-plus-tool sum
can double-count overlap; its excess over the union is `concurrency_savings_seconds`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from inspect_ai.scorer import value_to_float

from ._loader import load_eval_sample_index, load_records_by_sample, records_of_kind
from .pricing import cost_usd


@dataclass
class EpisodeLayerSummary:
    sample_uuid: str
    sample_id: str | int | None
    eval_id: str
    sample_end_to_end_latency_seconds: float | None
    sample_working_time_seconds: float | None
    model_tool_window_seconds: float | None
    total_busy_seconds: float | None
    """Deduplicated (union) time during which at least one model/tool event was actively running."""
    concurrency_savings_seconds: float
    """(exclusive_model + exclusive_tool) - total_busy_seconds. 0 whenever nothing overlapped."""
    exclusive_model_seconds: float
    exclusive_tool_seconds: float
    model_to_tool_gap_seconds: float
    tool_to_model_gap_seconds: float
    n_llm_calls: int
    n_tool_calls: int
    n_retries: int
    observed_parallel: bool
    success: bool | None
    cost_usd: float | None
    per_role_model_latency_seconds: dict[str, float] = field(default_factory=dict)
    per_role_model_working_seconds: dict[str, float] = field(default_factory=dict)
    per_role_model_calls: dict[str, int] = field(default_factory=dict)


def _interval_union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Total wall-clock time covered by a set of (possibly overlapping) intervals.

    This *is* critical-path time for a flat (non-nested) set of spans: overlapping work doesn't
    get double-counted, matching efficient-harness.md's "并行阶段不能简单相加" requirement.
    """
    if not intervals:
        return 0.0
    merged = sorted(intervals)
    total = 0.0
    cur_start, cur_end = merged[0]
    for start, end in merged[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def summarize_episode_layer(trace_dir: Path) -> list[EpisodeLayerSummary]:
    by_sample = load_records_by_sample(trace_dir)
    eval_index = load_eval_sample_index(trace_dir)

    out: list[EpisodeLayerSummary] = []
    for su, records in by_sample.items():
        s = eval_index.samples.get(su)
        if s is None or not records:
            continue
        eval_id = records[0]["eval_id"]
        attempts = records_of_kind(records, "attempt_group")
        attribution = records_of_kind(records, "token_attribution")

        model_intervals = []
        tool_intervals = []
        for e in s.events:
            if e.event == "model" and e.completed is not None:
                model_intervals.append(
                    (e.timestamp.timestamp(), e.completed.timestamp())
                )
            elif e.event == "tool" and e.completed is not None:
                tool_intervals.append(
                    (e.timestamp.timestamp(), e.completed.timestamp())
                )

        all_intervals = model_intervals + tool_intervals
        if all_intervals:
            episode_start = min(i[0] for i in all_intervals)
            episode_end = max(i[1] for i in all_intervals)
            model_tool_window = episode_end - episode_start
            total_busy = _interval_union_seconds(all_intervals)
        else:
            model_tool_window = None
            total_busy = None

        exclusive_model = sum(b - a for a, b in model_intervals)
        exclusive_tool = sum(b - a for a, b in tool_intervals)
        concurrency_savings = (
            max(0.0, (exclusive_model + exclusive_tool) - total_busy)
            if total_busy is not None
            else 0.0
        )
        observed_parallel = _has_real_overlap(model_intervals + tool_intervals)

        # Adjacent transition gaps, named by direction (previous kind -> current kind).
        model_to_tool_gap = 0.0
        tool_to_model_gap = 0.0
        timeline = sorted(
            [(a, b, "model") for a, b in model_intervals]
            + [(a, b, "tool") for a, b in tool_intervals]
        )
        for i in range(1, len(timeline)):
            prev_end = timeline[i - 1][1]
            prev_kind = timeline[i - 1][2]
            cur_start, _, cur_kind = timeline[i]
            gap = max(0.0, cur_start - prev_end)
            if prev_kind == "model" and cur_kind == "tool":
                model_to_tool_gap += gap
            elif prev_kind == "tool" and cur_kind == "model":
                tool_to_model_gap += gap

        # Don't hardcode a scorer name (e.g. "bfcl_scorer") -- take whichever scorer(s) this
        # task actually used and normalize via inspect_ai's own value_to_float (handles
        # numeric 0/1, "C"/"I", bool, etc. uniformly across benchmarks).
        success = None
        if s.scores:
            to_float = value_to_float()
            floats = [to_float(sc.value) for sc in s.scores.values()]
            if floats:
                success = (sum(floats) / len(floats)) >= 1.0

        total_input = sum((r["billed_input_tokens"] or 0) for r in attribution)
        total_output = sum((r["billed_output_tokens"] or 0) for r in attribution)
        model_name = attribution[0]["model_name"] if attribution else None
        cost = (
            cost_usd(model_name, total_input, total_output) if model_name else None
        )

        n_retries = sum(
            1
            for g in attempts
            if g["resolution"] in ("success", "error") and g["total_attempts"] > 1
        )

        role_calls = {role: 0 for role in ("predictor", "main", "unknown")}
        role_latency = {role: 0.0 for role in role_calls}
        role_working = {role: 0.0 for role in role_calls}
        for event in eval_index.model_events.values():
            if event.sample_uuid != su:
                continue
            role = event.role if event.role in role_calls else "unknown"
            role_calls[role] += 1
            if event.interval is not None:
                role_latency[role] += event.interval[1] - event.interval[0]
            if event.working_time is not None:
                role_working[role] += event.working_time

        out.append(
            EpisodeLayerSummary(
                sample_uuid=su,
                sample_id=s.id,
                eval_id=eval_id,
                sample_end_to_end_latency_seconds=s.total_time,
                sample_working_time_seconds=s.working_time,
                model_tool_window_seconds=model_tool_window,
                total_busy_seconds=total_busy,
                concurrency_savings_seconds=concurrency_savings,
                exclusive_model_seconds=exclusive_model,
                exclusive_tool_seconds=exclusive_tool,
                model_to_tool_gap_seconds=model_to_tool_gap,
                tool_to_model_gap_seconds=tool_to_model_gap,
                n_llm_calls=len(attribution),
                n_tool_calls=len(tool_intervals),
                n_retries=n_retries,
                observed_parallel=observed_parallel,
                success=success,
                cost_usd=cost,
                per_role_model_latency_seconds=role_latency,
                per_role_model_working_seconds=role_working,
                per_role_model_calls=role_calls,
            )
        )
    return out


def _has_real_overlap(intervals: list[tuple[float, float]]) -> bool:
    for i, (a_start, a_end) in enumerate(intervals):
        for b_start, b_end in intervals[i + 1 :]:
            if a_start < b_end and b_start < a_end:
                return True
    return False


@dataclass
class EpisodeLayerRunSummary:
    n_episodes: int
    success_rate: float | None
    total_cost_usd: float | None
    cost_per_successful_episode_usd: float | None
    mean_sample_end_to_end_latency_seconds: float | None
    mean_sample_working_time_seconds: float | None
    mean_model_tool_window_seconds: float | None
    mean_n_llm_calls: float
    mean_n_tool_calls: float
    total_retries: int
    episodes_with_observed_parallel: int
    per_role_model_calls: dict[str, int] = field(default_factory=dict)
    per_role_model_latency_seconds: dict[str, float] = field(default_factory=dict)
    per_role_model_working_seconds: dict[str, float] = field(default_factory=dict)
    per_episode: list[EpisodeLayerSummary] = field(default_factory=list)


def summarize_run(trace_dir: Path) -> EpisodeLayerRunSummary:
    per_episode = summarize_episode_layer(trace_dir)
    n = len(per_episode)
    scored = [e for e in per_episode if e.success is not None]
    successes = [e for e in scored if e.success]
    costs = [e.cost_usd for e in per_episode if e.cost_usd is not None]
    sample_latencies = [
        e.sample_end_to_end_latency_seconds
        for e in per_episode
        if e.sample_end_to_end_latency_seconds is not None
    ]
    sample_working_times = [
        e.sample_working_time_seconds
        for e in per_episode
        if e.sample_working_time_seconds is not None
    ]
    model_tool_windows = [
        e.model_tool_window_seconds
        for e in per_episode
        if e.model_tool_window_seconds is not None
    ]

    total_cost = sum(costs) if costs else None
    cost_per_success = (
        (total_cost / len(successes)) if total_cost is not None and successes else None
    )

    return EpisodeLayerRunSummary(
        n_episodes=n,
        success_rate=(len(successes) / len(scored)) if scored else None,
        total_cost_usd=total_cost,
        cost_per_successful_episode_usd=cost_per_success,
        mean_sample_end_to_end_latency_seconds=(
            sum(sample_latencies) / len(sample_latencies) if sample_latencies else None
        ),
        mean_sample_working_time_seconds=(
            sum(sample_working_times) / len(sample_working_times)
            if sample_working_times
            else None
        ),
        mean_model_tool_window_seconds=(
            sum(model_tool_windows) / len(model_tool_windows)
            if model_tool_windows
            else None
        ),
        mean_n_llm_calls=(sum(e.n_llm_calls for e in per_episode) / n) if n else 0.0,
        mean_n_tool_calls=(sum(e.n_tool_calls for e in per_episode) / n) if n else 0.0,
        total_retries=sum(e.n_retries for e in per_episode),
        episodes_with_observed_parallel=sum(
            1 for e in per_episode if e.observed_parallel
        ),
        per_role_model_calls=_sum_role_values(
            e.per_role_model_calls for e in per_episode
        ),
        per_role_model_latency_seconds=_sum_role_values(
            e.per_role_model_latency_seconds for e in per_episode
        ),
        per_role_model_working_seconds=_sum_role_values(
            e.per_role_model_working_seconds for e in per_episode
        ),
        per_episode=per_episode,
    )


def _sum_role_values(values):
    result = {role: 0 for role in ("predictor", "main", "unknown")}
    for value in values:
        for role, amount in value.items():
            result[role] = result.get(role, 0) + amount
    return result
