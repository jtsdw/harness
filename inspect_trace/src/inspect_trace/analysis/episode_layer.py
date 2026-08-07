"""目标二 episode 层: per-episode timing/success/cost.

Joins inspect_trace's `attempt_group` records with the real `.eval` log (for raw event
timestamps and `Score` -- inspect_trace itself never copies either, by design; see schema.py's
module docstring).

Critical-path latency: for a single episode this is *by definition* the same number as
end-to-end latency (both are "when did the whole thing finish") -- there's no separate algorithm
that produces a different total. What critical-path *analysis* actually gives you is an
attribution: which portions of that fixed total were on the necessary sequential chain vs. which
were concurrent slack that didn't add to the total. That's `total_busy_seconds` (deduplicated
union of all model+tool intervals) vs. `exclusive_model_seconds + exclusive_tool_seconds` (naive
sum, double-counts any real overlap): if the naive sum exceeds `total_busy_seconds`, the
difference (`concurrency_savings_seconds`) is exactly how much wall-clock time concurrency saved.
On every real run analyzed so far (see goal1_r3_r4_real_benchmark_findings.md: 98 opportunities
for overlap, 0 observed), `concurrency_savings_seconds` is 0 and `total_busy_seconds` is very
slightly less than `end_to_end_latency_seconds` (the residual is real model-waiting-for-tool /
tool-waiting-for-model gaps, themselves reported separately below).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from inspect_ai.log import read_eval_log
from inspect_ai.scorer import value_to_float

from ._loader import load_manifest, load_records_by_sample, records_of_kind
from .pricing import cost_usd


@dataclass
class EpisodeLayerSummary:
    sample_uuid: str
    sample_id: str | int | None
    eval_id: str
    end_to_end_latency_seconds: float | None
    critical_path_latency_seconds: float | None
    """Always equal to `end_to_end_latency_seconds` -- kept as a separate field only because
    efficient-harness.md names it separately; see module docstring for why there's no distinct
    computation for it."""
    total_busy_seconds: float | None
    """Deduplicated (union) time during which at least one model/tool event was actively running."""
    concurrency_savings_seconds: float
    """(exclusive_model + exclusive_tool) - total_busy_seconds. 0 whenever nothing overlapped."""
    exclusive_model_seconds: float
    exclusive_tool_seconds: float
    model_waiting_for_tool_seconds: float
    tool_waiting_for_model_seconds: float
    n_llm_calls: int
    n_tool_calls: int
    n_retries: int
    observed_parallel: bool
    success: bool | None
    cost_usd: float | None


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
    manifest = {m["eval_id"]: m for m in load_manifest(trace_dir)}

    # group sample_uuids by which eval log they came from, so each .eval file is opened once
    eval_id_of_sample: dict[str, str] = {}
    for su, records in by_sample.items():
        if records:
            eval_id_of_sample[su] = records[0]["eval_id"]

    out: list[EpisodeLayerSummary] = []
    for eval_id, log_location in {
        m["eval_id"]: m["log_location"] for m in manifest.values()
    }.items():
        if log_location is None:
            continue
        log = read_eval_log(log_location)
        if not log.samples:
            continue
        for s in log.samples:
            assert s.uuid is not None, (
                "uuid is only unset before a sample is first recorded"
            )
            su = s.uuid
            if eval_id_of_sample.get(su) != eval_id:
                continue
            records = by_sample.get(su, [])
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
                end_to_end = episode_end - episode_start
                total_busy = _interval_union_seconds(all_intervals)
            else:
                end_to_end = None
                total_busy = None

            exclusive_model = sum(b - a for a, b in model_intervals)
            exclusive_tool = sum(b - a for a, b in tool_intervals)
            concurrency_savings = (
                max(0.0, (exclusive_model + exclusive_tool) - total_busy)
                if total_busy is not None
                else 0.0
            )
            observed_parallel = _has_real_overlap(model_intervals + tool_intervals)

            # model-waiting-for-tool / tool-waiting-for-model: gaps between a tool interval ending
            # and the next model interval starting, and vice versa, in chronological order.
            model_wait = 0.0
            tool_wait = 0.0
            timeline = sorted(
                [(a, b, "model") for a, b in model_intervals]
                + [(a, b, "tool") for a, b in tool_intervals]
            )
            for i in range(1, len(timeline)):
                prev_end = timeline[i - 1][1]
                cur_start, _, cur_kind = timeline[i]
                gap = max(0.0, cur_start - prev_end)
                if cur_kind == "model":
                    model_wait += gap
                else:
                    tool_wait += gap

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

            out.append(
                EpisodeLayerSummary(
                    sample_uuid=su,
                    sample_id=s.id,
                    eval_id=eval_id,
                    end_to_end_latency_seconds=end_to_end,
                    critical_path_latency_seconds=end_to_end,
                    total_busy_seconds=total_busy,
                    concurrency_savings_seconds=concurrency_savings,
                    exclusive_model_seconds=exclusive_model,
                    exclusive_tool_seconds=exclusive_tool,
                    model_waiting_for_tool_seconds=model_wait,
                    tool_waiting_for_model_seconds=tool_wait,
                    n_llm_calls=len(attribution),
                    n_tool_calls=len(tool_intervals),
                    n_retries=n_retries,
                    observed_parallel=observed_parallel,
                    success=success,
                    cost_usd=cost,
                )
            )
    return out


_P99_MIN_SAMPLES = 100
"""需求 B 类验收标准: "样本不足时不报告 P99" -- P99 is only meaningful with roughly this many
points (fewer than this and the 99th-percentile point is extrapolated past the actual data)."""


def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolated percentile (matches numpy's default `interpolation="linear"`)."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = p / 100 * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac


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
    mean_end_to_end_latency_seconds: float | None
    p50_end_to_end_latency_seconds: float | None
    p95_end_to_end_latency_seconds: float | None
    p99_end_to_end_latency_seconds: float | None
    """`None` whenever there are fewer than `_P99_MIN_SAMPLES` latency samples -- reporting a P99
    off a handful of points would just be relabeling the max, per B 类验收标准's "样本不足时不报告
    P99"."""
    mean_n_llm_calls: float
    mean_n_tool_calls: float
    total_retries: int
    episodes_with_observed_parallel: int
    per_episode: list[EpisodeLayerSummary] = field(default_factory=list)


def summarize_run(trace_dir: Path) -> EpisodeLayerRunSummary:
    per_episode = summarize_episode_layer(trace_dir)
    n = len(per_episode)
    scored = [e for e in per_episode if e.success is not None]
    successes = [e for e in scored if e.success]
    costs = [e.cost_usd for e in per_episode if e.cost_usd is not None]
    latencies = [
        e.end_to_end_latency_seconds
        for e in per_episode
        if e.end_to_end_latency_seconds is not None
    ]

    total_cost = sum(costs) if costs else None
    cost_per_success = (
        (total_cost / len(successes)) if total_cost is not None and successes else None
    )
    sorted_latencies = sorted(latencies)

    return EpisodeLayerRunSummary(
        n_episodes=n,
        success_rate=(len(successes) / len(scored)) if scored else None,
        total_cost_usd=total_cost,
        cost_per_successful_episode_usd=cost_per_success,
        mean_end_to_end_latency_seconds=(sum(latencies) / len(latencies))
        if latencies
        else None,
        p50_end_to_end_latency_seconds=(
            _percentile(sorted_latencies, 50) if sorted_latencies else None
        ),
        p95_end_to_end_latency_seconds=(
            _percentile(sorted_latencies, 95) if sorted_latencies else None
        ),
        p99_end_to_end_latency_seconds=(
            _percentile(sorted_latencies, 99)
            if len(sorted_latencies) >= _P99_MIN_SAMPLES
            else None
        ),
        mean_n_llm_calls=(sum(e.n_llm_calls for e in per_episode) / n) if n else 0.0,
        mean_n_tool_calls=(sum(e.n_tool_calls for e in per_episode) / n) if n else 0.0,
        total_retries=sum(e.n_retries for e in per_episode),
        episodes_with_observed_parallel=sum(
            1 for e in per_episode if e.observed_parallel
        ),
        per_episode=per_episode,
    )
