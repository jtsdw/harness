"""目标二 model invocation 层: real-time capture from vLLM's own `/metrics` Prometheus endpoint.

Confirmed by a real spike (see `goal2_design.md`) that our pinned `vllm==0.6.3.post1` exposes
per-request TTFT/inter-token-latency/e2e-latency as histograms (`vllm:time_to_first_token_seconds`,
`vllm:time_per_output_token_seconds`, `vllm:e2e_request_latency_seconds`), plus real-time gauges
(`vllm:num_requests_running`/`num_requests_waiting`/`gpu_cache_usage_perc`) and a cumulative
`vllm:num_preemptions_total` counter. None of this needs inspect_ai core changes: we scrape
`/metrics` ourselves from `on_before_model_generate` (a snapshot right before the call) and again
from the `ModelEvent` branch of `on_sample_event` (right after), and take the histogram delta.

Correlation depends on serialized execution (`MAX_CONNECTIONS=1`, this project's own established
convention for the local vLLM setup -- see goal1_r3_r4_real_benchmark_findings.md's background
section for why): with no concurrent request in flight, exactly one new histogram observation
(`_count` delta == 1) occurs between the before/after scrape, so its `_sum` delta is *exactly* this
call's value, not an estimate. If `_count` delta != 1 (concurrent calls happened, or this call was
served from inspect_ai's own local cache with no new vLLM request at all), we say so explicitly via
`attribution_confidence` rather than silently reporting a number that isn't really this call's.

Correlation key follows the same pattern as `attempt_groups.py` (and for the same reason:
`BeforeModelGenerate.sample_id` is per-attempt, not the stable `sample_uuid`) -- `(model_name,
hash_message_list(input))`, resolved when the matching `ModelEvent` arrives at `on_sample_event`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
from inspect_ai.event import ModelEvent
from inspect_ai.hooks import BeforeModelGenerate

from .hashing import hash_message_list

DEFAULT_METRICS_URL = "http://localhost:8000/metrics"


def metrics_url() -> str:
    return os.environ.get("INSPECT_TRACE_VLLM_METRICS_URL", DEFAULT_METRICS_URL)


_LINE_RE = re.compile(r"^(?P<name>[a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+(?P<value>\S+)$")


def parse_metrics_text(text: str) -> dict[str, float]:
    """Flat `metric_name -> last-seen value` map.

    We only ever have one `model_name` label value in play (one model served per local vLLM
    process), so dropping labels entirely and keeping "last value wins" per exact metric name is
    sufficient -- this is not a general Prometheus text-format parser, just enough of one for the
    handful of single-series metrics we read.
    """
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        try:
            values[m.group("name")] = float(m.group("value"))
        except ValueError:
            continue
    return values


@dataclass
class VLLMSnapshot:
    ttft_sum: float | None
    ttft_count: float | None
    itl_sum: float | None
    itl_count: float | None
    e2e_sum: float | None
    e2e_count: float | None
    num_requests_running: float | None
    num_requests_waiting: float | None
    gpu_cache_usage_perc: float | None
    num_preemptions_total: float | None


def _snapshot_from_values(values: dict[str, float]) -> VLLMSnapshot:
    return VLLMSnapshot(
        ttft_sum=values.get("vllm:time_to_first_token_seconds_sum"),
        ttft_count=values.get("vllm:time_to_first_token_seconds_count"),
        itl_sum=values.get("vllm:time_per_output_token_seconds_sum"),
        itl_count=values.get("vllm:time_per_output_token_seconds_count"),
        e2e_sum=values.get("vllm:e2e_request_latency_seconds_sum"),
        e2e_count=values.get("vllm:e2e_request_latency_seconds_count"),
        num_requests_running=values.get("vllm:num_requests_running"),
        num_requests_waiting=values.get("vllm:num_requests_waiting"),
        gpu_cache_usage_perc=values.get("vllm:gpu_cache_usage_perc"),
        num_preemptions_total=values.get("vllm:num_preemptions_total"),
    )


async def fetch_snapshot(client: httpx.AsyncClient) -> VLLMSnapshot | None:
    try:
        response = await client.get(metrics_url(), timeout=2.0)
        response.raise_for_status()
    except httpx.HTTPError:
        # Not running vLLM (hosted model, or local server not up) -- this layer simply doesn't
        # apply, not an error. Callers must not treat None as "zero".
        return None
    return _snapshot_from_values(parse_metrics_text(response.text))


class VLLMMetricsTracker:
    """Delta-scrapes vLLM `/metrics` around each model call.

    See module docstring for the correlation methodology. No per-sample cleanup: entries are
    keyed by `(model_name, input_hash)` with no sample context, and every attempt (including
    failed ones) reaches `sample_event()` per
    `attempt_groups.py`'s own confirmed observation of `on_sample_event` delivery -- so an entry
    only leaks if a call never completes at all (process crash mid-request). Not worth the extra
    bookkeeping to guard against that; a long-running eval could in principle accumulate a few
    stale entries, but this has not been observed in practice.
    """

    def __init__(self) -> None:
        self._before_by_key: dict[tuple[str, str], VLLMSnapshot] = {}
        self._client = httpx.AsyncClient()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def before_model_generate(self, data: BeforeModelGenerate) -> None:
        snapshot = await fetch_snapshot(self._client)
        if snapshot is None:
            return
        input_hash = hash_message_list(data.input)
        # A retry resends the same input -- overwriting here is correct: we want the snapshot
        # taken right before whichever attempt actually completes, not the first attempt's.
        self._before_by_key[(data.model_name, input_hash)] = snapshot

    async def sample_event(self, event: ModelEvent) -> dict[str, Any] | None:
        input_hash = hash_message_list(event.input)
        before = self._before_by_key.pop((event.model, input_hash), None)
        if before is None:
            # Not a vLLM-backed model (hosted provider), or metrics weren't reachable at
            # before_model_generate time.
            return None

        after = await fetch_snapshot(self._client)
        if after is None:
            return None

        ttft_count_delta = _delta(before.ttft_count, after.ttft_count)
        itl_count_delta = _delta(before.itl_count, after.itl_count)
        e2e_count_delta = _delta(before.e2e_count, after.e2e_count)

        exact = ttft_count_delta == 1 and e2e_count_delta == 1
        confidence = (
            "exact"
            if exact
            else ("no_new_observation" if ttft_count_delta == 0 else "ambiguous")
        )

        ttft_seconds = (
            _delta(before.ttft_sum, after.ttft_sum) if ttft_count_delta else None
        )
        e2e_seconds = _delta(before.e2e_sum, after.e2e_sum) if e2e_count_delta else None
        itl_sum_delta = _delta(before.itl_sum, after.itl_sum)
        itl_seconds_avg = (
            itl_sum_delta / itl_count_delta
            if itl_count_delta and itl_sum_delta is not None
            else None
        )

        return dict(
            attribution_confidence=confidence,
            ttft_seconds=ttft_seconds,
            itl_seconds_avg=itl_seconds_avg,
            e2e_latency_seconds=e2e_seconds,
            queue_depth_running_at_start=before.num_requests_running,
            queue_depth_waiting_at_start=before.num_requests_waiting,
            gpu_cache_usage_perc_at_end=after.gpu_cache_usage_perc,
            preemptions_delta=_delta(
                before.num_preemptions_total, after.num_preemptions_total
            ),
        )


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before
