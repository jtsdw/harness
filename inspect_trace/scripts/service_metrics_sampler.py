#!/usr/bin/env python3
"""需求 B4 服务层: background poller for vLLM `/metrics` + `nvidia-smi`.

Independent of any inspect_ai hook lifecycle.

Why a separate process instead of piggybacking on the per-invocation hooks (`vllm_metrics.py`,
`vllm_per_request_metrics.py`): service state (queue depth, KV cache, preemptions, batch/iteration
tokens, GPU util/mem/power) changes *between* calls too, especially under the concurrency B5
actually tests (concurrency=4/8) -- a snapshot only taken right before/after each call would miss
exactly the queueing/preemption behavior B5 exists to observe. This runs on a fixed wall-clock
interval for the duration of a benchmark, independent of how many model calls happen to land in
that window.

Also self-reports its own poll latency (B4's "指标采集本身的 CPU/latency overhead" line item) --
each record's `poll_duration_seconds` is how long this one poll (vLLM /metrics fetch + nvidia-smi
call) took, not an estimate.

Field names below were confirmed by reading vLLM's actual current `/metrics` definitions on GitHub
(`vllm/v1/metrics/loggers.py`, read 2026-08-07 after docs.vllm.ai kept 429-ing): `vllm:prefix_cache_queries`
/ `vllm:prefix_cache_hits` (plain counters -- no `gpu_` prefix, no `_total` suffix, and no separate
"hit rate" gauge at all, unlike this module's first guess), `vllm:num_preemptions` (no `_total`
suffix -- also unlike the first guess), `vllm:kv_cache_usage_perc` (renamed from the old
`vllm:gpu_cache_usage_perc` this dev machine's vLLM 0.6.3.post1 uses), `vllm:iteration_tokens_total`.
Still not the same as verified against a real response -- vLLM's `main` branch today isn't
necessarily the exact version `nscc_model_server`'s `vllm>=0.9.0` floor resolves to (same caveat as
`vllm_per_request_metrics.py`'s module docstring, which has the fuller story) -- but this is a real
step up from the original blind guesses (kept as lower-priority fallbacks, in case the actual
version differs from what's on GitHub's `main` today). Whichever candidate actually has a value
wins; if none match, the field is `null` and its paired `*_source` field
(`prefix_cache_hit_rate_source`, `iteration_tokens_source`) records that nothing was found, rather
than silently reporting 0.0 as if it were a real reading of "no cache hits"/"no tokens".

Usage:
    uv run python scripts/service_metrics_sampler.py --output runs/some_run/service_metrics.jsonl
    # stop with Ctrl-C / SIGTERM; each line is flushed immediately so a kill loses at most one
    # in-flight poll, not the whole file.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType

import httpx

from inspect_trace.vllm_metrics import metrics_url, parse_metrics_text

_RUNNING_CANDIDATES = ("vllm:num_requests_running",)
_WAITING_CANDIDATES = ("vllm:num_requests_waiting",)
_KV_CACHE_CANDIDATES = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
_PREEMPTIONS_CANDIDATES = ("vllm:num_preemptions", "vllm:num_preemptions_total")
_PREFIX_CACHE_QUERIES_COUNTER_CANDIDATES = (
    "vllm:prefix_cache_queries",
    "vllm:gpu_prefix_cache_queries_total",
    "vllm:prefix_cache_queries_total",
)
_PREFIX_CACHE_HITS_COUNTER_CANDIDATES = (
    "vllm:prefix_cache_hits",
    "vllm:gpu_prefix_cache_hits_total",
    "vllm:prefix_cache_hits_total",
)
_PREFIX_CACHE_HIT_RATE_GAUGE_CANDIDATES = (
    "vllm:gpu_prefix_cache_hit_rate",
    "vllm:prefix_cache_hit_rate",
)
"""No such gauge was found in vLLM's real /metrics source (only the queries/hits counters above
exist there) -- kept only as a last-resort fallback in case some version does expose a ready-made
rate."""
_ITERATION_TOKENS_COUNTER_CANDIDATES = (
    "vllm:iteration_tokens_total",
    "vllm:generation_tokens_total",
)
_ITERATION_TOKENS_HISTOGRAM_SUM_CANDIDATES = (
    "vllm:num_generation_tokens_iteration_sum",
    "vllm:iteration_tokens_seconds_sum",
)
_ITERATION_TOKENS_HISTOGRAM_COUNT_CANDIDATES = (
    "vllm:num_generation_tokens_iteration_count",
    "vllm:iteration_tokens_seconds_count",
)


def _first_present(
    values: dict[str, float], candidates: tuple[str, ...]
) -> float | None:
    for name in candidates:
        if name in values:
            return values[name]
    return None


def _iteration_tokens(values: dict[str, float]) -> tuple[float | None, str | None]:
    """需求 B4 服务层 "batch/iteration token": how many tokens a scheduler step is processing.

    A batching-efficiency signal. `vllm:iteration_tokens_total` (`_ITERATION_TOKENS_COUNTER_CANDIDATES`'
    first entry) is confirmed real (vLLM source, see module docstring) -- tried first now. The
    histogram-mean path is unconfirmed (no such histogram was found in the real source) and kept
    only as a fallback in case a different version exposes one; if it ever wins, it's the more
    useful "average tokens per iteration so far" rather than the counter's blunter running total.
    """
    total = _first_present(values, _ITERATION_TOKENS_COUNTER_CANDIDATES)
    if total is not None:
        return total, "counter_cumulative_total"
    hist_sum = _first_present(values, _ITERATION_TOKENS_HISTOGRAM_SUM_CANDIDATES)
    hist_count = _first_present(values, _ITERATION_TOKENS_HISTOGRAM_COUNT_CANDIDATES)
    if hist_sum is not None and hist_count:
        return hist_sum / hist_count, "histogram_mean"
    return None, None


def _prefix_cache_hit_rate(values: dict[str, float]) -> tuple[float | None, str | None]:
    queries = _first_present(values, _PREFIX_CACHE_QUERIES_COUNTER_CANDIDATES)
    hits = _first_present(values, _PREFIX_CACHE_HITS_COUNTER_CANDIDATES)
    if queries and hits is not None:
        # Cumulative-counter ratio at this instant, not a delta since the last poll -- a coarse
        # "hit rate so far" reading, good enough for a service-level time series sample, not a
        # per-window rate. Fine for B4's purpose (is prefix caching doing anything at all).
        return hits / queries, "counter_ratio"
    gauge = _first_present(values, _PREFIX_CACHE_HIT_RATE_GAUGE_CANDIDATES)
    if gauge is not None:
        return gauge, "gauge"
    return None, None


def _gpu_sample() -> dict[str, float | None]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return {
            "gpu_utilization_percent": None,
            "gpu_memory_used_mib": None,
            "gpu_power_watts": None,
        }
    line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        return {
            "gpu_utilization_percent": None,
            "gpu_memory_used_mib": None,
            "gpu_power_watts": None,
        }
    try:
        return {
            "gpu_utilization_percent": float(parts[0]),
            "gpu_memory_used_mib": float(parts[1]),
            "gpu_power_watts": float(parts[2]),
        }
    except ValueError:
        return {
            "gpu_utilization_percent": None,
            "gpu_memory_used_mib": None,
            "gpu_power_watts": None,
        }


def sample_once(client: httpx.Client) -> dict:
    poll_start = time.monotonic()
    vllm_reachable = True
    values: dict[str, float] = {}
    try:
        response = client.get(metrics_url(), timeout=2.0)
        response.raise_for_status()
        values = parse_metrics_text(response.text)
    except httpx.HTTPError:
        vllm_reachable = False

    prefix_hit_rate, prefix_hit_rate_source = _prefix_cache_hit_rate(values)
    iteration_tokens, iteration_tokens_source = _iteration_tokens(values)
    gpu = _gpu_sample()
    poll_duration_seconds = time.monotonic() - poll_start

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "poll_duration_seconds": poll_duration_seconds,
        "vllm_metrics_reachable": vllm_reachable,
        "num_requests_running": _first_present(values, _RUNNING_CANDIDATES),
        "num_requests_waiting": _first_present(values, _WAITING_CANDIDATES),
        "kv_cache_usage_perc": _first_present(values, _KV_CACHE_CANDIDATES),
        "preemptions_total": _first_present(values, _PREEMPTIONS_CANDIDATES),
        "prefix_cache_hit_rate": prefix_hit_rate,
        "prefix_cache_hit_rate_source": prefix_hit_rate_source,
        "iteration_tokens": iteration_tokens,
        "iteration_tokens_source": iteration_tokens_source,
        **gpu,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stop = False

    def _handle_stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    n_samples = 0
    with httpx.Client() as client, args.output.open("a") as f:
        print(
            f"service_metrics_sampler: writing to {args.output}, interval={args.interval_seconds}s",
            file=sys.stderr,
        )
        while not stop:
            record = sample_once(client)
            f.write(json.dumps(record) + "\n")
            f.flush()
            n_samples += 1
            time.sleep(args.interval_seconds)
    print(
        f"service_metrics_sampler: stopped after {n_samples} samples", file=sys.stderr
    )


if __name__ == "__main__":
    main()
