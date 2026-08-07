"""需求 B4 episode 层: 单调用 speedup 到 episode speedup 的转化率.

A method that speeds up individual model calls doesn't automatically speed up an episode by the
same factor -- tool-call time, waiting, and non-model-bound work don't shrink just because
decoding got faster (`episode_layer.py`'s own critical-path accounting already shows most of an
episode's wall clock is model-exclusive today, but that's a property of the workloads measured so
far, not a law). This module answers "how much of the per-call win actually shows up at the
episode level" given two runs (baseline vs a treatment method) that cover the same workload.
"""

from __future__ import annotations

from dataclasses import dataclass

from .episode_layer import EpisodeLayerRunSummary


def per_call_speedup(
    baseline_tokens_per_second: list[float], treatment_tokens_per_second: list[float]
) -> float | None:
    """Ratio of the two runs' mean tokens/s.

    Mean-of-ratios would need call-level pairing this project doesn't have across runs (calls
    aren't 1:1 addressable the way episodes are); ratio-of-means is the honest choice here.
    """
    if not baseline_tokens_per_second or not treatment_tokens_per_second:
        return None
    baseline_mean = sum(baseline_tokens_per_second) / len(baseline_tokens_per_second)
    if baseline_mean <= 0:
        return None
    treatment_mean = sum(treatment_tokens_per_second) / len(treatment_tokens_per_second)
    return treatment_mean / baseline_mean


def episode_speedup(
    baseline: EpisodeLayerRunSummary, treatment: EpisodeLayerRunSummary
) -> float | None:
    if (
        baseline.mean_end_to_end_latency_seconds is None
        or treatment.mean_end_to_end_latency_seconds in (None, 0)
    ):
        return None
    return (
        baseline.mean_end_to_end_latency_seconds
        / treatment.mean_end_to_end_latency_seconds
    )


@dataclass
class SpeedupConversion:
    call_speedup: float | None
    episode_speedup: float | None
    conversion_rate: float | None
    """Fraction of the per-call speedup *gain* (speedup - 1, i.e. "how much faster", not the raw
    ratio) that survives to the episode level. 1.0 means the full per-call gain shows up at the
    episode level; 0.3 means only 30% of it does (the rest was eaten by tool time/waiting/other
    non-accelerated work); values can exceed 1.0 or go negative for a noisy/small sample --
    reported as-is, not clamped, so a surprising number is visible rather than hidden."""


def compute(
    baseline_episodes: EpisodeLayerRunSummary,
    treatment_episodes: EpisodeLayerRunSummary,
    baseline_tokens_per_second: list[float],
    treatment_tokens_per_second: list[float],
) -> SpeedupConversion:
    call = per_call_speedup(baseline_tokens_per_second, treatment_tokens_per_second)
    episode = episode_speedup(baseline_episodes, treatment_episodes)
    conversion = None
    if call is not None and episode is not None and call != 1.0:
        conversion = (episode - 1.0) / (call - 1.0)
    return SpeedupConversion(
        call_speedup=call, episode_speedup=episode, conversion_rate=conversion
    )
