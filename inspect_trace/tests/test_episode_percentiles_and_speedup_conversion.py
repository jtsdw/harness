"""Unit tests for episode_layer.py's percentile math and speedup_conversion.py.

Pure-function tests, no eval/hooks machinery needed -- the integration path (percentiles computed
from a real hooks-produced trace dir) is covered by test_analysis_layers.py's
test_episode_layer_summarizes_real_hooks_output.
"""

import pytest

from inspect_trace.analysis import speedup_conversion
from inspect_trace.analysis.episode_layer import (
    _P99_MIN_SAMPLES,
    EpisodeLayerRunSummary,
    _percentile,
)


def test_percentile_matches_known_values() -> None:
    values = sorted([10.0, 20.0, 30.0, 40.0, 50.0])
    assert _percentile(values, 50) == 30.0
    assert _percentile(values, 0) == 10.0
    assert _percentile(values, 100) == 50.0
    # rank = 62.5/100 * (5-1) = 2.5 -> halfway between index 2 (30.0) and index 3 (40.0)
    assert _percentile(values, 62.5) == pytest.approx(35.0)


def test_percentile_single_value() -> None:
    assert _percentile([5.0], 50) == 5.0
    assert _percentile([5.0], 99) == 5.0


def _run_summary(mean_latency: float | None) -> EpisodeLayerRunSummary:
    return EpisodeLayerRunSummary(
        n_episodes=1,
        success_rate=1.0,
        total_cost_usd=None,
        cost_per_successful_episode_usd=None,
        mean_end_to_end_latency_seconds=mean_latency,
        p50_end_to_end_latency_seconds=mean_latency,
        p95_end_to_end_latency_seconds=mean_latency,
        p99_end_to_end_latency_seconds=None,
        mean_n_llm_calls=1.0,
        mean_n_tool_calls=0.0,
        total_retries=0,
        episodes_with_observed_parallel=0,
        per_episode=[],
    )


def test_p99_min_samples_threshold_is_documented_and_positive() -> None:
    # a change to this constant should be a deliberate decision, not an accidental typo -- pin it.
    assert _P99_MIN_SAMPLES == 100


def test_speedup_conversion_full_gain_survives_to_episode_level() -> None:
    baseline = _run_summary(10.0)
    treatment = _run_summary(5.0)  # 2x faster episodes
    result = speedup_conversion.compute(
        baseline,
        treatment,
        baseline_tokens_per_second=[50.0],
        treatment_tokens_per_second=[100.0],
    )
    assert result.call_speedup == pytest.approx(2.0)
    assert result.episode_speedup == pytest.approx(2.0)
    assert result.conversion_rate == pytest.approx(1.0)


def test_speedup_conversion_partial_gain() -> None:
    baseline = _run_summary(10.0)
    treatment = _run_summary(8.0)  # only 1.25x faster episodes despite 2x faster calls
    result = speedup_conversion.compute(
        baseline,
        treatment,
        baseline_tokens_per_second=[50.0],
        treatment_tokens_per_second=[100.0],
    )
    assert result.call_speedup == pytest.approx(2.0)
    assert result.episode_speedup == pytest.approx(1.25)
    assert result.conversion_rate == pytest.approx(0.25)


def test_speedup_conversion_missing_data_returns_none() -> None:
    baseline = _run_summary(None)
    treatment = _run_summary(5.0)
    result = speedup_conversion.compute(baseline, treatment, [], [100.0])
    assert result.call_speedup is None
    assert result.episode_speedup is None
    assert result.conversion_rate is None
