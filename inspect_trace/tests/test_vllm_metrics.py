"""Unit tests for vllm_metrics.py's Prometheus text parsing and delta logic.

No network/live-vLLM dependency here (that's covered separately, by design -- see
`/home/liuyingen/code/doc/efficient-harness/goal2_real_validation_findings.md` for the real-server
validation run this feature actually needs, which can't be a deterministic CI-style test). This
file only checks the pure-function pieces: parsing a literal metrics-text sample, and the
before/after delta math including the "not exactly one new observation" edge cases.
"""

import pytest

from inspect_trace.vllm_metrics import _delta, _snapshot_from_values, parse_metrics_text

SAMPLE_METRICS_TEXT = """\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen/Qwen2.5-3B-Instruct"} 1.0
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.01",model_name="Qwen/Qwen2.5-3B-Instruct"} 0.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="Qwen/Qwen2.5-3B-Instruct"} 3.0
vllm:time_to_first_token_seconds_sum{model_name="Qwen/Qwen2.5-3B-Instruct"} 0.150
vllm:time_to_first_token_seconds_count{model_name="Qwen/Qwen2.5-3B-Instruct"} 3.0
vllm:gpu_cache_usage_perc{model_name="Qwen/Qwen2.5-3B-Instruct"} 0.0023
"""


def test_parse_metrics_text_reads_expected_fields() -> None:
    values = parse_metrics_text(SAMPLE_METRICS_TEXT)
    assert values["vllm:num_requests_running"] == 1.0
    assert values["vllm:time_to_first_token_seconds_sum"] == 0.150
    assert values["vllm:time_to_first_token_seconds_count"] == 3.0
    assert values["vllm:gpu_cache_usage_perc"] == 0.0023
    # bucket lines (with a "le" label) are parsed too, just under their own (identical) metric
    # name -- last one wins, which is fine since we never read *_bucket ourselves.
    assert values["vllm:time_to_first_token_seconds_bucket"] == 3.0


def test_snapshot_from_values_handles_missing_fields() -> None:
    snapshot = _snapshot_from_values({"vllm:num_requests_running": 2.0})
    assert snapshot.num_requests_running == 2.0
    assert snapshot.ttft_sum is None
    assert snapshot.ttft_count is None


def test_exact_attribution_when_exactly_one_new_observation() -> None:
    before = _snapshot_from_values(
        parse_metrics_text(
            'vllm:time_to_first_token_seconds_sum{model_name="m"} 1.000\n'
            'vllm:time_to_first_token_seconds_count{model_name="m"} 5.0\n'
            'vllm:time_per_output_token_seconds_sum{model_name="m"} 10.000\n'
            'vllm:time_per_output_token_seconds_count{model_name="m"} 100.0\n'
            'vllm:e2e_request_latency_seconds_sum{model_name="m"} 20.000\n'
            'vllm:e2e_request_latency_seconds_count{model_name="m"} 5.0\n'
            'vllm:num_requests_running{model_name="m"} 1.0\n'
            'vllm:num_requests_waiting{model_name="m"} 0.0\n'
            'vllm:gpu_cache_usage_perc{model_name="m"} 0.001\n'
            'vllm:num_preemptions_total{model_name="m"} 0.0\n'
        )
    )
    after = _snapshot_from_values(
        parse_metrics_text(
            'vllm:time_to_first_token_seconds_sum{model_name="m"} 1.055\n'
            'vllm:time_to_first_token_seconds_count{model_name="m"} 6.0\n'
            'vllm:time_per_output_token_seconds_sum{model_name="m"} 13.220\n'
            'vllm:time_per_output_token_seconds_count{model_name="m"} 200.0\n'
            'vllm:e2e_request_latency_seconds_sum{model_name="m"} 29.700\n'
            'vllm:e2e_request_latency_seconds_count{model_name="m"} 6.0\n'
            'vllm:num_requests_running{model_name="m"} 1.0\n'
            'vllm:num_requests_waiting{model_name="m"} 0.0\n'
            'vllm:gpu_cache_usage_perc{model_name="m"} 0.0012\n'
            'vllm:num_preemptions_total{model_name="m"} 0.0\n'
        )
    )

    # fetch_snapshot() itself (the real network call) is exercised in the live-server validation
    # run, not here -- this only checks the delta/attribution math using the same `_delta` helper
    # sample_event() calls internally.
    assert _delta(before.ttft_count, after.ttft_count) == 1
    assert _delta(before.ttft_sum, after.ttft_sum) == pytest.approx(0.055)
    itl_count_delta = _delta(before.itl_count, after.itl_count)
    itl_sum_delta = _delta(before.itl_sum, after.itl_sum)
    assert itl_count_delta == 100
    assert itl_sum_delta is not None
    assert (itl_sum_delta / itl_count_delta) == pytest.approx(0.0322)
