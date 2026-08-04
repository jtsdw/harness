"""Model pricing table for `cost per successful episode` (episode_layer.py).

Deliberately does NOT ship invented/possibly-stale published $/token rates for hosted models --
those change and guessing them wrong would silently corrupt a cost number. Fill in
`price_per_1m_input_tokens`/`price_per_1m_output_tokens` yourself from the provider's current
pricing page before relying on cost-per-episode for a hosted model. Local vLLM is the one entry
that's genuinely always correct: self-hosted inference has no per-token metering, so it's 0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    price_per_1m_input_tokens: float
    price_per_1m_output_tokens: float


# Keyed by the *exact* string recorded in ModelEvent.model / TokenAttributionRecord.model_name --
# for a provider/model string like "openai-api/vllm/Qwen/Qwen2.5-3B-Instruct" that's the full
# string, not just the trailing model name (verified against real recorded data; don't assume a
# bare model name matches).
PRICING_TABLE: dict[str, ModelPricing] = {
    # Local vLLM deployment: self-hosted, no per-token billing.
    "openai-api/vllm/Qwen/Qwen2.5-3B-Instruct": ModelPricing(0.0, 0.0),
}


def cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float | None:
    pricing = PRICING_TABLE.get(model_name)
    if pricing is None:
        return None
    return (
        input_tokens / 1_000_000 * pricing.price_per_1m_input_tokens
        + output_tokens / 1_000_000 * pricing.price_per_1m_output_tokens
    )
