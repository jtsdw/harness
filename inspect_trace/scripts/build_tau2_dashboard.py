#!/usr/bin/env python3
"""Builds a real-data dashboard for the tau2-bench integration.

See docs/tau2_bench_integration_findings.md for the full write-up.

Reads three real runs:
  - the native `tau2 run` CLI baseline (runs/tau2_native_baseline/)
  - our inspect_ai adapter with the agent forced onto emulate_tools=true, a workaround for the
    "strict" tool field vLLM rejects (runs/tau2_adapter_full/)
  - our adapter again after fixing that properly (a custom ModelAPI that omits the field, see
    tau2_adapter/_registry.py), so the agent uses real native tool-calling like the other two
    paths (runs/tau2_adapter_native/)

...and produces a single self-contained HTML file: no external assets, no network calls, no
server needed to view it.

Usage (from the inspect_trace/ project root, inside efficient-harness/):
    uv run python scripts/build_tau2_dashboard.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

from inspect_trace.analysis import episode_layer, token_layer
from inspect_trace.analysis._loader import load_records_by_sample, records_of_kind

REPO_ROOT = Path(__file__).resolve().parents[2]
NATIVE_RESULTS = (
    REPO_ROOT / "runs/tau2_native_baseline/results_mock_baseline/results.json"
)
ADAPTER_EMULATE_RUN_DIR = REPO_ROOT / "runs/tau2_adapter_full"
ADAPTER_NATIVE_RUN_DIR = REPO_ROOT / "runs/tau2_adapter_native"
OUTPUT_PATH = REPO_ROOT / "docs/tau2_dashboard.html"

LM_SANS_REGULAR = Path("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-regular.otf")
LM_SANS_BOLD = Path("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-bold.otf")


def truncate(s: str | None, n: int = 500) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


def load_native_results() -> dict[str, dict]:
    d = json.loads(NATIVE_RESULTS.read_text())
    sims = d["simulations"] if isinstance(d, dict) and "simulations" in d else d
    return {s["task_id"]: s for s in sims}


def native_messages(sim: dict) -> list[dict]:
    out = []
    for m in sim.get("messages") or []:
        out.append(
            {
                "role": m.get("role"),
                "content": truncate(m.get("content")),
                "tool_calls": [
                    {"name": tc.get("name"), "arguments": tc.get("arguments")}
                    for tc in (m.get("tool_calls") or [])
                ],
                "turn_idx": m.get("turn_idx"),
            }
        )
    return out


def adapter_eval_log(run_dir: Path):
    logs = list((run_dir / "logs").glob("*.eval"))
    assert len(logs) == 1, f"expected exactly one .eval log in {run_dir}, found {logs}"
    return read_eval_log(str(logs[0]), resolve_attachments=True)


def adapter_scores_from_log(log) -> dict[str, dict]:
    scores = {}
    for s in log.samples:
        score = s.scores.get("tau2_reward_scorer") if s.scores else None
        md = score.metadata if score and score.metadata else {}
        scores[str(s.id)] = {
            "reward": md.get("tau2_reward", 0.0),
            "termination": md.get("tau2_termination_reason", "?"),
        }
    return scores


def classify(a_reward: float, a_term: str, b_reward: float, b_term: str) -> str:
    if a_reward == b_reward and a_term == b_term:
        return "exact"
    elif a_reward == b_reward:
        return "reward_same_term_diff"
    else:
        return "reward_diff"


def build_comparison(
    native: dict[str, dict],
    emulate_scores: dict[str, dict],
    native_tool_scores: dict[str, dict],
) -> list[dict]:
    rows = []
    for task_id, sim in native.items():
        n_reward = sim["reward_info"]["reward"] if sim.get("reward_info") else 0.0
        n_term = sim["termination_reason"]
        e = emulate_scores.get(task_id, {})
        nt = native_tool_scores.get(task_id, {})
        rows.append(
            {
                "task_id": task_id,
                "native_reward": n_reward,
                "native_termination": n_term,
                "emulate_reward": e.get("reward", 0.0),
                "emulate_termination": e.get("termination", "?"),
                "emulate_match": classify(
                    n_reward, n_term, e.get("reward", 0.0), e.get("termination", "?")
                ),
                "native_tool_reward": nt.get("reward", 0.0),
                "native_tool_termination": nt.get("termination", "?"),
                "native_tool_match": classify(
                    n_reward, n_term, nt.get("reward", 0.0), nt.get("termination", "?")
                ),
            }
        )
    rows.sort(key=lambda r: r["task_id"])
    return rows


def summarize_match_counts(comparison: list[dict], key: str) -> dict:
    return {
        "exact": sum(1 for r in comparison if r[key] == "exact"),
        "reward_same_term_diff": sum(
            1 for r in comparison if r[key] == "reward_same_term_diff"
        ),
        "reward_diff": sum(1 for r in comparison if r[key] == "reward_diff"),
    }


def extract_all() -> dict:
    native = load_native_results()
    emulate_log = adapter_eval_log(ADAPTER_EMULATE_RUN_DIR)
    native_tool_log = adapter_eval_log(ADAPTER_NATIVE_RUN_DIR)
    emulate_scores = adapter_scores_from_log(emulate_log)
    native_tool_scores = adapter_scores_from_log(native_tool_log)

    comparison = build_comparison(native, emulate_scores, native_tool_scores)
    n = len(comparison)
    native_acc = sum(1 for r in comparison if r["native_reward"] >= 1.0) / n
    emulate_acc = sum(1 for r in comparison if r["emulate_reward"] >= 1.0) / n
    native_tool_acc = sum(1 for r in comparison if r["native_tool_reward"] >= 1.0) / n

    # Token/episode layer and the TTFT scatter use the native-tool-calling adapter run --
    # it's the more apples-to-apples comparison against the native CLI baseline (both now use
    # the same tool-calling mechanism), so it's the more meaningful one to profile in depth.
    trace_dir = ADAPTER_NATIVE_RUN_DIR / ".inspect_trace"
    tl = token_layer.summarize_run(trace_dir)
    el = episode_layer.summarize_run(trace_dir)

    by_sample = load_records_by_sample(trace_dir)
    all_records = [r for recs in by_sample.values() for r in recs]
    vllm_calls = records_of_kind(all_records, "vllm_metrics")
    attribution_by_event = {
        r["model_event_uuid"]: r
        for r in records_of_kind(all_records, "token_attribution")
    }
    scatter = []
    for c in vllm_calls:
        attr = attribution_by_event.get(c["model_event_uuid"])
        if attr is None or c["ttft_seconds"] is None:
            continue
        scatter.append(
            {
                "task_id": c["sample_id"],
                "billed_input_tokens": attr["billed_input_tokens"],
                "ttft_seconds": c["ttft_seconds"],
                "itl_seconds_avg": c["itl_seconds_avg"],
                "confidence": c["attribution_confidence"],
            }
        )

    # Full conversation transcripts (native CLI path -- it's the one with a complete dual-control
    # message list) for a representative spread. Picked by hand from the comparison table, not
    # auto-selected, so the examples are actually illustrative rather than whatever sorts first.
    transcript_ids = [
        "update_task_with_message_history",  # exact match on both adapter variants
        "update_task_1",  # emulate_tools flipped this one; native tool-calling recovered it
        "create_task_1",  # fails on max_steps on all three paths
    ]
    transcripts = {}
    for tid in transcript_ids:
        sim = native.get(tid)
        if sim is None:
            continue
        transcripts[tid] = {
            "native_messages": native_messages(sim),
            "native_duration": sim.get("duration"),
        }

    episode_by_id = {ep.sample_id: ep for ep in el.per_episode}
    token_by_id = {t.sample_uuid: t for t in tl.per_episode}
    # token_layer keys by sample_uuid not sample_id -- join through the eval log's samples.
    uuid_by_task_id = {str(s.id): s.uuid for s in native_tool_log.samples}
    for tid, t in transcripts.items():
        ep = episode_by_id.get(tid)
        tok = token_by_id.get(uuid_by_task_id.get(tid))
        t["adapter_episode"] = (
            {
                "end_to_end_latency_seconds": ep.end_to_end_latency_seconds,
                "n_llm_calls": ep.n_llm_calls,
                "success": ep.success,
            }
            if ep
            else None
        )
        t["adapter_tokens"] = (
            {
                "billed_input_tokens": tok.billed_input_tokens,
                "billed_output_tokens": tok.billed_output_tokens,
                "tool_schema_tokens_estimate": tok.tool_schema_tokens_estimate,
                "conversation_tokens_estimate": tok.conversation_tokens_estimate,
            }
            if tok
            else None
        )

    return {
        "meta": {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "domain": "mock",
            "n_tasks": n,
        },
        "summary": {
            "native_accuracy": native_acc,
            "emulate_accuracy": emulate_acc,
            "native_tool_accuracy": native_tool_acc,
            "emulate_match_counts": summarize_match_counts(comparison, "emulate_match"),
            "native_tool_match_counts": summarize_match_counts(
                comparison, "native_tool_match"
            ),
        },
        "comparison": comparison,
        "token_layer": {
            "n_episodes": tl.n_episodes,
            "total_billed_input_tokens": tl.total_billed_input_tokens,
            "total_billed_output_tokens": tl.total_billed_output_tokens,
            "total_reused_tool_schema_tokens_estimate": tl.total_reused_tool_schema_tokens_estimate,
            "total_reused_message_tokens_estimate": tl.total_reused_message_tokens_estimate,
        },
        "episode_layer": {
            "mean_end_to_end_latency_seconds": el.mean_end_to_end_latency_seconds,
            "mean_n_llm_calls": el.mean_n_llm_calls,
            "episodes_with_observed_parallel": el.episodes_with_observed_parallel,
        },
        "vllm_scatter": scatter,
        "transcripts": transcripts,
    }


def main() -> None:
    data = extract_all()
    lm_regular_b64 = base64.b64encode(LM_SANS_REGULAR.read_bytes()).decode()
    lm_bold_b64 = base64.b64encode(LM_SANS_BOLD.read_bytes()).decode()

    template_path = Path(__file__).parent / "_tau2_dashboard_template.html"
    html = template_path.read_text()
    html = html.replace("__LM_REGULAR__", lm_regular_b64)
    html = html.replace("__LM_BOLD__", lm_bold_b64)
    html = html.replace("__DATA_JSON__", json.dumps(data))

    OUTPUT_PATH.write_text(html)
    print(f"wrote {len(html)} bytes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
