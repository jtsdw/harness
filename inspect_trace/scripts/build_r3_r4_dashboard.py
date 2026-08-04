#!/usr/bin/env python3
"""Regenerates goal1_r3_r4_dashboard.html from real inspect_trace/`.eval` data.

Reads two real runs -- multi_turn_base and live_parallel, by default the FULL BFCL
category (200 and 15 samples respectively; see goal1_r3_r4_real_benchmark_findings.md
for the smaller 5/6-sample runs this originally started from) -- extracts everything
the dashboard needs (real transcripts, real token_attribution/prefill_diff/
execution_topology/action_parsing records, real bfcl_scorer results, real per-event
timestamps for the timeline), and writes a single self-contained HTML file -- no
external assets, no network calls, no server needed to view it.

Usage (from the inspect_trace/ project root, inside efficient-harness/):
    uv run python scripts/build_r3_r4_dashboard.py

Optional overrides (env vars):
    MULTI_TURN_RUN_DIR   (default: ../runs/goal1_bfcl_multi_turn_base_full)
    LIVE_PARALLEL_RUN_DIR (default: ../runs/goal1_bfcl_live_parallel_full)
    OUTPUT_PATH           (default: ../docs/goal1_r3_r4_dashboard.html,
                            relative to the efficient-harness/ repo root)

Both run directories must already exist (produced by run_bfcl_benchmark.sh -- see
that script's usage comment). The full-dataset runs were produced with:
    CATEGORIES="live_parallel" LIMIT=15 MAX_CONNECTIONS=1 \
      OUTPUT_DIR=.../runs/goal1_bfcl_live_parallel_full ./run_bfcl_benchmark.sh
    CATEGORIES="multi_turn_base" LIMIT=200 MAX_CONNECTIONS=1 \
      OUTPUT_DIR=.../runs/goal1_bfcl_multi_turn_base_full ./run_bfcl_benchmark.sh
(the 200-sample run takes roughly an hour serialized on a single local GPU). This
script only reads existing `.eval` logs and `inspect_trace` JSONL output; it does not
run any eval itself.
"""

from __future__ import annotations

import base64
import glob
import json
import os
from pathlib import Path

from inspect_ai._util.content import ContentReasoning
from inspect_ai.log import read_eval_log

from inspect_trace.analysis import episode_layer, token_layer

# parents[2]: scripts/ -> inspect_trace/ -> efficient-harness/ (the repo root). runs/ and
# docs/ both live directly under this root as siblings of inspect_trace/.
REPO_ROOT = Path(__file__).resolve().parents[2]
GOAL2_VLLM_METRICS_RUN_DIR = Path(
    os.environ.get(
        "GOAL2_VLLM_METRICS_RUN_DIR",
        REPO_ROOT / "runs/goal2_vllm_metrics_validation",
    )
)
MULTI_TURN_RUN_DIR = Path(
    os.environ.get(
        "MULTI_TURN_RUN_DIR", REPO_ROOT / "runs/goal1_bfcl_multi_turn_base_full"
    )
)
LIVE_PARALLEL_RUN_DIR = Path(
    os.environ.get(
        "LIVE_PARALLEL_RUN_DIR", REPO_ROOT / "runs/goal1_bfcl_live_parallel_full"
    )
)
OUTPUT_PATH = Path(
    os.environ.get(
        "OUTPUT_PATH",
        REPO_ROOT / "docs/goal1_r3_r4_dashboard.html",
    )
)

# LM Sans (Latin Modern Sans) -- ships with any texlive/texmf install; used as the
# display typeface. Not embedded as a repo asset since it's already present system-wide.
LM_SANS_REGULAR = Path("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-regular.otf")
LM_SANS_BOLD = Path("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-bold.otf")


def truncate(s: str | None, n: int = 260) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


def load_kind(pattern: str, kind: str) -> list[dict]:
    out = []
    for f in glob.glob(pattern):
        for line in open(f):
            r = json.loads(line)
            if r["kind"] == kind:
                out.append(r)
    return out


def extract_multi_turn_base(run_dir: Path) -> list[dict]:
    trace_glob = str(run_dir / ".inspect_trace/*/*/sample-*.jsonl")
    topology = {
        r["sample_uuid"]: r for r in load_kind(trace_glob, "execution_topology")
    }
    action_by_sample: dict[str, list[dict]] = {}
    for r in load_kind(trace_glob, "action_parsing"):
        action_by_sample.setdefault(r["sample_uuid"], []).append(r)
    prefill_by_sample: dict[str, list[dict]] = {}
    for r in load_kind(trace_glob, "prefill_diff"):
        prefill_by_sample.setdefault(r["sample_uuid"], []).append(r)
    attr_by_sample: dict[str, list[dict]] = {}
    for r in load_kind(trace_glob, "token_attribution"):
        attr_by_sample.setdefault(r["sample_uuid"], []).append(r)

    log_file = glob.glob(str(run_dir / "logs/*.eval"))[0]
    log = read_eval_log(log_file, resolve_attachments=True)

    samples_out = []
    for s in log.samples:
        su = s.uuid
        prefill_steps = sorted(
            prefill_by_sample.get(su, []), key=lambda x: x["step_index"]
        )
        attr_steps = sorted(attr_by_sample.get(su, []), key=lambda x: x["step_index"])
        msg_details_by_step = {
            r["step_index"]: {m["index"]: m for m in r["messages"]}
            for r in prefill_steps
        }

        model_events = [e for e in s.events if e.event == "model"]
        transcript_steps = []
        for i, e in enumerate(model_events, start=1):
            msg_details = msg_details_by_step.get(i, {})
            messages_out = []
            for idx, m in enumerate(e.input):
                detail = msg_details.get(idx, {})
                messages_out.append(
                    {
                        "index": idx,
                        "role": m.role,
                        "status": detail.get("status", "?"),
                        "content_category": detail.get("content_category", "?"),
                        "function": detail.get("function"),
                        "tool_call_id": detail.get("tool_call_id"),
                        "text": truncate(m.text),
                    }
                )
            reasoning_text = ""
            for block in e.output.message.content_list:
                if isinstance(block, ContentReasoning):
                    reasoning_text = (
                        block.reasoning if not block.redacted else (block.summary or "")
                    )
            tool_calls_out = [
                {"function": tc.function, "arguments": tc.arguments}
                for tc in (e.output.message.tool_calls or [])
            ]
            transcript_steps.append(
                {
                    "step": i,
                    "messages": messages_out,
                    "reasoning": truncate(reasoning_text),
                    "tool_calls": tool_calls_out,
                    "final_text": truncate(e.output.message.text),
                }
            )

        # unified timeline: every model/tool event, real wall-clock timestamps,
        # relative to this sample's first event -- this is what the "完整执行时间线"
        # panel renders directly (no separate parallel/waiting breakdown).
        timeline_events = [e for e in s.events if e.event in ("model", "tool")]
        timeline = []
        if timeline_events:
            t0 = timeline_events[0].timestamp
            for e in timeline_events:
                start = (e.timestamp - t0).total_seconds()
                end = (e.completed - t0).total_seconds() if e.completed else start
                if e.event == "model":
                    timeline.append(
                        {
                            "type": "model",
                            "start": start,
                            "end": end,
                            "working_time": e.working_time,
                            "label": "generate",
                        }
                    )
                else:
                    timeline.append(
                        {
                            "type": "tool",
                            "start": start,
                            "end": end,
                            "working_time": e.working_time,
                            "label": e.function,
                            "error": e.error.type if e.error else None,
                        }
                    )

        topo = topology.get(su)
        actions = action_by_sample.get(su, [])
        score = s.scores.get("bfcl_scorer") if s.scores else None

        samples_out.append(
            {
                "id": s.id,
                "uuid": su,
                "uuid_short": su[:8],
                "input_text": truncate(
                    str(s.input) if not isinstance(s.input, str) else s.input, 200
                ),
                "score": score.value if score else None,
                "explanation": truncate(score.explanation, 300)
                if score and score.explanation
                else None,
                "total_model_calls": len(model_events),
                "total_messages_final": len(model_events[-1].input) + 1
                if model_events
                else 0,
                "topology": {
                    "total_stages": topo["total_stages"] if topo else 0,
                    "total_tool_calls": topo["total_tool_calls"] if topo else 0,
                    "linear": topo["linear"] if topo else True,
                    "stages": [
                        {
                            "idx": st["stage_index"],
                            "functions": st["tool_functions"],
                            "count": st["tool_count"],
                            "parallel": st["observed_parallel"],
                            "model_wait": st["model_waiting_for_tool_seconds"],
                            "tool_wait": st["tool_waiting_for_model_seconds"],
                        }
                        for st in (topo["stages"] if topo else [])
                    ],
                },
                "errors": [
                    {
                        "function": a["function"],
                        "tool_call_id": a["tool_call_id"],
                        "error_present": a["error_present"],
                        "error_type": a["error_type"],
                        "message": a["error_message"],
                    }
                    for a in sorted(actions, key=lambda x: x["recorded_at"])
                ],
                "prefill": [
                    {
                        "step": r["step_index"],
                        "new_messages": r["new_messages"],
                        "reused_messages": r["reused_messages"],
                        "tools_new": r["tools_new"],
                        "tools_reused": r["tools_reused"],
                        "tools_total": r["tools_total"],
                        "system_template_messages": r["system_template_messages"],
                    }
                    for r in prefill_steps
                ],
                "attribution": [
                    {
                        "step": r["step_index"],
                        "system_template": r["system_template_tokens_estimate"],
                        "tool_schema": r["tool_schema_tokens_estimate"],
                        "conversation": r["conversation_tokens_estimate"],
                        "reasoning": r["reasoning_tokens_estimate"],
                        "tool_calling": r["tool_calling_tokens_estimate"],
                        "final_response": r["final_response_tokens_estimate"],
                        "billed_input": r["billed_input_tokens"],
                        "billed_output": r["billed_output_tokens"],
                    }
                    for r in attr_steps
                ],
                "transcript": transcript_steps,
                "timeline": timeline,
            }
        )
    return samples_out


def extract_live_parallel(run_dir: Path) -> list[dict]:
    log_files = sorted(glob.glob(str(run_dir / "logs/*.eval")), key=os.path.getmtime)
    log = read_eval_log(log_files[-1], resolve_attachments=True)

    samples_out = []
    for s in log.samples:
        model_events = [e for e in s.events if e.event == "model"]
        e = model_events[0] if model_events else None
        score = s.scores.get("bfcl_scorer") if s.scores else None
        tool_calls_out = []
        if e:
            tool_calls_out = [
                {"function": tc.function, "arguments": tc.arguments}
                for tc in (e.output.message.tool_calls or [])
            ]
        samples_out.append(
            {
                "id": s.id,
                "uuid": s.uuid,
                "uuid_short": s.uuid[:8],
                "input_text": truncate(
                    str(s.input) if not isinstance(s.input, str) else s.input, 200
                ),
                "score": score.value if score else None,
                "n_tools_declared": len(e.tools) if e else 0,
                "tool_calls": tool_calls_out,
                "final_text": truncate(e.output.message.text) if e else "",
                "billed_input": e.output.usage.input_tokens
                if e and e.output.usage
                else None,
                "billed_output": e.output.usage.output_tokens
                if e and e.output.usage
                else None,
            }
        )
    return samples_out


def extract_goal2_layers() -> dict:
    """目标二三层数据：token/episode/model invocation 层。

    token/episode 层复用已有的 multi_turn_base/live_parallel run（不需要新数据，这两层从
    goal1 的 run 里就能算），model invocation 层单独读 vllm_metrics 验证 run（唯一带
    `vllm_metrics` 记录的 run，见 goal2_real_validation_findings.md 的"复现命令"一节）。
    """
    mt_trace_dir = MULTI_TURN_RUN_DIR / ".inspect_trace"
    lp_trace_dir = LIVE_PARALLEL_RUN_DIR / ".inspect_trace"

    def token_run_summary(trace_dir: Path) -> dict:
        s = token_layer.summarize_run(trace_dir)
        return {
            "n_episodes": s.n_episodes,
            "total_billed_input_tokens": s.total_billed_input_tokens,
            "total_billed_output_tokens": s.total_billed_output_tokens,
            "total_reused_message_tokens_estimate": s.total_reused_message_tokens_estimate,
            "total_reused_tool_schema_tokens_estimate": s.total_reused_tool_schema_tokens_estimate,
            "total_retry_wasted_output_tokens_estimate": s.total_retry_wasted_output_tokens_estimate,
        }

    def episode_run_summary(trace_dir: Path) -> dict:
        s = episode_layer.summarize_run(trace_dir)
        return {
            "n_episodes": s.n_episodes,
            "success_rate": s.success_rate,
            "total_cost_usd": s.total_cost_usd,
            "cost_per_successful_episode_usd": s.cost_per_successful_episode_usd,
            "mean_end_to_end_latency_seconds": s.mean_end_to_end_latency_seconds,
            "mean_n_llm_calls": s.mean_n_llm_calls,
            "mean_n_tool_calls": s.mean_n_tool_calls,
            "total_retries": s.total_retries,
            "episodes_with_observed_parallel": s.episodes_with_observed_parallel,
        }

    vllm_glob = str(GOAL2_VLLM_METRICS_RUN_DIR / ".inspect_trace/*/*/sample-*.jsonl")
    vllm_records = load_kind(vllm_glob, "vllm_metrics")
    attribution_by_event = {
        r["model_event_uuid"]: r for r in load_kind(vllm_glob, "token_attribution")
    }
    model_invocation_calls = []
    for r in sorted(vllm_records, key=lambda x: x["recorded_at"]):
        attr = attribution_by_event.get(r["model_event_uuid"])
        model_invocation_calls.append(
            {
                "attribution_confidence": r["attribution_confidence"],
                "ttft_seconds": r["ttft_seconds"],
                "itl_seconds_avg": r["itl_seconds_avg"],
                "e2e_latency_seconds": r["e2e_latency_seconds"],
                "queue_depth_running_at_start": r["queue_depth_running_at_start"],
                "queue_depth_waiting_at_start": r["queue_depth_waiting_at_start"],
                "gpu_cache_usage_perc_at_end": r["gpu_cache_usage_perc_at_end"],
                "preemptions_delta": r["preemptions_delta"],
                "billed_input_tokens": attr["billed_input_tokens"] if attr else None,
                "billed_output_tokens": attr["billed_output_tokens"] if attr else None,
            }
        )

    exact_calls = [c for c in model_invocation_calls if c["attribution_confidence"] == "exact"]
    ttfts = [c["ttft_seconds"] for c in exact_calls if c["ttft_seconds"] is not None]
    itls = [c["itl_seconds_avg"] for c in exact_calls if c["itl_seconds_avg"] is not None]

    return {
        "token_layer": {
            "multi_turn_base": token_run_summary(mt_trace_dir),
            "live_parallel": token_run_summary(lp_trace_dir),
        },
        "episode_layer": {
            "multi_turn_base": episode_run_summary(mt_trace_dir),
            "live_parallel": episode_run_summary(lp_trace_dir),
        },
        "model_invocation": {
            "run_dir": str(GOAL2_VLLM_METRICS_RUN_DIR.name),
            "n_calls": len(model_invocation_calls),
            "n_exact": len(exact_calls),
            "mean_ttft_seconds": (sum(ttfts) / len(ttfts)) if ttfts else None,
            "mean_itl_seconds": (sum(itls) / len(itls)) if itls else None,
            "calls": model_invocation_calls,
        },
    }


def extract_all() -> dict:
    mt_samples = extract_multi_turn_base(MULTI_TURN_RUN_DIR)
    lp_samples = extract_live_parallel(LIVE_PARALLEL_RUN_DIR)

    mt_episode_by_uuid = {
        e.sample_uuid: e
        for e in episode_layer.summarize_episode_layer(MULTI_TURN_RUN_DIR / ".inspect_trace")
    }
    for samp in mt_samples:
        ep = mt_episode_by_uuid.get(samp["uuid"])
        samp["episode"] = (
            {
                "end_to_end_latency_seconds": ep.end_to_end_latency_seconds,
                "critical_path_latency_seconds": ep.critical_path_latency_seconds,
                "total_busy_seconds": ep.total_busy_seconds,
                "concurrency_savings_seconds": ep.concurrency_savings_seconds,
                "cost_usd": ep.cost_usd,
            }
            if ep
            else None
        )

    all_glob_a = str(REPO_ROOT / "runs/*/.inspect_trace/*/*/sample-*.jsonl")
    all_glob_b = str(REPO_ROOT / "runs/*/*/.inspect_trace/*/*/sample-*.jsonl")
    all_attempts = load_kind(all_glob_a, "attempt_group") + load_kind(
        all_glob_b, "attempt_group"
    )

    all_model_wait = [
        s["model_wait"]
        for samp in mt_samples
        for s in samp["topology"]["stages"]
        if s["model_wait"] is not None
    ]
    all_tool_wait = [
        w
        for samp in mt_samples
        for s in samp["topology"]["stages"]
        for w in s["tool_wait"]
    ]

    total_tool_calls = sum(s["topology"]["total_tool_calls"] for s in mt_samples)
    multi_call_stages = sum(
        1 for s in mt_samples for st in s["topology"]["stages"] if st["count"] > 1
    )
    observed_parallel_stages = sum(
        1 for s in mt_samples for st in s["topology"]["stages"] if st["parallel"]
    )
    all_errors = [e for s in mt_samples for e in s["errors"]]
    error_count = sum(1 for e in all_errors if e["error_present"])

    return {
        "multi_turn_base": {
            "n_samples": len(mt_samples),
            "n_correct": sum(1 for s in mt_samples if s["score"] == 1),
            "samples": mt_samples,
        },
        "live_parallel": {
            "n_samples": len(lp_samples),
            "n_correct": sum(1 for s in lp_samples if s["score"] == 1),
            "samples": lp_samples,
        },
        "aggregate": {
            "total_tool_calls": total_tool_calls,
            "multi_call_stages": multi_call_stages,
            "observed_parallel_stages": observed_parallel_stages,
            "total_action_calls": len(all_errors),
            "total_action_errors": error_count,
        },
        "waiting": {
            "model_waiting_for_tool": {
                "n": len(all_model_wait),
                "mean_ms": (sum(all_model_wait) / len(all_model_wait) * 1000)
                if all_model_wait
                else None,
                "max_ms": max(all_model_wait) * 1000 if all_model_wait else None,
            },
            "tool_waiting_for_model": {
                "n": len(all_tool_wait),
                "mean_ms": (sum(all_tool_wait) / len(all_tool_wait) * 1000)
                if all_tool_wait
                else None,
                "max_ms": max(all_tool_wait) * 1000 if all_tool_wait else None,
            },
        },
        "retries": {
            "total_requests_all_runs": len(all_attempts),
            "requests_with_retry": len(
                [a for a in all_attempts if a["total_attempts"] > 1]
            ),
        },
        "goal2": extract_goal2_layers(),
    }


def build_html(data: dict) -> str:
    lm_regular = base64.b64encode(LM_SANS_REGULAR.read_bytes()).decode()
    lm_bold = base64.b64encode(LM_SANS_BOLD.read_bytes()).decode()
    data_json = json.dumps(data, ensure_ascii=False)

    template_path = Path(__file__).with_name("_dashboard_template.html")
    html = template_path.read_text()
    return (
        html.replace("__LM_REGULAR__", lm_regular)
        .replace("__LM_BOLD__", lm_bold)
        .replace("__DATA_JSON__", data_json)
    )


def main() -> None:
    for label, path in [
        ("MULTI_TURN_RUN_DIR", MULTI_TURN_RUN_DIR),
        ("LIVE_PARALLEL_RUN_DIR", LIVE_PARALLEL_RUN_DIR),
    ]:
        if not path.exists():
            raise SystemExit(
                f"{label}={path} does not exist. Run run_bfcl_benchmark.sh first -- see "
                "this script's module docstring or goal1_r3_r4_real_benchmark_findings.md."
            )

    data = extract_all()
    html = build_html(data)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html)
    print(f"wrote {len(html)} bytes to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
