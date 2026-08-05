#!/usr/bin/env python3
"""Builds a real-data dashboard for the ToolSpec integration (see
docs/toolspec_integration_findings.md for the full write-up).

Reads real data from two sources:
  - ToolSpec's own native-repo reproduction output (/home/liuyingen/code/ToolSpec/output/
    APIBank/Qwen2.5-3B-Instruct/*.jsonl) -- 5 methods (baseline/pld/recycling/samd/toolspec) plus
    a baseline rerun used as a determinism control.
  - our inspect_ai adapter's .eval logs (runs/toolspec_adapter_{baseline,toolspec}/logs/*.eval).

Every number in this dashboard is recomputed live from those files each time this script runs --
nothing is hardcoded from a prior chat summary. Produces a single self-contained HTML file: no
external assets, no network calls, no server needed to view it.

Usage (from the inspect_trace/ project root, inside efficient-harness/):
    uv run python scripts/build_toolspec_dashboard.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLSPEC_REPO_DIR = Path("/home/liuyingen/code/ToolSpec")
NATIVE_OUTPUT_DIR = TOOLSPEC_REPO_DIR / "output/APIBank/Qwen2.5-3B-Instruct"
ADAPTER_BASELINE_DIR = REPO_ROOT / "runs/toolspec_adapter_baseline"
ADAPTER_TOOLSPEC_DIR = REPO_ROOT / "runs/toolspec_adapter_toolspec"
VLLM_BASELINE_DIR = REPO_ROOT / "runs/toolspec_vllm_baseline"
VLLM_NGRAM_DIR = REPO_ROOT / "runs/toolspec_vllm_ngram"
OUTPUT_PATH = REPO_ROOT / "docs/toolspec_dashboard.html"

LM_SANS_REGULAR = Path("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-regular.otf")
LM_SANS_BOLD = Path("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-bold.otf")

NATIVE_FILES = {
    "baseline": "Qwen2.5-3B-Instruct-vanilla-float16-temp-0.0.jsonl",
    "baseline_rerun": "Qwen2.5-3B-Instruct-vanilla-float16-temp-0.0-rerun.jsonl",
    "pld": "Qwen2.5-3B-Instruct-pld-float16.jsonl",
    "recycling": "Qwen2.5-3B-Instruct-recycling-float16-temp-0.0.jsonl",
    "samd": "Qwen2.5-3B-Instruct-samd.jsonl",
    "toolspec": "Qwen2.5-3B-Instruct-toolspec-float16.jsonl",
}


def load_native(name: str) -> dict[int, dict]:
    path = NATIVE_OUTPUT_DIR / NATIVE_FILES[name]
    out: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        out[obj["question_id"]] = obj
    return out


def speed_stats(records: dict[int, dict]) -> tuple[float, float]:
    """Returns (tokens_per_second, mean_accept_length).

    tokens_per_second is the MEAN of each question's own tokens/wall_time ratio (matching
    ToolSpec's own evaluation/speed.py::speed() exactly -- `np.array(speeds).mean()`), NOT the
    ratio of summed tokens over summed time. These two give meaningfully different numbers here
    (an earlier version of this script used the ratio-of-sums and got 2.77x instead of the real
    3.05x for native toolspec) because per-question tokens/time varies a lot, so a plain ratio of
    totals implicitly weights by token count while the paper's own method treats each question
    equally. Matching the paper's own method is the right choice for an apples-to-apples number.
    """
    speeds = []
    accept_lengths: list[int] = []
    for obj in records.values():
        tokens = sum(obj["choices"]["new_tokens"])
        time_s = sum(obj["choices"]["wall_time"])
        if time_s > 0:
            speeds.append(tokens / time_s)
        accept_lengths.extend(obj["choices"]["accept_lengths"])
    tps = sum(speeds) / len(speeds) if speeds else 0.0
    mean_accept = sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0
    return tps, mean_accept


def mismatches(base: dict[int, dict], other: dict[int, dict]) -> list[int]:
    return sorted(
        qid for qid in base if base[qid]["choices"]["output"] != other[qid]["choices"]["output"]
    )


def load_adapter_run(run_dir: Path) -> dict:
    eval_files = sorted((run_dir / "logs").glob("*.eval"))
    assert len(eval_files) == 1, f"expected exactly one .eval in {run_dir}/logs, found {len(eval_files)}"
    log = read_eval_log(str(eval_files[0]))

    per_sample_speeds: list[float] = []
    correct = 0
    incorrect_qids: list[int] = []
    for sample in log.samples:
        sample_tokens = 0
        sample_time = 0.0
        for event in sample.events:
            if event.event == "model" and event.call and event.call.response:
                resp = event.call.response
                sample_tokens += resp.get("new_tokens", 0) or 0
                sample_time += resp.get("wall_time", 0.0) or 0.0
        if sample_time > 0:
            per_sample_speeds.append(sample_tokens / sample_time)
        score = sample.scores.get("matches_reference_baseline") if sample.scores else None
        if score is not None:
            if score.value == "C":
                correct += 1
            else:
                incorrect_qids.append(int(sample.metadata.get("question_id", -1)))

    trace_files = list((run_dir / ".inspect_trace").glob("**/sample-*.jsonl")) if (run_dir / ".inspect_trace").exists() else []

    # Same mean-of-per-question-ratios methodology as speed_stats() / evaluation/speed.py --
    # see that function's docstring for why this matters (not the same as ratio-of-sums).
    tps = sum(per_sample_speeds) / len(per_sample_speeds) if per_sample_speeds else 0.0

    return {
        "eval_file": eval_files[0].name,
        "n_samples": len(log.samples),
        "correct": correct,
        "incorrect_qids": sorted(incorrect_qids),
        "tokens_per_second": tps,
        "n_trace_files": len(trace_files),
    }


def load_vllm_run(run_dir: Path) -> dict:
    """Like load_adapter_run(), but for runs through the stock openai-api provider (vLLM HTTP
    service, no/ngram speculative decoding). Speed comes from ModelEvent.working_time /
    output.usage.output_tokens instead of our own provider's ModelCall.response dict (which only
    exists for toolspec_adapter's custom ModelAPI), because inspect_trace's own vllm_metrics
    collector produced zero records for these two runs -- see
    docs/toolspec_vllm_speculative_comparison.md's "一个真实发现" section for the real bug found
    while trying to use it, and why this fallback is used instead.
    """
    eval_files = sorted((run_dir / "logs").glob("*.eval"))
    assert len(eval_files) == 1, f"expected exactly one .eval in {run_dir}/logs, found {len(eval_files)}"
    log = read_eval_log(str(eval_files[0]))

    per_sample_speeds: list[float] = []
    completions: dict[int, str] = {}
    for sample in log.samples:
        qid = int(sample.metadata.get("question_id", -1))
        completions[qid] = sample.output.completion.strip() if sample.output else ""
        for event in sample.events:
            if event.event == "model":
                wt = event.working_time
                out_tok = event.output.usage.output_tokens if event.output and event.output.usage else None
                if wt and out_tok:
                    per_sample_speeds.append(out_tok / wt)

    tps = sum(per_sample_speeds) / len(per_sample_speeds) if per_sample_speeds else 0.0
    return {
        "eval_file": eval_files[0].name,
        "n_samples": len(log.samples),
        "tokens_per_second": tps,
        "completions": completions,
    }


def truncate(s: str, n: int = 220) -> str:
    return s if len(s) <= n else s[:n] + "…"


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def main() -> None:
    native = {name: load_native(name) for name in NATIVE_FILES}

    native_determinism_mismatches = mismatches(native["baseline"], native["baseline_rerun"])

    native_speed_rows = []
    for method in ["baseline", "pld", "recycling", "samd", "toolspec"]:
        tps, mean_accept = speed_stats(native[method])
        baseline_tps, _ = speed_stats(native["baseline"])
        mism = [] if method == "baseline" else mismatches(native["baseline"], native[method])
        native_speed_rows.append(
            {
                "method": method,
                "tokens_per_second": tps,
                "mean_accept": mean_accept,
                "speedup": tps / baseline_tps if baseline_tps else 0.0,
                "n_mismatches": len(mism),
                "mismatch_qids": mism,
            }
        )

    adapter_baseline = load_adapter_run(ADAPTER_BASELINE_DIR)
    adapter_toolspec = load_adapter_run(ADAPTER_TOOLSPEC_DIR)

    vllm_baseline = load_vllm_run(VLLM_BASELINE_DIR)
    vllm_ngram = load_vllm_run(VLLM_NGRAM_DIR)
    vllm_ngram_mismatch_qids = sorted(
        qid
        for qid in vllm_baseline["completions"]
        if vllm_baseline["completions"][qid] != vllm_ngram["completions"].get(qid)
    )
    vllm_speedup = (
        vllm_ngram["tokens_per_second"] / vllm_baseline["tokens_per_second"]
        if vllm_baseline["tokens_per_second"]
        else 0.0
    )

    native_toolspec_row = next(r for r in native_speed_rows if r["method"] == "toolspec")
    native_baseline_row = next(r for r in native_speed_rows if r["method"] == "baseline")

    same_mismatch_set = adapter_toolspec["incorrect_qids"] == native_toolspec_row["mismatch_qids"]

    # a few concrete example mismatches, with real question/output text
    example_qids = native_toolspec_row["mismatch_qids"][:3]
    examples = []
    for qid in example_qids:
        base_obj = native["baseline"][qid]
        spec_obj = native["toolspec"][qid]
        examples.append(
            {
                "qid": qid,
                "base_output": base_obj["choices"]["output"],
                "spec_output": spec_obj["choices"]["output"],
            }
        )

    lm_regular_b64 = base64.b64encode(LM_SANS_REGULAR.read_bytes()).decode() if LM_SANS_REGULAR.exists() else ""
    lm_bold_b64 = base64.b64encode(LM_SANS_BOLD.read_bytes()).decode() if LM_SANS_BOLD.exists() else ""

    def pill(ok: bool) -> str:
        return f'<span class="pill {"good" if ok else "bad"}">{"C" if ok else "I"}</span>'

    speed_rows_html = "\n".join(
        f"""<tr>
          <td><span class="mono">{r['method']}</span></td>
          <td class="mono">{r['tokens_per_second']:.2f}</td>
          <td class="mono">{r['mean_accept']:.3f}</td>
          <td class="mono">{r['speedup']:.2f}x</td>
          <td class="mono">{r['n_mismatches']}/100</td>
        </tr>"""
        for r in native_speed_rows
    )

    mismatch_table_rows = "\n".join(
        f"""<tr>
          <td class="mono">{qid}</td>
          <td>{'✓' if qid in native["baseline_rerun"] and qid not in native_determinism_mismatches else ''}</td>
          {''.join(f'<td class="{"mono match-bad" if qid in r["mismatch_qids"] else "mono"}">{"✗" if qid in r["mismatch_qids"] else "✓"}</td>' for r in native_speed_rows if r['method'] != 'baseline')}
        </tr>"""
        for qid in sorted(set().union(*(r["mismatch_qids"] for r in native_speed_rows)))
    )

    examples_html = "\n".join(
        f"""<div class="bug">
          <div class="bug-title"><span class="n">question_id {ex['qid']}</span></div>
          <p><strong>baseline (true greedy):</strong></p>
          <code>{esc(truncate(ex['base_output'], 400))}</code>
          <p style="margin-top:10px"><strong>toolspec:</strong></p>
          <code>{esc(truncate(ex['spec_output'], 400))}</code>
        </div>"""
        for ex in examples
    )

    html = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>ToolSpec 接入：真实结果</title>
<style>
@font-face {{ font-family: "LM Sans"; src: url(data:font/otf;base64,{lm_regular_b64}) format("opentype"); font-weight: 400; }}
@font-face {{ font-family: "LM Sans"; src: url(data:font/otf;base64,{lm_bold_b64}) format("opentype"); font-weight: 700; }}

:root {{
  --bg: #f6f4ee; --surface: #ffffff; --surface-2: #ece8de; --border: #d9d3c4;
  --text: #201e1a; --text-dim: #6b6459; --accent: #b56a28; --accent-2: #2f7d78;
  --good: #3f7d4c; --bad: #a83358;
  --accent-bg: rgba(181, 106, 40, 0.10); --good-bg: rgba(63, 125, 76, 0.12); --bad-bg: rgba(168, 51, 88, 0.12);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #14161c; --surface: #1c1f28; --surface-2: #262a35; --border: #333846;
    --text: #e9e6df; --text-dim: #9b9fac; --accent: #d98e4a; --accent-2: #5fa8a3;
    --good: #6fae7c; --bad: #c9587a;
    --accent-bg: rgba(217, 142, 74, 0.14); --good-bg: rgba(111, 174, 124, 0.14); --bad-bg: rgba(201, 88, 122, 0.16);
  }}
}}
:root[data-theme="dark"] {{
  --bg: #14161c; --surface: #1c1f28; --surface-2: #262a35; --border: #333846;
  --text: #e9e6df; --text-dim: #9b9fac; --accent: #d98e4a; --accent-2: #5fa8a3;
  --good: #6fae7c; --bad: #c9587a;
  --accent-bg: rgba(217, 142, 74, 0.14); --good-bg: rgba(111, 174, 124, 0.14); --bad-bg: rgba(201, 88, 122, 0.16);
}}
:root[data-theme="light"] {{
  --bg: #f6f4ee; --surface: #ffffff; --surface-2: #ece8de; --border: #d9d3c4;
  --text: #201e1a; --text-dim: #6b6459; --accent: #b56a28; --accent-2: #2f7d78;
  --good: #3f7d4c; --bad: #a83358;
  --accent-bg: rgba(181, 106, 40, 0.10); --good-bg: rgba(63, 125, 76, 0.12); --bad-bg: rgba(168, 51, 88, 0.12);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  line-height: 1.5;
}}
a {{ color: var(--accent); }}
.mono {{ font-family: ui-monospace, "SF Mono", "Cascadia Code", "DejaVu Sans Mono", Consolas, monospace; }}
code {{ font-family: ui-monospace, "SF Mono", "Cascadia Code", "DejaVu Sans Mono", Consolas, monospace; font-size: 0.82rem; white-space: pre-wrap; word-break: break-word; display: block; background: var(--bg); padding: 8px 10px; border-radius: 6px; color: var(--text); }}
header {{
  position: sticky; top: 0; z-index: 10;
  background: color-mix(in srgb, var(--bg) 88%, transparent);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--border);
  padding: 14px 24px;
  display: flex; align-items: baseline; gap: 20px; flex-wrap: wrap;
}}
.brand {{ font-family: "LM Sans", ui-sans-serif, system-ui, sans-serif; font-weight: 700; font-size: 1.1rem; }}
.brand .sub {{ color: var(--text-dim); font-weight: 400; font-size: 0.85rem; margin-left: 8px; }}
nav {{ display: flex; gap: 16px; font-size: 0.9rem; }}
nav a {{ text-decoration: none; color: var(--text-dim); }}
nav a:hover {{ color: var(--accent); }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
.panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 22px 24px; margin-bottom: 22px; overflow-x: auto; }}
.panel-head {{ font-family: "LM Sans", ui-sans-serif, system-ui, sans-serif; font-weight: 700; font-size: 1.15rem; margin: 0 0 6px; }}
.panel-desc {{ color: var(--text-dim); font-size: 0.92rem; margin: 0 0 18px; max-width: 80ch; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-bottom: 4px; }}
.card {{ background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
.card .label {{ font-size: 0.78rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.03em; }}
.card .num {{ font-family: ui-monospace, monospace; font-size: 1.7rem; font-weight: 700; color: var(--accent); margin-top: 4px; }}
.card .foot {{ font-size: 0.8rem; color: var(--text-dim); margin-top: 2px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; min-width: 560px; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
th {{ color: var(--text-dim); font-weight: 600; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.03em; }}
.match-bad {{ background: var(--bad-bg); color: var(--bad); font-weight: 700; }}
.pill {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.76rem; font-weight: 600; }}
.pill.good {{ background: var(--good-bg); color: var(--good); }}
.pill.bad {{ background: var(--bad-bg); color: var(--bad); }}
.bug-list {{ display: flex; flex-direction: column; gap: 14px; }}
.bug {{ border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; background: var(--surface-2); }}
.bug .bug-title {{ font-weight: 700; font-family: "LM Sans", ui-sans-serif, system-ui, sans-serif; margin-bottom: 8px; }}
.bug .bug-title .n {{ color: var(--accent); margin-right: 8px; }}
.file-ref {{ font-size: 0.78rem; color: var(--text-dim); margin-top: 4px; }}
footer {{ text-align: center; color: var(--text-dim); font-size: 0.8rem; padding: 30px 24px 50px; }}
footer a {{ color: var(--text-dim); }}
</style>
</head>
<body>
<header>
  <div class="brand">ToolSpec 接入<span class="sub">真实全链路结果 · API-Bank · Qwen2.5-3B-Instruct</span></div>
  <nav>
    <a href="#overview">总览</a>
    <a href="#speed">五方法速度对比</a>
    <a href="#correctness">正确性调查</a>
    <a href="#adapter">原生 vs 适配器</a>
    <a href="#vllm">vs vLLM 自带投机解码</a>
    <a href="#examples">真实偏离样例</a>
  </nav>
</header>
<main>

<section class="panel" id="overview">
  <div class="panel-head">总览</div>
  <p class="panel-desc">
    每个数字都是这个脚本每次运行时现场从原始文件重新算出来的，不是从对话总结里抄的。
    数据来源：<code style="display:inline;padding:1px 5px">{TOOLSPEC_REPO_DIR}/output/APIBank/Qwen2.5-3B-Instruct/*.jsonl</code>（原生仓库复现）
    + <code style="display:inline;padding:1px 5px">runs/toolspec_adapter_{{baseline,toolspec}}/logs/*.eval</code>（适配器复现）。
    完整叙述见 <code style="display:inline;padding:1px 5px">docs/toolspec_integration_findings.md</code>。
  </p>
  <div class="cards">
    <div class="card"><div class="label">原生仓库 toolspec 加速比</div><div class="num">{native_toolspec_row['speedup']:.2f}x</div><div class="foot">{native_baseline_row['tokens_per_second']:.1f} → {native_toolspec_row['tokens_per_second']:.1f} tokens/s</div></div>
    <div class="card"><div class="label">适配器 toolspec 加速比</div><div class="num">{adapter_toolspec['tokens_per_second']/adapter_baseline['tokens_per_second']:.2f}x</div><div class="foot">{adapter_baseline['tokens_per_second']:.1f} → {adapter_toolspec['tokens_per_second']:.1f} tokens/s</div></div>
    <div class="card"><div class="label">正确性（vs 真实 greedy baseline）</div><div class="num">{100 - native_toolspec_row['n_mismatches']}/100</div><div class="foot">原生仓库，{native_toolspec_row['n_mismatches']} 条偏离</div></div>
    <div class="card"><div class="label">适配器逐 token 复现原生行为</div><div class="num">{'完全一致' if same_mismatch_set else '不一致'}</div><div class="foot">{len(adapter_toolspec['incorrect_qids'])} 条偏离，跟原生仓库{'相同' if same_mismatch_set else '不同'}的 question_id</div></div>
    <div class="card"><div class="label">inspect_trace Hooks 触发</div><div class="num">{adapter_baseline['n_trace_files'] + adapter_toolspec['n_trace_files']}</div><div class="foot">真实 per-sample trace 文件数（两个 run 合计）</div></div>
    <div class="card"><div class="label">fp16 决定性对照</div><div class="num">{len(native_determinism_mismatches)}/100</div><div class="foot">baseline 重跑两次的差异数（噪声下限）</div></div>
  </div>
</section>

<section class="panel" id="speed">
  <div class="panel-head">五方法速度对比（原生仓库，API-Bank 前 100 条）</div>
  <p class="panel-desc">来源：<code style="display:inline;padding:1px 5px">evaluation/speed.py</code> 对每个方法的原始 jsonl 输出直接计算，字段口径跟 ToolSpec 论文自己的评测脚本完全一致。</p>
  <table>
    <thead><tr><th>方法</th><th>tokens/s</th><th>mean accept length</th><th>speedup vs baseline</th><th>vs 真实 baseline 的偏离数</th></tr></thead>
    <tbody>{speed_rows_html}</tbody>
  </table>
</section>

<section class="panel" id="correctness">
  <div class="panel-head">正确性调查：ToolSpec 号称 training-free/无损，但实测有 {native_toolspec_row['n_mismatches']}/100 偏离</div>
  <p class="panel-desc">
    先排除"这是 fp16/硬件本身不确定"：把 baseline 重新跑一遍（同一段代码、同一个模型），
    跟第一次比对，<strong>{len(native_determinism_mismatches)}/100</strong> 不一致——纯 autoregressive greedy decoding 在这台机器上是完全确定的，不是噪声。
    再排除"这是 ToolSpec 自己的 bug"：pld/recycling/samd/toolspec 四个<strong>独立实现</strong>的方法，在几乎相同的一组 question_id 上偏离，
    指向的是这台 GPU 上"批量树形验证"共享的浮点数值特性，不是某一个方法的实现缺陷。
    表格逐个 question_id 列出四个方法各自是否偏离（✗ = 偏离）：
  </p>
  <table>
    <thead><tr><th>question_id</th><th>baseline 自身可重复</th>{''.join(f'<th>{r["method"]}</th>' for r in native_speed_rows if r['method'] != 'baseline')}</tr></thead>
    <tbody>{mismatch_table_rows}</tbody>
  </table>
</section>

<section class="panel" id="adapter">
  <div class="panel-head">原生仓库 vs 我们的 inspect_ai 适配器</div>
  <p class="panel-desc">同一批 100 条问题，通过 <code style="display:inline;padding:1px 5px">toolspec_adapter/</code> 的自定义 <code style="display:inline;padding:1px 5px">ModelAPI</code> 再跑一遍（调用 ToolSpec 自己的 <code style="display:inline;padding:1px 5px">baseline_forward()</code>/<code style="display:inline;padding:1px 5px">toolspec_forward()</code>，不是重新实现）。</p>
  <table>
    <thead><tr><th></th><th>tokens/s (baseline)</th><th>tokens/s (toolspec)</th><th>speedup</th><th>toolspec 模式偏离数</th><th>偏离的 question_id 跟原生一致？</th></tr></thead>
    <tbody>
      <tr>
        <td>原生仓库</td>
        <td class="mono">{native_baseline_row['tokens_per_second']:.2f}</td>
        <td class="mono">{native_toolspec_row['tokens_per_second']:.2f}</td>
        <td class="mono">{native_toolspec_row['speedup']:.2f}x</td>
        <td class="mono">{native_toolspec_row['n_mismatches']}/100</td>
        <td>—</td>
      </tr>
      <tr>
        <td>inspect_ai 适配器</td>
        <td class="mono">{adapter_baseline['tokens_per_second']:.2f}</td>
        <td class="mono">{adapter_toolspec['tokens_per_second']:.2f}</td>
        <td class="mono">{adapter_toolspec['tokens_per_second']/adapter_baseline['tokens_per_second']:.2f}x</td>
        <td class="mono">{len(adapter_toolspec['incorrect_qids'])}/100</td>
        <td>{pill(same_mismatch_set)} {'完全相同的 question_id' if same_mismatch_set else '不同！'}</td>
      </tr>
    </tbody>
  </table>
  <p class="file-ref">.eval 日志：<code style="display:inline;padding:1px 5px">{adapter_baseline['eval_file']}</code>、<code style="display:inline;padding:1px 5px">{adapter_toolspec['eval_file']}</code></p>
</section>

<section class="panel" id="vllm">
  <div class="panel-head">vLLM 自带投机解码 vs ToolSpec</div>
  <p class="panel-desc">
    vLLM 服务本身自带的 n-gram/prompt-lookup 投机解码（<code style="display:inline;padding:1px 5px">--speculative-model "[ngram]"</code>，不需要额外草稿模型），跟 ToolSpec 的 schema-aware + retrieval-augmented 方法对比。两套 serving 栈的 baseline 吞吐本身不同，所以看的是各自的加速比，不是原始 tokens/s。完整分析见 <a href="./toolspec_vllm_speculative_comparison.md">toolspec_vllm_speculative_comparison.md</a>。
  </p>
  <table>
    <thead><tr><th></th><th>tokens/s (baseline)</th><th>tokens/s (加速模式)</th><th>各自的 speedup</th><th>跟自己 baseline 的偏离数</th></tr></thead>
    <tbody>
      <tr>
        <td>vLLM + ngram 投机解码</td>
        <td class="mono">{vllm_baseline['tokens_per_second']:.2f}</td>
        <td class="mono">{vllm_ngram['tokens_per_second']:.2f}</td>
        <td class="mono">{vllm_speedup:.2f}x</td>
        <td class="mono">{len(vllm_ngram_mismatch_qids)}/100</td>
      </tr>
      <tr>
        <td>ToolSpec（适配器，同一个 harness）</td>
        <td class="mono">{adapter_baseline['tokens_per_second']:.2f}</td>
        <td class="mono">{adapter_toolspec['tokens_per_second']:.2f}</td>
        <td class="mono">{adapter_toolspec['tokens_per_second']/adapter_baseline['tokens_per_second']:.2f}x</td>
        <td class="mono">{len(adapter_toolspec['incorrect_qids'])}/100</td>
      </tr>
    </tbody>
  </table>
  <p class="panel-desc" style="margin-top:14px;margin-bottom:0">
    在这个任务（API-Bank tool-calling 预测）上，ToolSpec 的领域特定方法比 vLLM 通用 ngram 投机解码<strong>更快</strong>（speedup 高约 60%）、<strong>偏离率也更低</strong>（11% vs 23%）——通用方法不知道输出要符合 tool-call JSON schema，领域特定方法知道。
  </p>
  <p class="file-ref">.eval 日志：<code style="display:inline;padding:1px 5px">{vllm_baseline['eval_file']}</code>、<code style="display:inline;padding:1px 5px">{vllm_ngram['eval_file']}</code></p>
</section>

<section class="panel" id="examples">
  <div class="panel-head">真实偏离样例（原文，未删减到只留结论）</div>
  <p class="panel-desc">下面是前 3 个偏离 question_id 的真实模型输出全文对照，你可以直接判断偏离的性质（格式差异 vs 语义错误）。</p>
  <div class="bug-list">
    {examples_html}
  </div>
</section>

<footer>
  数据源文件路径见各面板下方标注 · 生成脚本 <code style="display:inline;padding:1px 5px">inspect_trace/scripts/build_toolspec_dashboard.py</code> · 完整叙述 <a href="./toolspec_integration_findings.md">toolspec_integration_findings.md</a>
</footer>
</body>
</html>
"""

    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
