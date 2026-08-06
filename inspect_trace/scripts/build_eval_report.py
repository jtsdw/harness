#!/usr/bin/env python3
"""Generic, self-contained HTML report for any single .eval log -- no inspect view, no server, no port forwarding.

Generalizes the one-off dashboard scripts already in this directory
(build_tau2_dashboard.py, build_toolspec_dashboard.py, build_r3_r4_dashboard.py, all written for
one specific comparison experiment) into something you can point at any run's output.

Reads real data only -- every number comes straight from inspect_ai's own EvalLog object
(read_eval_log), nothing is estimated or hand-entered. Produces one .html file with no external
assets (fonts/CSS/JS all inlined), so it's viewable by just opening the file directly, or copying
it back from a remote machine (scp/rsync) -- no need to keep a server running or forward a port.

Usage:
    uv run python scripts/build_eval_report.py <path-to-.eval-or-runs-dir> [-o output.html]

<path> can be:
  - a directory containing logs/*.eval (e.g. runs/goal1_bfcl_multi_turn_base/) -- picks the most
    recently modified .eval file in there
  - a direct path to a specific .eval file

Trade-off vs the one-off dashboards: this can't show a side-by-side comparison between two runs
(that needs to know what's being compared and how, which is inherently specific to each
experiment) -- it's a report of ONE run. For a real A-vs-B comparison, write a dedicated script
like the existing ones, or run this twice and look at two tabs.
"""

from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

LM_SANS_REGULAR = Path("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-regular.otf")
LM_SANS_BOLD = Path("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-bold.otf")


def resolve_eval_path(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.is_file():
        return p
    if p.is_dir():
        candidates = (
            sorted((p / "logs").glob("*.eval"))
            if (p / "logs").exists()
            else sorted(p.glob("*.eval"))
        )
        if not candidates:
            candidates = sorted(p.glob("**/*.eval"))
        if not candidates:
            sys.exit(f"ERROR: no .eval file found under {p}")
        return max(candidates, key=lambda f: f.stat().st_mtime)
    sys.exit(f"ERROR: {p} is neither a file nor a directory")


def truncate(s: str | None, n: int = 600) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


def esc(s: str) -> str:
    return html.escape(s or "")


def message_text(m) -> str:
    content = getattr(m, "content", m)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            text = getattr(c, "text", None)
            parts.append(text if text is not None else str(c))
        return "\n".join(parts)
    return str(content)


def sample_messages(sample) -> list[tuple[str, str]]:
    """Prefers the full conversation over the raw input.

    sample.messages includes model turns; sample.input is pre-generation only -- falls back to
    input if messages is empty.
    """
    msgs = sample.messages or []
    if not msgs and sample.input:
        raw = sample.input if isinstance(sample.input, list) else [sample.input]
        return [("user", message_text(m) if not isinstance(m, str) else m) for m in raw]
    return [(m.role, message_text(m)) for m in msgs]


SCORE_GOOD = {"c", "correct", "1", "1.0", "true"}
SCORE_BAD = {"i", "incorrect", "0", "0.0", "false"}


def score_pill(value) -> tuple[str, str]:
    """Returns (css_class, display_text)."""
    text = str(value)
    norm = text.strip().lower()
    if norm in SCORE_GOOD:
        return "good", text
    if norm in SCORE_BAD:
        return "bad", text
    return "warn", text


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path", help="Path to a .eval file, or a directory containing logs/*.eval"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HTML path (default: <eval-file-stem>_report.html next to the .eval file)",
    )
    args = parser.parse_args()

    eval_path = resolve_eval_path(args.path)
    output_path = (
        Path(args.output)
        if args.output
        else eval_path.with_name(eval_path.stem + "_report.html")
    )

    log = read_eval_log(str(eval_path))

    total_samples = len(log.samples)
    model_usage = (
        next(iter(log.stats.model_usage.values()), None)
        if log.stats.model_usage
        else None
    )
    input_tokens = model_usage.input_tokens if model_usage else None
    output_tokens = model_usage.output_tokens if model_usage else None

    started = log.stats.started_at
    completed = log.stats.completed_at

    score_rows = []
    if log.results:
        for s in log.results.scores:
            for metric_name, metric in s.metrics.items():
                score_rows.append((s.scorer, metric_name, metric.value))

    sample_rows = []
    for sample in log.samples:
        scores = sample.scores or {}
        first_scorer = next(iter(scores.keys()), None)
        score_obj = scores.get(first_scorer) if first_scorer else None
        pill_class, pill_text = (
            score_pill(score_obj.value) if score_obj else ("warn", "—")
        )
        output_text = sample.output.completion if sample.output else ""
        sample_rows.append(
            {
                "id": sample.id,
                "pill_class": pill_class,
                "pill_text": pill_text,
                "explanation": score_obj.explanation if score_obj else None,
                "output_preview": truncate(output_text, 160),
                "output_full": output_text,
                "messages": sample_messages(sample),
                "working_time": sample.working_time,
            }
        )

    lm_regular_b64 = (
        base64.b64encode(LM_SANS_REGULAR.read_bytes()).decode()
        if LM_SANS_REGULAR.exists()
        else ""
    )
    lm_bold_b64 = (
        base64.b64encode(LM_SANS_BOLD.read_bytes()).decode()
        if LM_SANS_BOLD.exists()
        else ""
    )

    score_cards_html = "\n".join(
        f"""<div class="card"><div class="label">{esc(scorer)} · {esc(metric)}</div><div class="num">{value:.3f}</div></div>"""
        if isinstance(value, (int, float))
        else f"""<div class="card"><div class="label">{esc(scorer)} · {esc(metric)}</div><div class="num">{esc(str(value))}</div></div>"""
        for scorer, metric, value in score_rows
    )

    table_rows_html = "\n".join(
        f"""<tr>
          <td class="mono">{esc(str(r["id"]))}</td>
          <td><span class="pill {r["pill_class"]}">{esc(r["pill_text"])}</span></td>
          <td>{esc(r["output_preview"])}</td>
          <td class="mono">{f"{r['working_time']:.2f}s" if r["working_time"] else "—"}</td>
          <td><a href="#sample-{esc(str(r["id"]))}">详情 ↓</a></td>
        </tr>"""
        for r in sample_rows
    )

    def detail_block(r: dict) -> str:
        turns = "\n".join(
            f"""<div class="turn role-{esc(role)}"><div class="role-tag">{esc(role)}</div><div class="turn-body">{esc(truncate(text, 2000))}</div></div>"""
            for role, text in r["messages"]
        )
        explanation_html = (
            f"""<p><strong>Score explanation:</strong></p><code>{esc(truncate(r["explanation"], 1500))}</code>"""
            if r["explanation"]
            else ""
        )
        return f"""<div class="bug" id="sample-{esc(str(r["id"]))}">
          <div class="bug-title"><span class="n">sample {esc(str(r["id"]))}</span><span class="pill {r["pill_class"]}">{esc(r["pill_text"])}</span></div>
          {turns}
          <p style="margin-top:10px"><strong>Model output (full):</strong></p>
          <code>{esc(r["output_full"])}</code>
          {explanation_html}
        </div>"""

    details_html = "\n".join(detail_block(r) for r in sample_rows)

    html_doc = f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Eval Report: {esc(str(log.eval.task))}</title>
<style>
@font-face {{ font-family: "LM Sans"; src: url(data:font/otf;base64,{lm_regular_b64}) format("opentype"); font-weight: 400; }}
@font-face {{ font-family: "LM Sans"; src: url(data:font/otf;base64,{lm_bold_b64}) format("opentype"); font-weight: 700; }}
:root {{
  --bg: #f6f4ee; --surface: #ffffff; --surface-2: #ece8de; --border: #d9d3c4;
  --text: #201e1a; --text-dim: #6b6459; --accent: #b56a28; --accent-2: #2f7d78;
  --good: #3f7d4c; --bad: #a83358;
  --good-bg: rgba(63, 125, 76, 0.12); --bad-bg: rgba(168, 51, 88, 0.12); --accent-bg: rgba(181, 106, 40, 0.10);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #14161c; --surface: #1c1f28; --surface-2: #262a35; --border: #333846;
    --text: #e9e6df; --text-dim: #9b9fac; --accent: #d98e4a; --accent-2: #5fa8a3;
    --good: #6fae7c; --bad: #c9587a;
    --good-bg: rgba(111, 174, 124, 0.14); --bad-bg: rgba(201, 88, 122, 0.16); --accent-bg: rgba(217, 142, 74, 0.14);
  }}
}}
:root[data-theme="dark"] {{
  --bg: #14161c; --surface: #1c1f28; --surface-2: #262a35; --border: #333846;
  --text: #e9e6df; --text-dim: #9b9fac; --accent: #d98e4a; --accent-2: #5fa8a3;
  --good: #6fae7c; --bad: #c9587a;
  --good-bg: rgba(111, 174, 124, 0.14); --bad-bg: rgba(201, 88, 122, 0.16); --accent-bg: rgba(217, 142, 74, 0.14);
}}
:root[data-theme="light"] {{
  --bg: #f6f4ee; --surface: #ffffff; --surface-2: #ece8de; --border: #d9d3c4;
  --text: #201e1a; --text-dim: #6b6459; --accent: #b56a28; --accent-2: #2f7d78;
  --good: #3f7d4c; --bad: #a83358;
  --good-bg: rgba(63, 125, 76, 0.12); --bad-bg: rgba(168, 51, 88, 0.12); --accent-bg: rgba(181, 106, 40, 0.10);
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.5; }}
a {{ color: var(--accent); }}
.mono {{ font-family: ui-monospace, "SF Mono", "Cascadia Code", Consolas, monospace; }}
code {{ font-family: ui-monospace, monospace; font-size: 0.82rem; white-space: pre-wrap; word-break: break-word; display: block; background: var(--bg); padding: 8px 10px; border-radius: 6px; color: var(--text); }}
header {{ position: sticky; top: 0; z-index: 10; background: color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter: blur(8px); border-bottom: 1px solid var(--border); padding: 14px 24px; }}
.brand {{ font-family: "LM Sans", ui-sans-serif, system-ui, sans-serif; font-weight: 700; font-size: 1.1rem; }}
.brand .sub {{ color: var(--text-dim); font-weight: 400; font-size: 0.85rem; margin-left: 8px; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
.panel {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 22px 24px; margin-bottom: 22px; overflow-x: auto; }}
.panel-head {{ font-family: "LM Sans", ui-sans-serif, system-ui, sans-serif; font-weight: 700; font-size: 1.15rem; margin: 0 0 6px; }}
.panel-desc {{ color: var(--text-dim); font-size: 0.9rem; margin: 0 0 16px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
.card {{ background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
.card .label {{ font-size: 0.76rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.03em; }}
.card .num {{ font-family: ui-monospace, monospace; font-size: 1.5rem; font-weight: 700; color: var(--accent); margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; min-width: 560px; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
th {{ color: var(--text-dim); font-weight: 600; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.03em; }}
.pill {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.76rem; font-weight: 600; }}
.pill.good {{ background: var(--good-bg); color: var(--good); }}
.pill.bad {{ background: var(--bad-bg); color: var(--bad); }}
.pill.warn {{ background: var(--accent-bg); color: var(--accent); }}
.bug-list {{ display: flex; flex-direction: column; gap: 14px; }}
.bug {{ border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; background: var(--surface-2); scroll-margin-top: 70px; }}
.bug-title {{ font-weight: 700; font-family: "LM Sans", ui-sans-serif, system-ui, sans-serif; margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }}
.bug-title .n {{ color: var(--accent); }}
.turn {{ display: flex; gap: 10px; margin-bottom: 8px; align-items: flex-start; }}
.turn .role-tag {{ flex-shrink: 0; width: 70px; text-align: right; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: var(--text-dim); padding-top: 3px; }}
.turn-body {{ flex: 1; padding: 6px 10px; border-radius: 8px; font-size: 0.85rem; background: var(--bg); border: 1px solid var(--border); white-space: pre-wrap; }}
footer {{ text-align: center; color: var(--text-dim); font-size: 0.8rem; padding: 30px 24px 50px; }}
</style>
</head>
<body>
<header>
  <div class="brand">Eval Report<span class="sub">{esc(str(log.eval.task))} · {esc(str(log.eval.model))}</span></div>
</header>
<main>

<section class="panel">
  <div class="panel-head">总览</div>
  <p class="panel-desc">
    源文件：<code style="display:inline;padding:1px 5px">{esc(str(eval_path))}</code><br>
    开始：{esc(str(started))} · 结束：{esc(str(completed))}
  </p>
  <div class="cards">
    <div class="card"><div class="label">样本数</div><div class="num">{total_samples}</div></div>
    {score_cards_html}
    <div class="card"><div class="label">Input tokens</div><div class="num">{input_tokens if input_tokens is not None else "—"}</div></div>
    <div class="card"><div class="label">Output tokens</div><div class="num">{output_tokens if output_tokens is not None else "—"}</div></div>
  </div>
</section>

<section class="panel">
  <div class="panel-head">逐样本结果</div>
  <table>
    <thead><tr><th>ID</th><th>分数</th><th>模型输出（预览）</th><th>耗时</th><th></th></tr></thead>
    <tbody>{table_rows_html}</tbody>
  </table>
</section>

<section class="panel">
  <div class="panel-head">逐样本详情（完整对话 + 完整输出）</div>
  <div class="bug-list">
    {details_html}
  </div>
</section>

<footer>
  生成脚本 <code style="display:inline;padding:1px 5px">inspect_trace/scripts/build_eval_report.py</code>，数据全部来自 <code style="display:inline;padding:1px 5px">read_eval_log()</code>，不是手填的
</footer>
</body>
</html>
"""

    output_path.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {output_path} ({len(html_doc)} bytes)")


if __name__ == "__main__":
    main()
