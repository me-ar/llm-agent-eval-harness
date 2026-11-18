"""Evaluation harness: run the task suite against one or more agent configurations.

Usage:
  python -m agentkit.harness                                  # open models (Qwen 0.5B + 1.5B) on CPU
  python -m agentkit.harness --backend anthropic               # needs ANTHROPIC_API_KEY
  python -m agentkit.harness --smoke                           # offline, scripted model
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .agent import Agent, Trace
from .tasks import Task, build_tasks
from .tools import build_shop_db, default_tools

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ["arithmetic", "sql", "units", "policy", "multi_tool"]


def classify_failure(trace: Trace, correct: bool) -> str:
    if correct:
        return "correct"
    if trace.stop_reason == "max_steps":
        return "ran out of steps"
    if trace.n_invalid and not trace.n_tool_calls:
        return "invalid tool-call format"
    if trace.n_tool_errors:
        return "tool error not recovered"
    if trace.n_tool_calls == 0:
        return "answered without tools"
    return "wrong answer after tool use"


def evaluate(agent: Agent, tasks: list[Task], label: str, log=print) -> tuple[pd.DataFrame, list[dict]]:
    rows, traces = [], []
    for t in tasks:
        tr = agent.run(t.id, t.question)
        ok = bool(tr.final_answer) and t.check(tr.final_answer)
        rows.append({"config": label, "task": t.id, "category": t.category, "correct": ok,
                     "steps": len(tr.steps), "tool_calls": tr.n_tool_calls, "invalid_replies": tr.n_invalid,
                     "tool_errors": tr.n_tool_errors, "seconds": tr.seconds,
                     "failure": classify_failure(tr, ok)})
        traces.append({"config": label, "question": t.question, "expected": t.expected, "correct": ok,
                       **tr.to_dict()})
        log(f"[{label}] {t.id:8s} {'PASS' if ok else 'FAIL'} steps={len(tr.steps)} "
            f"answer={str(tr.final_answer)[:60]!r} expected={t.expected}")
    return pd.DataFrame(rows), traces


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("config", sort=False)
    out = pd.DataFrame({
        "accuracy": g["correct"].mean().round(3),
        "avg_steps": g["steps"].mean().round(2),
        "avg_tool_calls": g["tool_calls"].mean().round(2),
        "invalid_reply_rate": (g["invalid_replies"].sum() / g["steps"].sum()).round(3),
        "tool_error_rate": (g["tool_errors"].sum() / g["tool_calls"].sum().clip(lower=1)).round(3),
        "avg_seconds_per_task": g["seconds"].mean().round(1),
    })
    for c in CATEGORIES:
        out[f"acc_{c}"] = df[df.category == c].groupby("config", sort=False)["correct"].mean().round(3)
    return out.reset_index()


def make_configs(backend: str, smoke: bool):
    """Returns (label, model_key, use_tools) triples and a loader that builds an LLM from a model_key."""
    if smoke:
        from .llm import ScriptedLLM
        script = ['{"thought": "use calculator", "tool": "calculator", "args": {"expression": "4837*2569"}}',
                  '{"final_answer": "12426253"}']
        scripts = {"agent": script * 40, "direct": ['{"final_answer": "42"}'] * 40}
        return [("scripted agent", "agent", True), ("scripted direct", "direct", False)], \
            lambda key: ScriptedLLM(scripts[key])
    if backend == "anthropic":
        from .llm import AnthropicLLM
        return [("Claude Haiku - direct (no tools)", "haiku", False),
                ("Claude Haiku - agent + tools", "haiku", True)], lambda key: AnthropicLLM()
    from .llm import HFLocalLLM
    return [("Qwen2.5-1.5B - direct (no tools)", "Qwen/Qwen2.5-1.5B-Instruct", False),
            ("Qwen2.5-1.5B - agent + tools", "Qwen/Qwen2.5-1.5B-Instruct", True),
            ("Qwen2.5-0.5B - agent + tools", "Qwen/Qwen2.5-0.5B-Instruct", True)], HFLocalLLM


def main(backend: str = "hf", smoke: bool = False, out: Path | None = None, limit: int | None = None) -> pd.DataFrame:
    out = out or ROOT / "reports"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    con = build_shop_db()
    tasks = build_tasks(con)[:limit]
    tools = default_tools(con)
    frames, traces, cache = [], [], {}
    configs, loader = make_configs(backend, smoke)
    for label, key, use_tools in configs:
        if smoke or key not in cache:          # scripted models are single-use; real models are reused
            cache = {key: loader(key)}         # keep only one model in memory at a time
        llm = cache[key]
        df, tr = evaluate(Agent(llm, tools, use_tools=use_tools), tasks, label)
        frames.append(df)
        traces += tr
    df = pd.concat(frames, ignore_index=True)
    summary = summarise(df)
    failures = df.groupby(["config", "failure"]).size().unstack(fill_value=0)
    print(summary.to_string(index=False))

    df.to_csv(out / "per_task_results.csv", index=False)
    (out / "summary.md").write_text(summary.to_markdown(index=False) + "\n\n" + failures.to_markdown() + "\n")
    with open(out / "traces.jsonl", "w") as fh:
        for t in traces:
            fh.write(json.dumps(t, default=str) + "\n")
    (out / "results.json").write_text(json.dumps({
        "n_tasks": len(tasks), "categories": dict(Counter(t.category for t in tasks)),
        "summary": summary.to_dict(orient="records"),
        "failures": {c: {k: int(v) for k, v in row.items()} for c, row in failures.iterrows()}}, indent=2) + "\n")

    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    configs, w = list(summary["config"]), 0.8 / len(summary)
    for i, cfg in enumerate(configs):
        row = summary[summary.config == cfg].iloc[0]
        vals = [row[f"acc_{c}"] for c in CATEGORIES] + [row["accuracy"]]
        ax.bar(np.arange(len(vals)) + i * w - 0.4 + w / 2, vals, w, label=cfg)
    ax.set_xticks(range(len(CATEGORIES) + 1), CATEGORIES + ["OVERALL"])
    ax.set(ylim=(0, 1.05), ylabel="accuracy", title=f"Task success by category ({len(tasks)} tasks)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "accuracy_by_category.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    failures.drop(columns=["correct"], errors="ignore").plot.barh(stacked=True, ax=ax, colormap="tab10")
    ax.set(xlabel="tasks", title="Why did tasks fail?")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "figures" / "failure_modes.png")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "anthropic"], default="hf")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, help="output directory (default: reports/)")
    a = ap.parse_args()
    main(a.backend, a.smoke, a.out or (Path("/tmp/agent_smoke") if a.smoke else None), a.limit)
