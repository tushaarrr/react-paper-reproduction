"""The two combination conditions (C7 of paper/notes.md §6), and the results.csv row.

Specs: D6 + D40 (the rules), D15 (what a combination line inherits), D25 (its
trajectory), D39 (`winner_votes`).

**This module issues no LLM calls and imports nothing that can make one.** Both rules
are pure post-processing over two `runs/` JSONL files joined on `idx`; that is what
makes their `total_cost` 0.0 and what lets step 07 re-derive them for free.

Every output line records `source` ∈ {"react", "cotsc"} — the fallback fraction is the
point of the experiment, not a detail: if ReAct's step-limit rate is high,
`react_to_cotsc` quietly collapses onto CoT-SC, and that must be visible from `runs/`
alone without re-reading either input.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "results" / "results.csv"
RESULTS_HEADER = [
    "task", "condition", "model", "n",
    "metric", "mean_steps", "pct_hit_step_limit", "total_cost",
]
# Paper §3.2's n, duplicated from baselines.N_SAMPLES rather than imported so that
# nothing here can reach src.llm; tests/test_combine.py pins the two together.
N_SAMPLES = 21
STEPPED = ("act", "react", "react_to_cotsc", "cotsc_to_react")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def _pick_react_to_cotsc(react, cotsc):
    """D40 A: ReAct, unless it never answered within the step limit."""
    return "cotsc" if react["hit_step_limit"] else "react"


def _pick_cotsc_to_react(react, cotsc):
    """D40 B: CoT-SC, unless its majority is thinner than n/2 = 10.5 votes.

    The denominator is the paper's `n` (21), never the non-empty sample count — a
    question whose samples mostly failed to parse backs off to ReAct by construction
    (D13). `< N_SAMPLES / 2` is deliberately the float 10.5: `// 2` would give 10 and
    keep CoT-SC on exactly-10 questions, which the paper backs off.
    """
    return "react" if cotsc["winner_votes"] < N_SAMPLES / 2 else "cotsc"


def _combine(react_path, cotsc_path, out_path, pick, condition):
    by_idx = {line["idx"]: line for line in read_jsonl(cotsc_path)}
    out = []
    for react in read_jsonl(react_path):
        cotsc = by_idx[react["idx"]]  # KeyError = the two runs are not the same items
        assert react["gold"] == cotsc["gold"], react["idx"]  # a mis-joined pair
        source = pick(react, cotsc)
        line = dict(react if source == "react" else cotsc)
        # D15: everything (prediction, em, f1, n_steps, hit_step_limit, trajectory)
        # is inherited from the line that supplied the prediction; only these change.
        line["source"] = source
        line["condition"] = condition
        line["cost"] = 0.0  # D6/D15: no LLM call was issued here.
        out.append(line)
    write_jsonl(out_path, out)
    return out


def react_to_cotsc(react_path, cotsc_path, out_path):
    return _combine(react_path, cotsc_path, out_path,
                    _pick_react_to_cotsc, "react_to_cotsc")


def cotsc_to_react(react_path, cotsc_path, out_path):
    return _combine(react_path, cotsc_path, out_path,
                    _pick_cotsc_to_react, "cotsc_to_react")


def write_results_row(task, condition, model, lines, path=None):
    """One results.csv row per (task, condition, model, n) — CLAUDE.md rule 6.

    `metric` is mean EM (mean FEVER accuracy on FEVER, which `run.py` stores in the
    same `em` field). D15: `mean_steps` / `pct_hit_step_limit` are the empty string,
    not 0, for the three conditions that have no loop — 0 would read as a measurement.
    """
    path = Path(RESULTS_CSV if path is None else path)  # module attr, so tests redirect
    n = len(lines)
    stepped = condition in STEPPED
    row = [
        task, condition, model, n,
        round(sum(l["em"] for l in lines) / n, 4) if n else "",
        round(sum(l["n_steps"] for l in lines) / n, 3) if stepped and n else "",
        round(100 * sum(bool(l["hit_step_limit"]) for l in lines) / n, 1)
        if stepped and n else "",
        round(sum(l["cost"] for l in lines), 6),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(RESULTS_HEADER)
        writer.writerow(row)
    return row
