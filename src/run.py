"""The runner (C8 of paper/notes.md §6) — one condition, one task, one JSONL, one
results.csv row.

    python -m src.run --task hotpotqa --condition react --n 100 [--split dev]

Specs: CLAUDE.md rules 4, 6 and 8; D15 (step accounting), D21 (env ownership),
D25 (trajectory per condition), D39 (`winner_votes` / `empty_samples`), D41 (log EM
*and* F1 on every condition).

Load-bearing details:

* **`check_budget_for_batch` runs before the first call**, and a refusal is a
  `SystemExit`, not a warning. Rule 8's other trigger — "ask before any run over 100
  questions" — is `--yes-over-100`, so step 07's 500-question runs cannot start by
  autocomplete.
* **One `WikiEnv` for the whole run, reset once per question** (D21). It is a closure
  argument of the graph, never a state field.
* **`n_steps` is the loop index, never `env.steps`**, which reaches 8 after the forced
  `finish[]` (§1). `state["step"] - 1` is that index for both exits: a normal finish at
  step i leaves `step == i + 1`, and a step-limit episode leaves `step == 8` (D24).
* **F1 is logged for every condition** (D41): EM alone cannot tell a wrong answer from
  a right answer that EM rejected (`torpedo boats and submarines` vs `torpedoes`).
"""

import argparse
import json
from pathlib import Path

from src import baselines, data, graph_react, llm, wiki_env
from src.combine import write_jsonl, write_results_row

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
NO_TOOL = {"standard": "standard", "cot": "cot", "cotsc": "cot_sc"}

# Pre-flight sizing only — never a decoding parameter. ReAct/Act are billed per step,
# so the gate assumes a full 7-step episode plus a parse-failure retry (D42 measured
# 5.33 calls/question on the step-05 live run; the gate rounds up, never down), and
# ~9,500 prompt chars, the step-05 maximum (2,374 tokens), because llm._estimate wants
# the LAST step's prompt, not the mean.
CALLS_PER_Q = {"standard": 1, "cot": 1, "cotsc": baselines.N_SAMPLES, "act": 8, "react": 8}
LOOP_PROMPT_CHARS = 9500


def load(task, split):
    """-> [(idx, question, gold)] in evaluation order."""
    if task == "fever":
        assert split == "dev", "no FEVER train split in data/ (phase 3 is HotpotQA)"
        rows = data.load_fever()  # [(claim, label)]
    elif split == "dev":
        rows = [(q, a) for q, a, _ in data.load_hotpotqa()]
    else:
        with open(data.DATA_DIR / "hotpot_train_v1.1_simplified.json") as f:
            rows = [(d["question"], d["answer"]) for d in json.load(f)]
    # The 500-item sample is a dev construct (EXPECTED.md); train is read in file order.
    idxs = data.eval_indices_for(task) if split == "dev" else range(len(rows))
    return [(i, rows[i][0], rows[i][1]) for i in idxs]


def score(task, prediction, gold):
    """D41. FEVER's `em` column holds the label accuracy of §3 `fever_score`."""
    em = data.fever_score(prediction, gold) if task == "fever" \
        else data.exact_match(prediction, gold)
    return em, data.f1(prediction, gold)


def run_one(idx, question, gold, task, condition, graph=None, env=None):
    # ponytail: per-question cost is a before/after read of results/calls.csv, which is
    # O(rows) per question. ~7k rows at the end of a 500-question ReAct run, so a few
    # seconds total; carry the cost out of llm.complete only if that stops being true.
    before = llm._spend_so_far()
    if condition in NO_TOOL:
        out = getattr(baselines, NO_TOOL[condition])(question, task)
        n_steps, hit_step_limit = 0, False  # D15: no loop, and 0/false, never null
    else:
        env.reset()  # D21: once per question, by the runner, never by a node
        state = graph.invoke(graph_react.initial_state(question, task, condition))
        out = {
            "prediction": state["answer"], "trajectory": state["scratchpad"],
            "n_calls": state["n_calls"], "n_badcalls": state["n_badcalls"],
            "winner_votes": None, "empty_samples": None,
        }
        n_steps, hit_step_limit = state["step"] - 1, state["hit_step_limit"]
    em, f1 = score(task, out["prediction"], gold)
    return {
        "idx": idx, "question": question, "gold": gold,
        "prediction": out["prediction"], "em": em, "f1": round(f1, 6),
        "n_steps": n_steps, "n_calls": out["n_calls"], "n_badcalls": out["n_badcalls"],
        "hit_step_limit": hit_step_limit,
        "empty_samples": out["empty_samples"],  # cotsc only; null elsewhere (D39)
        "winner_votes": out["winner_votes"],    # cotsc only; null elsewhere (D39)
        "cost": round(llm._spend_so_far() - before, 10),
        "condition": condition, "trajectory": out["trajectory"],
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("hotpotqa", "fever"), required=True)
    p.add_argument("--condition", choices=tuple(NO_TOOL) + ("act", "react"), required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--split", choices=("dev", "train"), default="dev")
    p.add_argument("--yes-over-100", action="store_true",
                   help="CLAUDE.md rule 8: required to launch more than 100 questions")
    args = p.parse_args(argv)

    items = load(args.task, args.split)[:args.n]
    if len(items) > 100 and not args.yes_over_100:
        raise SystemExit(
            f"{len(items)} questions > 100: CLAUDE.md rule 8 wants an explicit ask. "
            f"Re-run with --yes-over-100 once it is authorised."
        )
    prompt_chars = (
        len(baselines.build_prompt(items[0][1], args.task, args.condition))
        if args.condition in NO_TOOL else LOOP_PROMPT_CHARS
    )
    if not llm.check_budget_for_batch(
        len(items), CALLS_PER_Q[args.condition], prompt_chars
    ):
        raise SystemExit("budget: refused before the first call (CLAUDE.md rule 8)")

    env = wiki_env.WikiEnv() if args.condition not in NO_TOOL else None
    graph = graph_react.build_graph(args.condition, env) if env else None
    lines = [run_one(i, q, g, args.task, args.condition, graph, env)
             for i, q, g in items]

    path = RUNS_DIR / f"{args.task}_{args.condition}_{llm.MODEL}.jsonl"
    write_jsonl(path, lines)
    row = write_results_row(args.task, args.condition, llm.MODEL, lines)
    print(f"{path}: {len(lines)} questions")
    print(dict(zip(["task", "condition", "model", "n", "metric", "mean_steps",
                    "pct_hit_step_limit", "total_cost"], row)))
    return lines


if __name__ == "__main__":
    main()
