# ReAct reproduction — agent rules

This repo reproduces "ReAct: Synergizing Reasoning and Acting in Language Models"
(Yao et al., ICLR 2023, arXiv 2210.03629) in LangGraph, then reproduces the paper's
fine-tuning experiment (Section 3.3).

- Paper: `paper/paper.pdf`
- Numbers we compare against: `paper/targets.csv`
- Authors' original code: `reference/` (read-only — never edit, never import from it)
- Acceptance values that must hold: `tests/EXPECTED.md`
- Step-by-step prompts: `prompts/claude-code/` — run them in order
- Progress checklist: `PROGRESS.md` — tick a box only when its tests pass

## Rules

1. Implement from the paper and `paper/notes.md` first. Open `reference/` only to
   settle a detail the paper leaves out, and log every such detail in
   `paper/notes.md` under "Taken from code, not paper".
2. Prompts in `prompts/` are the authors' exemplars. Use them verbatim. Never
   rewrite an exemplar.
3. Build the agent as an explicit LangGraph `StateGraph` with nodes and conditional
   edges. Do NOT use `create_react_agent` or native tool-calling — the paper's ReAct
   is text-based (Thought / Action / Observation) and that is what we reproduce.
4. One model for all conditions. Temperature 0 everywhere except CoT-SC (0.7).
   Max 100 new tokens per step.
5. Every LLM call and every Wikipedia fetch goes through a disk cache keyed by the
   exact input. Log tokens and cost per call to `results/calls.csv`.
6. Every run writes one JSONL line per question to `runs/` with: idx, question,
   gold, prediction, em, n_steps, hit_step_limit, full trajectory.
   `results/results.csv` holds one row per (task, condition, model, n).
7. Each module gets tests in `tests/` BEFORE it is used in a run.
8. Ask before starting any run over 100 questions, and before any run whose
   estimated cost passes $5.
9. Never commit `data/`, `runs/`, `.env`, or model weights.
