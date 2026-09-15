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

## Corrections log

Defects found in this project's OWN files and fixed at source. Newest last.

1. `tests/EXPECTED.md` — fine-tuning Claim A said "fine-tuned ReAct-8B beats every
   PROMPTING method on the 540B model". Wrong by a full model tier. Paper §3.3 says
   "PaLM-8B finetuned ReAct outperforming all PaLM-62B prompting methods" and
   "PaLM-62B finetuned ReAct outperforming all 540B prompting methods". Split into
   testable orderings A1/A2 and marked not directly reproducible at our scale.
2. `prompts/claude-code/02-environment.md` — omitted two load-bearing behaviours of
   `reference/wikienv.py`: the disambiguation recursion `search_step("[" + entity + "]")`
   (why the exemplars read `Could not find [Adam Clayton Powell]` with literal brackets),
   and `clean_str`, including that the driver's `obs.replace('\\n','')` strips a literal
   backslash-n rather than a newline. Added both, plus five required mutation checks.
3. `prompts/claude-code/10-collect-trajectories.md` — its $15 cost gate silently conflicted
   with rule 8's $5. Both are now explicit and cumulative: $5 binding, $15 a second hard
   checkpoint that stops again even after $5 was approved.
4. `prompts/claude-code/05-react-graph.md` — specifies setting `hit_step_limit=True` and
   calling `finish[]` inside a LangGraph conditional edge. Not implementable: verified
   against langgraph 1.2.11 that a path function's state writes are discarded, so
   `hit_step_limit` would have stayed `False` in every step-limited episode and the
   README's `%hit_step_limit` column would have read 0% while being wrong. The step limit
   needs a `force_finish` NODE with a pure router. Corrected at source in the prompt file,
   and recorded in `paper/notes.md` (D22).
5. `.gitignore` / `setup.sh` — `.gitignore` excluded only `data/sft/` while rule 9 forbids
   committing `data/` at all. Widened, and `setup.sh` now restores `data/` and `prompts/`
   from the reference clone (both are byte-identical to it) so a clean clone still works.
6. `paper/notes.md` — its `results/calls.csv` header block declared five columns while
   `src/llm.py` writes the seven the step-04 brief specifies, and `sample_index` was
   described there only as a cache-key field. Downstream readers (step 07's cost gate,
   `results.csv` aggregation) are written against notes.md, so the two had to agree.
   Corrected to the seven columns, with a reason for each of the two extras — including
   the honest note that `cache_hit` is constant `False` by construction and exists so a
   later decision to log hits does not move the header. `tests/test_llm.py` now pins
   `CALLS_HEADER` against that literal, as it already did for the system message.
7. `paper/notes.md` — its price-table bullet specified that a model missing from `PRICES`
   "costs 0.0 and logs a warning". That silently disables rule 8: the pre-call estimate,
   every logged `cost_usd` and therefore the whole ledger go to zero together, so the $5
   ceiling can never fire — one typo in `.env`'s `MODEL` buys a 25x-priced model
   (~$66.50 for 14,000 calls) against a ledger reading $0.00. Corrected to a refusal
   before the first call; the deliberately-free phase-3 models get explicit `0.0 / 0.0`
   rows instead, so free is stated rather than assumed.

### Rule 10

If you find another defect in a file this repo treats as authoritative — `CLAUDE.md`,
`tests/EXPECTED.md`, `paper/targets.csv`, anything under `prompts/` — fix it at source,
append a line to this log, and tell me. Never fix it silently. Never leave a file that
CLAUDE.md calls authoritative knowingly wrong, and never work around such a defect in
`src/` while leaving the source of truth uncorrected.
