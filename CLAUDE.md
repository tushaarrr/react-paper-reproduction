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
8. `prompts/claude-code/05-react-graph.md` — its 10-field state has no `action`, so `execute`
   had to recover the action by re-splitting the scratchpad tail (`rsplit(f"Action {i}: ", 1)`).
   That is not what the reference does (it holds the action in a local for the rest of the loop
   iteration) and it is wrong in both directions: `rsplit` truncates an action whose own text
   contains `Action {i}: ` (`Search[Action 1: The Movie]` → `the Movie]`), `split` truncates on a
   thought that does, and a chat model echoing the retry prompt's own `Action 1:` label gets
   silently "repaired" instead of scoring the `Invalid action:` the reference scores. The
   scratchpad is byte-correct either way, so `runs/` records nothing unusual while a different
   Wikipedia action was executed. State gains an 11th field, `action` (a plain `str` — unlike the
   env of D21, so checkpointing is unaffected). Corrected at source in the prompt file, recorded
   in `paper/notes.md` section 1, and pinned by `tests/test_graph_react.py`.
9. `src/llm.py` — `_key` json-dumped the raw `temperature` and `max_tokens`, so `0` and `0.0`
   hashed to different sha256s and a caller spelling temperature `0` (as `src/graph_react.py`
   does) missed every entry a `0.0` caller wrote — including the one already in
   `data/cache/llm/`. Rule 5's cache would have re-issued and re-billed a whole rerun while
   producing identical text. `_key` now normalises: `float(temperature)`, `int(max_tokens)`.
10. `paper/notes.md` — D3's parse bullet and the C6 row specified the answer as the text after the
    **LAST** `Answer:` in the completion. Wrong whenever the stop list fails: the completion then
    runs on into `Question: <next>\nThought: ...\nAnswer: <other>` and LAST returns a *different
    question's* answer, scoring it against this question's gold — a silent, plausible-looking
    wrong answer in `runs/`. Corrected to the FIRST occurrence (D37), with D3 retracted by name
    rather than overwritten, and C6's worked example fixed (it also assumed a first-line cut that
    no split-and-strip rule performs). Pinned by `tests/test_baselines.py`.
11. `paper/notes.md` (D41) — the decision cites question 5388, prediction `torpedo boats and
    submarines` against gold `torpedoes`, as "EM 0, F1 > 0". Measured: **F1 is 0.0**, because
    SQuAD normalization does not stem, so `torpedo` and `torpedoes` share no token. The decision
    (log EM and F1 everywhere) stands; the example does not, and a reader checking the claim would
    have concluded the F1 column was broken. Corrected in place with a pair from the same question
    that does show the gap — `torpedoes and submarines` → EM 0, F1 0.5 — and both pairs are pinned
    in `tests/test_run.py`.
12. `tests/test_llm.py` — asserted `llm.MAX_SPEND_USD == 5.00` after reloading the module, which
    re-reads `.env`. D42 raised the ceiling there to 20.00, so four parametrised cases failed on a
    change that was authorised and correct. The assertion now pins the invariant the module
    actually enforces (finite and non-negative), not the operator's current number.

### Rule 10

If you find another defect in a file this repo treats as authoritative — `CLAUDE.md`,
`tests/EXPECTED.md`, `paper/targets.csv`, anything under `prompts/` — fix it at source,
append a line to this log, and tell me. Never fix it silently. Never leave a file that
CLAUDE.md calls authoritative knowingly wrong, and never work around such a defect in
`src/` while leaving the source of truth uncorrected.

### Rule 11 — tests assert the outbound request, not just the return value

A fake LLM or fake env must RECORD every outbound request, and tests must assert what was
sent: the exact prompt text, the stop list, temperature and max_tokens on each call. A test
that only checks the returned value cannot see a client that silently stops forwarding
temperature, which would turn CoT-SC's 21 samples into 21 greedy duplicates at full price.

### Rule 12 — a fixture must separate every plausible wrong implementation

A fixture's values must be chosen so that every plausible WRONG implementation produces a
DIFFERENT result from the correct one. Before writing one, enumerate the wrong implementations
you are excluding, then pick values that separate all of them:

  aggregation   sum / max / first / last / mean      -> three rows with distinct values, no two
                                                        of which coincide under any of them
  metric        em / f1                              -> at least one row where em != f1
  counts        winner count / total / non-empty      -> a vote fixture where all three differ
  selection     first / last / lowest-index           -> a tie whose members sit at known indices

Three rows is the MINIMUM, not the requirement — the requirement is separation. A fixture where
every candidate implementation returns the same answer makes its assertion vacuous, however many
rows it has.

**Why:** three defects in this repo shared exactly this shape. A one-row spend ledger made sum,
max, first and last indistinguishable, so `_spend_so_far` returning `max` passed every budget test
while the $5 ceiling could never fire. Both combine fixtures had `f1 == em` on all ten rows, so the
results.csv metric column could silently become mean F1 — systematically higher than EM — and we
would have concluded we beat the paper. A single-sample vote fixture could not tell the winner's
count from the sample total.

Applies retroactively: every fixture in `tests/` is audited against this rule.

### Rule 13 — the test suite can never spend money

`tests/conftest.py` severs the real LLM client for every test by default. A test that
forgets to patch `llm.complete` / `llm._request` ERRORS instead of silently billing.
Opt in explicitly: `@pytest.mark.slow` for a live smoke test (deselected by default),
`@pytest.mark.builds_client` for a test that constructs a client to inspect its payload
and sends nothing.

**Why:** on 2026-09-15 a test written against a FACTORY fixture (`fake`) as though it
were the fake instance left `llm.complete` unpatched, and `cot_sc` issued 42 unauthorised
live calls ($0.003318) across two parametrised cases. The per-test discipline was correct
but easy to get subtly wrong; the suite-wide block makes the failure loud instead.

