# Progress

Run the prompts in order. Tick a box only when that step's tests pass.
Commit after every step. Tag `v1-reproduced` after step 08 comes back clean.

## Setup
- [ ] `./setup.sh` runs clean (prints "first 5 eval indices: [3687, 6238, 5388, 3522, 3824] OK")
- [ ] `.env` filled in with MODEL and an API key
- [ ] `git init && git add . && git commit -m "scaffold"`

## Phase 1 — reproduce the prompting results
- [ ] 01 summarize        -> paper/notes.md exists and matches tests/EXPECTED.md
- [ ] 02 environment      -> tests/test_wiki_env.py passes
- [ ] 03 data + metrics   -> eval indices and EM tests pass
- [ ] 04 llm client       -> cache test passes, results/calls.csv appears
- [ ] 05 react graph      -> fake-LLM tests pass, 3 real trajectories look sane
- [ ] 06 baselines        -> 100-question table printed next to targets.csv
- [ ] 07 full runs        -> README.md has the results table and 4 claims marked

## Phase 2 — verify
- [ ] 08 verify           -> reports/verification.md clean, no unexplained DIFFERS
- [ ] tag v1-reproduced

## Phase 3 — fine-tuning
- [ ] 10 collect          -> data/sft/*.jsonl, keep rate reported
- [ ] 11 train lora       -> adapters saved, masking test passes
- [ ] 12 evaluate + serve -> 3 fine-tuning claims marked PASS/FAIL

## Optional
- [ ] 09 reflexion
- [ ] text ReAct vs create_react_agent on the same 500
- [ ] a second model

## Before calling it done
- [ ] README opens by saying what was reproduced and what was NOT (PaLM-540B vs our model)
- [ ] Every number in the README is produced by a script, not typed by hand
- [ ] `pytest` passes from a clean clone after ./setup.sh
