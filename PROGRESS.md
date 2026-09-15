# Progress

Run the prompts in order. Tick a box only when that step's tests pass.
Commit after every step. Tag `v1-reproduced` after step 08 comes back clean.

## Setup
- [x] `./setup.sh` runs clean (prints "first 5 eval indices: [3687, 6238, 5388, 3522, 3824] OK")
- [x] `.env` filled in with MODEL=gpt-4o-mini and a working OPENAI_API_KEY (auth verified)
- [x] `git init && git add . && git commit -m "scaffold"`

## Phase 1 — reproduce the prompting results
- [x] 01 summarize        -> paper/notes.md exists and matches tests/EXPECTED.md
      (3 disagreements FLAGGED, not amended: L44 partially untestable here,
       L45 misquotes paper 3.3, L46 is a paraphrase. See notes.md.)
- [ ] 02 environment      -> tests/test_wiki_env.py passes
- [ ] 03 data + metrics   -> eval indices and EM tests pass
- [x] 04 llm client       -> cache test passes, results/calls.csv appears
      (79 tests; 45 mutations killed + 3 adversarial budget holes closed;
       one live call: 1615+41 tok, $0.00026685)
- [x] 05 react graph      -> fake-LLM tests pass, 3 real trajectories look sane
      (99 tests; reference replay 23/23 byte-identical; 3 live questions,
       0/3 EM, 2 hit the 7-step limit, $0.004994)
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
