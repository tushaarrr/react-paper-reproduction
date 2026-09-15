# Acceptance values

These are verified facts about the authors' setup. Any implementation that does not
reproduce them is wrong, regardless of how reasonable it looks.

## Evaluation sample
- Both notebooks build `idxs = list(range(7405))`, shuffle with `random.Random(233)`,
  and take `idxs[:500]`.
- First five indices MUST be: `3687, 6238, 5388, 3522, 3824`
- This holds for FEVER too, even though `paper_dev.jsonl` has 9,999 lines and the
  HotpotQA dev file has 7,405 entries. Match the reference, do not "fix" it.

## Data files
- `data/hotpot_dev_v1_simplified.json` — 7,405 entries, keys: question, answer, type
- `data/hotpot_train_v1.1_simplified.json` — bootstrap source for fine-tuning
- `data/paper_dev.jsonl` — 9,999 lines, keys: id, verifiable, label, claim, evidence

## Prompt keys
- `prompts/prompts_naive.json`: webthink_simple6 (ReAct), webact_simple6 (Act),
  cotqa_simple6 (CoT), webqa_simple6 (Standard)
- `prompts/fever.json`: webthink_simple3, webact_simple3, cotqa_simple3, webqa_simple3

## Loop
- Step limit is 7 (`for i in range(1, 8)`); if not done after the loop, the code
  forces `finish[]`, which scores 0.
- Stop string is `f"\nObservation {i}:"`. On a parse failure the code makes a second
  call with `stop=["\n"]` to get the action alone, and counts it as a bad call.
- The action's first character is lowercased before being sent to the environment.

## Environment
- `search[entity]` fetches `https://en.wikipedia.org/w/index.php?search=<entity>`
  with spaces as `+`, returns the first 5 sentences, or
  `Could not find <entity>. Similar: [<up to 5 titles>].`
- `lookup[keyword]` returns `(Result i / n) <sentence>`, then `No more results.`
- Reward is 0 for every action; only the wrapper computes EM at the end.

## Metrics
- SQuAD normalization: lowercase, strip punctuation, remove articles (a|an|the),
  collapse whitespace. Then exact string equality.
- FEVER: prediction must normalize to supports / refutes / not enough info.

## Fine-tuning (paper Section 3.3)
- 3,000 bootstrapped trajectories, kept only where the final answer was correct.
  (The bootstrap itself is described in Section 3.2, under the "Finetuning" paragraph;
  Section 3.3 reports the results.)
- PaLM-8B and 62B, batch size 64, 4,000 steps for ReAct and Act.
  (Appendix B.1 adds: Standard and CoT get 2,000 steps on 8B and 1,000 on 62B.)
- Claim A: fine-tuning buys about ONE model tier, not two. Section 3.3, verbatim:
  "PaLM-8B finetuned ReAct outperforming all PaLM-62B prompting methods", and
  "PaLM-62B finetuned ReAct outperforming all 540B prompting methods".
  So the two testable orderings are:
    A1: 8B-finetuned ReAct  > every 62B  prompting method
    A2: 62B-finetuned ReAct > every 540B prompting method
- Claim B: fine-tuning Standard or CoT is much worse than fine-tuning ReAct or Act.
  Section 3.3's wording is "significantly worse ... for both PaLM-8/62B"; "much worse"
  above is a paraphrase that agrees in substance.

> CORRECTION (see CLAUDE.md "Corrections log"). An earlier version of this file stated
> Claim A as "fine-tuned ReAct-8B beats every PROMPTING method on the 540B model."
> That was WRONG: it attributed the 62B-finetuned result to the 8B model, overstating
> the effect by a full model tier. Corrected against the paper text.

Neither A1 nor A2 is directly reproducible here — we fine-tune one small open model, not
PaLM-8B/62B, and our teacher is a small API model rather than PaLM-540B. Phase 3 tests the
ORDERING at a single scale (fine-tuned ReAct beats fine-tuned Standard/CoT/Act, and beats
the same model prompted with six exemplars), not the absolute cross-scale claim.
