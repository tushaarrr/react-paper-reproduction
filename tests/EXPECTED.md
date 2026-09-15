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
- PaLM-8B and 62B, batch size 64, 4,000 steps for ReAct and Act.
- Claim A: fine-tuned ReAct-8B beats every PROMPTING method on the 540B model.
- Claim B: fine-tuning Standard or CoT is much worse than fine-tuning ReAct or Act.
