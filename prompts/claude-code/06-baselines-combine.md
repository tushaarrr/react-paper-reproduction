Implement src/baselines.py using the same exemplar files: standard (keys webqa_simple6 / webqa_simple3, answer only), cot (cotqa_simple6 / cotqa_simple3, greedy), and cot_sc (same CoT prompt, n=21 samples at temperature 0.7, majority vote after normalize_answer; record the vote count of the winner). Parse the final answer the same way the reference notebooks do for these prompts; check reference/ for the exact answer pattern and note it in paper/notes.md.

Implement src/combine.py that reads two runs/ JSONL files and writes a third:
- react_to_cotsc: use the ReAct prediction unless hit_step_limit is true, then use the CoT-SC prediction.
- cotsc_to_react: use the CoT-SC prediction unless its winning vote count is less than n/2 (10.5 for n=21), then use the ReAct prediction.

Implement src/run.py: --task {hotpotqa,fever} --condition {standard,cot,cotsc,act,react} --n 100 --model from .env. It writes runs/{task}_{condition}_{model}.jsonl and appends a summary row to results/results.csv (task, condition, model, n, metric, mean_steps, pct_hit_step_limit, total_cost).

Tests: majority vote tie-breaking is deterministic; both combination rules on a 5-line fixture produce the expected picks.

Run all five conditions on the first 100 HotpotQA questions, then both combinations, and print the results table next to paper/targets.csv.
