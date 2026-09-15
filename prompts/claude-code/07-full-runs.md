Run all five conditions on all 500 sampled questions for HotpotQA and then FEVER, in this order: standard, cot, act, react, cotsc (cotsc last because it is the expensive one; stop and tell me the cost so far before starting it). Then compute both combinations.

Write README.md with:
- A results table: condition, paper HotpotQA EM, ours, paper FEVER Acc, ours, mean steps, % hit step limit, cost.
- A "Claims" section with one line per claim and PASS/FAIL from our numbers: (1) ReAct > Act on both tasks; (2) ReAct > CoT on FEVER; (3) CoT-SC > CoT on both; (4) each combination > every single method on both.
- HotpotQA EM broken down by question type (bridge vs comparison) for CoT, Act and ReAct, since the data file has a type field.
- Three example trajectories: one ReAct success, one ReAct failure that CoT got right, one CoT hallucination that ReAct caught. Pick them from the logs, do not write them by hand.
