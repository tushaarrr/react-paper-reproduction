Read paper/paper.pdf.
Summarize the proposed algorithm and identify every component that needs to be implemented.

Write paper/notes.md with these sections:
1. The ReAct loop in one paragraph, then as pseudocode with the exact prompt format (Thought i / Action i / Observation i) and the stop condition.
2. The Wikipedia environment: the three actions, exactly what each returns on success and failure, and the step limit per task.
3. Datasets: HotpotQA and FEVER, which file, how the 500 evaluation items are selected, the answer format for each, and the metric with its normalization.
4. All seven conditions in Table 1 (Standard, CoT, CoT-SC, Act, ReAct, and the two combinations), with the precise rule for each combination and the number of self-consistency samples.
5. Prompting details: number of exemplars per task, decoding settings, stop tokens. Mark each as "stated in paper" or "must come from reference code".
6. Component checklist in dependency order, each with the test that proves it works: environment, data + metrics, LLM client + cache, ReAct graph, Act variant, the three no-tool baselines, the two combination rules, the runner, the results table.

Cross-check your section 2, 3 and 5 against tests/EXPECTED.md and flag any disagreement.
Do not write code yet.
