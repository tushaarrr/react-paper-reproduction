OPTIONAL extension (do phase 3 fine-tuning first).

Implement Reflexion (Shinn et al., NeurIPS 2023, arXiv 2303.11366) on top of our ReAct graph, for HotpotQA only.

Read the paper and the reflexion repo cloned into reference/reflexion. Add src/graph_reflexion.py: an outer loop of up to 5 trials per question. After each failed trial (EM = 0, judged by the environment as in the paper), a reflect node asks the model for a 2 to 3 sentence self-reflection using the repo's reflection prompt verbatim; reflections accumulate in state and are inserted into the ReAct prompt before the question on the next trial. Stop early on success.

Use the repo's 100-question HotpotQA sample so our numbers are comparable to the paper's HotpotQA figure, and also run our seed-233 first 100.

Report accuracy after trial 1, 2, 3, 4, 5 as a line chart (results/plots/reflexion_trials.png) and add a "Reflexion" section to README.md with the paper's curve description next to ours. Log two example reflections that led to a correct answer on the next trial.
