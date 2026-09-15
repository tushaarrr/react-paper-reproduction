Phase 3, step 1: bootstrap fine-tuning data, following Section 3.3 of paper/paper.pdf.

Add a train split to src/data.py reading data/hotpot_train_v1.1_simplified.json, and extend src/run.py with --split train.

Run the existing ReAct graph over the training split and keep only trajectories whose final answer is an exact match. Target 3,000 kept trajectories; stop early and ask me if the estimated cost passes $15. Also run the Act and CoT conditions over the SAME questions that ReAct got right, so all three fine-tuning datasets cover identical questions and any difference comes from the format, not the sample.

Write data/sft/{react,act,cot}_train.jsonl with one object per trajectory: question, the full text in the exemplar format, the final answer, n_steps, and the source index. Deduplicate by question.

Report: how many training questions were attempted, the keep rate, mean steps in kept trajectories, and total cost. Show me two kept trajectories in full.
