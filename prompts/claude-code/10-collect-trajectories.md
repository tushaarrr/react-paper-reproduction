Phase 3, step 1: bootstrap fine-tuning data, following Section 3.3 of paper/paper.pdf.

Add a train split to src/data.py reading data/hotpot_train_v1.1_simplified.json, and extend src/run.py with --split train.

Run the existing ReAct graph over the training split and keep only trajectories whose final answer is an exact match. Target 3,000 kept trajectories. TWO cost gates apply and they are cumulative, not alternatives: **$5 is binding** (CLAUDE.md rule 8) — stop and ask before the estimated cost passes $5, and separately before any run over 100 questions, which this sweep is by definition. **$15 is a second hard checkpoint** — stop and ask AGAIN at $15 even if the $5 gate was already approved. Approval at $5 authorises continuing to $15, never past it. Also run the Act and CoT conditions over the SAME questions that ReAct got right, so all three fine-tuning datasets cover identical questions and any difference comes from the format, not the sample.

Write data/sft/{react,act,cot}_train.jsonl with one object per trajectory: question, the full text in the exemplar format, the final answer, n_steps, and the source index. Deduplicate by question.

Report: how many training questions were attempted, the keep rate, mean steps in kept trajectories, and total cost. Show me two kept trajectories in full.
