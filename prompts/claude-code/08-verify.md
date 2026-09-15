Check the code against the paper and verify it matches.

Read paper/paper.pdf, paper/notes.md, tests/EXPECTED.md, every file in src/, and the reference notebooks and wikienv.py. Write reports/verification.md with:

1. One row per component from the checklist in paper/notes.md: component, paper section, our file and function, verdict MATCH / DIFFERS / NOT IN PAPER. For DIFFERS, quote the paper and show our code.

2. Evidence-backed checks (print the value or run the test; reasoning alone does not count):
   - The 500 evaluation indices equal the reference notebook's idxs[:500] for both tasks.
   - Exemplars in our prompts are byte-identical to the keys in reference/prompts.
   - Stop strings prevent the model from writing its own Observation; show a cached call where the raw output was cut.
   - The step limit is 7, forced finish[] scores 0, and hit_step_limit is set.
   - search returns 5 sentences on a known page and the Similar list on a known miss; lookup numbering matches the reference for the same page and keyword.
   - EM uses SQuAD normalization; FEVER accuracy requires a valid label.
   - CoT-SC uses n=21 at temperature 0.7 and the combination thresholds are n/2 and hit_step_limit.
   - Temperature and max_tokens are identical across conditions except CoT-SC.

3. Everything taken from the reference code rather than the paper, with the file and line.

4. Anything in our pipeline that the paper does not describe and that could move results (system message wording, client-side stop cutting, the chat-vs-completion adaptation), each with a one-line estimate of its likely effect.

If anything DIFFERS, do not fix it. Explain what it would change and stop.
