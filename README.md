# ReAct, reproduced in LangGraph (and fine-tuned)

A from-scratch reproduction of **ReAct: Synergizing Reasoning and Acting in Language
Models** (Yao et al., ICLR 2023, [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)),
built as an explicit LangGraph state machine, verified against the paper, and then
extended by reproducing the paper's fine-tuning experiment (Section 3.3) with QLoRA
on a 3B open model.

> Fill this file in as you go. Every number below must come from a script in this
> repo, never typed by hand. Delete this quote block when the project is done.

## What this reproduces, and what it does not

The paper's Table 1 numbers come from PaLM-540B. This reproduction runs `<MODEL>`,
so the absolute numbers differ. What is being reproduced is the **ordering** of the
seven conditions and the paper's four claims. Both are stated and tested below.

## Results — prompting (500 HotpotQA + 500 FEVER, seed 233)

| Condition | Paper HotpotQA EM | Ours | Paper FEVER Acc | Ours | Mean steps | % hit step limit |
|---|---|---|---|---|---|---|
| Standard | 28.7 | | 57.1 | | | |
| CoT | 29.4 | | 56.3 | | | |
| CoT-SC | 33.4 | | 60.4 | | | |
| Act | 25.7 | | 58.9 | | | |
| ReAct | 27.4 | | 60.9 | | | |
| CoT-SC -> ReAct | 34.2 | | 64.6 | | | |
| ReAct -> CoT-SC | 35.1 | | 62.0 | | | |

### Claims

| # | Claim | Verdict |
|---|---|---|
| 1 | ReAct beats Act on both tasks | |
| 2 | ReAct beats CoT on FEVER | |
| 3 | CoT-SC beats CoT on both | |
| 4 | Each combination beats every single method | |

## Results — fine-tuning (Section 3.3)

| Model | Prompted / fine-tuned | Exemplars | EM | Prompt tokens/question | tok/s |
|---|---|---|---|---|---|

| # | Claim | Verdict |
|---|---|---|
| A | Fine-tuned ReAct-3B beats the same model prompted with 6 exemplars | |
| B | Fine-tuned ReAct beats fine-tuned CoT and fine-tuned Act | |
| C | Fine-tuning cuts prompt tokens per question substantially | |

## Verification

See [`reports/verification.md`](reports/verification.md) — a component-by-component
check of this code against the paper, with evidence for each.

## Run it

    ./setup.sh
    cp .env.example .env      # add MODEL and your key
    pytest
    python -m src.run --task hotpotqa --condition react --n 100

## Limitations

<!-- teacher model, student size, number of trajectories, cost ceiling -->

## Credits

Paper and original code by Yao et al. The authors' repository is vendored read-only
under `reference/` ([ysymyth/ReAct](https://github.com/ysymyth/ReAct)); the exemplar
prompts and dataset files in `prompts/` and `data/` are theirs, used unchanged.
