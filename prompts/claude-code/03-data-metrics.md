Implement src/data.py:

- load_hotpotqa() from data/hotpot_dev_v1_simplified.json returning (question, answer, type) tuples; load_fever() from data/paper_dev.jsonl returning (claim, label).
- eval_indices(n_total, k=500): idxs = list(range(n_total)); random.Random(233).shuffle(idxs); return idxs[:k]. Both reference notebooks use range(7405), even though paper_dev.jsonl has 9,999 lines; match that. The first five indices must be 3687, 6238, 5388, 3522, 3824 (see tests/EXPECTED.md).
- normalize_answer() implementing SQuAD normalization: lowercase, remove punctuation, remove articles (a, an, the), collapse whitespace. exact_match(pred, gold) and f1(pred, gold) with the yes/no/noanswer special case.
- For FEVER, the prediction must be one of SUPPORTS / REFUTES / NOT ENOUGH INFO after normalization; anything else scores 0.

Tests: eval_indices(7405)[:5] equals the five values above; EM("The Chief of Protocol", "chief of protocol") is 1; F1 of a partial answer is between 0 and 1; a FEVER prediction of "supports." matches "SUPPORTS".

Print the first 10 sampled HotpotQA questions and their types so I can eyeball them.
