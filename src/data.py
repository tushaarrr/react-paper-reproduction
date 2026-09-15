"""Datasets, the 500-item sample, and the metrics (paper/notes.md section 3).

Normalization, EM and F1 are reference/wrappers.py:42-78; the sampler is
hotpotqa.ipynb:126-132 / FEVER.ipynb:9362-9368.
"""

import json
import random
import re
import string
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PUNCT = set(string.punctuation)
YESNO = ("yes", "no", "noanswer")
FEVER_LABELS = ("supports", "refutes", "not enough info")

# Both notebooks pass range(7405) for BOTH tasks, so FEVER indices 7405-9998 are never
# sampled even though paper_dev.jsonl has 9,999 lines. Reproduce, do not fix
# (notes.md section 3 "The 500 evaluation items", tests/EXPECTED.md:10-11).
EVAL_N = 7405


def load_hotpotqa():
    """-> [(question, answer, type)]. The `type` is our deviation (03-data-metrics.md:3)."""
    with open(DATA_DIR / "hotpot_dev_v1_simplified.json") as f:
        return [(d["question"], d["answer"], d["type"]) for d in json.load(f)]


def load_fever():
    """-> [(claim, label)], wrappers.py:143-154."""
    with open(DATA_DIR / "paper_dev.jsonl") as f:
        return [(d["claim"], d["label"]) for d in map(json.loads, f)]


def eval_indices(n_total, k=500):
    idxs = list(range(n_total))
    random.Random(233).shuffle(idxs)
    return idxs[:k]


def eval_indices_for(task):
    """The bound is the constant 7405 for both tasks — never len(load_fever())."""
    assert task in ("hotpotqa", "fever"), task
    return eval_indices(EVAL_N)


def normalize_answer(s):
    """SQuAD normalization, wrappers.py:42-56.

    Composition order is load-bearing: punctuation is stripped BEFORE articles, so
    "the-film" -> "thefilm" (the regex can no longer see a word boundary) rather than
    "film". Inlined from white_space_fix(remove_articles(remove_punc(lower(s)))).
    """
    s = "".join(ch for ch in s.lower() if ch not in PUNCT)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1(pred, gold):
    """wrappers.py:58-78, returning the float its callers take at [0] (wrappers.py:122)."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p in YESNO and p != g:
        return 0.0
    if g in YESNO and p != g:
        return 0.0
    p_toks, g_toks = p.split(), g.split()
    num_same = sum((Counter(p_toks) & Counter(g_toks)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def fever_score(pred, gold):
    """The prediction must normalize to one of the three labels; anything else is 0 —
    including "" from a forced finish[] and plausible-looking strings like "maybe"."""
    p = normalize_answer(pred)
    return int(p in FEVER_LABELS and p == normalize_answer(gold))
