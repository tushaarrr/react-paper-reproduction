"""Paired significance tests over two `runs/*.jsonl` files (CLAUDE.md rule 6).

All seven conditions answer the SAME questions, so every comparison is paired and an
independent-samples test is the wrong test: the paper's ReAct-vs-Act gap is 1.7 points
while the standard error of one proportion at n=500 is ~2.2 points, so an unpaired
comparison cannot resolve the effect at all. The information about a difference lives
in the DISCORDANT pairs — questions the two conditions disagree on — and both tests
here use only those (the bootstrap does so implicitly: a concordant pair contributes
0 to every resample).

stdlib only, on purpose: nothing here is worth a scipy dependency.
"""

import math
import random
import statistics

SEED = 20260915  # compare() records it, so a published interval is reproducible


def _em_by_idx(run_a, run_b):
    """Join on `idx`, never positionally — the two runs share a question set, not an
    order, and a resumed run appends in whatever order it filled its gaps."""
    a = {line["idx"]: int(bool(line["em"])) for line in run_a}
    b = {line["idx"]: int(bool(line["em"])) for line in run_b}
    if len(a) != len(run_a) or len(b) != len(run_b):
        raise ValueError("duplicate idx in a run — a resumed run wrote a question twice")
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b))[:5], sorted(set(b) - set(a))[:5]
        raise ValueError(
            f"the two runs are not the same questions: {len(a)} vs {len(b)} rows, "
            f"only in A {only_a}, only in B {only_b} — not a paired comparison"
        )
    return a, b


def discordant_counts(run_a, run_b):
    """-> (b, c): b = A right and B wrong, c = B right and A wrong.

    The concordant pairs (both right, both wrong) are deliberately not returned: they
    carry no information about the difference, and a test that uses them is testing
    something else.
    """
    a, b = _em_by_idx(run_a, run_b)
    return (sum(1 for i in a if a[i] and not b[i]),
            sum(1 for i in a if b[i] and not a[i]))


def mcnemar(b, c):
    """-> the EXACT two-sided binomial p-value of McNemar's test.

    Under H0 each discordant pair is a fair coin, so b ~ Binomial(b + c, 0.5). Exact,
    not the chi-square approximation: b + c is small on the pairs that matter here
    (the paper's 1.7-point gap over 500 questions), and chi-square is unusable there
    even with a continuity correction. Two-sided by symmetry of Binomial(n, 0.5):
    2 x P(X <= min(b, c)), capped at 1 for the near-tie case.
    """
    n = b + c
    if n == 0:
        return 1.0  # the two conditions never disagreed: no evidence of a difference
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(run_a, run_b, n_resamples=10000, seed=SEED):
    """-> (lo, hi), the 2.5/97.5 percentile CI of the EM difference A - B.

    QUESTIONS are resampled, not conditions: one draw takes a question's result under
    BOTH conditions, which is what makes the interval paired. Resampling the two
    conditions independently would import the between-question variance the pairing
    exists to cancel and widen the interval by roughly sqrt(2) x the per-condition SE.
    Resampling the per-question difference is that same draw, written shorter.
    """
    a, b = _em_by_idx(run_a, run_b)
    diffs = [a[i] - b[i] for i in sorted(a)]  # sorted: one seed, one interval
    rng = random.Random(seed)
    n = len(diffs)
    means = [sum(rng.choices(diffs, k=n)) / n for _ in range(n_resamples)]
    qs = statistics.quantiles(means, n=40, method="inclusive")
    return qs[0], qs[38]  # percentile interval, NOT the basic/reversed one


def compare(run_a, run_b, name_a="A", name_b="B", n_resamples=10000, seed=SEED):
    """Everything a results table needs for one pair, seed included (rule 6)."""
    a, b = _em_by_idx(run_a, run_b)
    disc_b, disc_c = discordant_counts(run_a, run_b)
    lo, hi = paired_bootstrap_ci(run_a, run_b, n_resamples, seed)
    return {
        "a": name_a, "b": name_b, "n": len(a),
        "em_a": sum(a.values()) / len(a), "em_b": sum(b.values()) / len(b),
        "em_diff": (sum(a.values()) - sum(b.values())) / len(a),
        "discordant_b": disc_b, "discordant_c": disc_c,
        "p_mcnemar_exact": mcnemar(disc_b, disc_c),
        "ci_lo": lo, "ci_hi": hi, "n_resamples": n_resamples, "seed": seed,
    }
