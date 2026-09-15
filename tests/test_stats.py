"""C9 — paired statistics (src/stats.py). Pure arithmetic, no network, no fixtures
on disk: the runs are built inline so every count is visible in the test.

Rule 12 — the wrong implementations these fixtures are built to separate:

  1. mcnemar uses the chi-square approximation   -> (9,1) gives 0.0269 / 0.0114, not 0.0215
  2. mcnemar returns the one-sided p             -> (9,1) gives 0.0107, exactly half
  3. discordant_counts returns (c, b) swapped    -> DOM gives (0,9), not (9,0)
  4. the bootstrap resamples the two conditions
     independently                               -> SAME gives a wide CI, not (0.0, 0.0)
  5. concordant pairs are counted                -> TIE's 45 both-right / 15 both-wrong
                                                    are not 20, 40 or 100
  6. b + c is used as the statistic              -> TIE has b+c 40 but b == c == 20
  7. the basic (reversed) interval               -> DOM gives (0.03, 0.14), not (0.04, 0.15)
  8. the join is positional                      -> SHUFFLED reorders B's rows
  9. the seed is ignored                         -> Random() is called with None, not 4242
 10. the diff vector is in insertion order       -> TIE in gap-fill order gives (-0.12, 0.12)
 11. compare() computes with the module defaults
     while recording its arguments               -> ci_hi 0.15 beside a recorded n=200
 12. quantiles' "exclusive" default              -> TIE at 200 gives (-0.13975, 0.14975)
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import stats


def runs(pattern):
    """pattern: [(count, em_a, em_b)] -> (run_a, run_b), idx 0..n-1."""
    a, b, idx = [], [], 0
    for count, em_a, em_b in pattern:
        for _ in range(count):
            a.append({"idx": idx, "em": em_a, "prediction": "x"})
            b.append({"idx": idx, "em": em_b, "prediction": "y"})
            idx += 1
    return a, b


# EM is identical (0.65 each) but the two disagree on 40 questions: b == c == 20.
# The concordant counts (45 both-right, 15 both-wrong) match nothing else here, so an
# implementation that counts them lands on a number no assertion below accepts.
TIE = runs([(45, 1, 1), (15, 0, 0), (20, 1, 0), (20, 0, 1)])
# A strictly dominates B: it is right everywhere B is, and on 9 more. b != c, so a
# swapped return is visible; c == 0, so the exact tail is 2 / 2**9.
DOM = runs([(30, 1, 1), (61, 0, 0), (9, 1, 0)])
# Identical question by question: every paired resample has a difference of exactly 0.
SAME = runs([(50, 1, 1), (50, 0, 0)])


# -- discordant_counts -----------------------------------------------------

def test_only_the_pairs_that_disagree_are_counted(): 
    a, b = TIE
    assert stats.discordant_counts(a, b) == (20, 20)
    # not the 45 both-right, not the 15 both-wrong, not the 60 concordant, not n=100,
    # and not the 40 that b + c would give.
    assert sum(stats.discordant_counts(a, b)) == 40
    assert len(a) == 100


def test_b_and_c_are_not_interchangeable():
    a, b = DOM
    assert stats.discordant_counts(a, b) == (9, 0)   # A right, B wrong on 9
    assert stats.discordant_counts(b, a) == (0, 9)   # ...and never the reverse


def test_the_join_is_on_idx_not_position():
    a, b = DOM
    shuffled = list(reversed(b))
    assert stats.discordant_counts(a, shuffled) == (9, 0)
    assert stats.paired_bootstrap_ci(a, shuffled) == stats.paired_bootstrap_ci(a, b)
    # Reversing B alone proves nothing about the DIFF VECTOR: B is read through key
    # lookup, and it is A's key order that the vector is built in. See
    # test_a_tied_pair_gives_an_interval_straddling_zero for the A-side reorder.


def test_a_different_question_set_fails_loudly():
    a, b = DOM
    with pytest.raises(ValueError, match="not the same questions"):
        stats.discordant_counts(a, b[:-1])
    with pytest.raises(ValueError, match="not the same questions"):
        stats.discordant_counts(a, [dict(l, idx=l["idx"] + 1000) for l in b])
    # A resumed run that double-wrote a question would otherwise be silently deduped.
    with pytest.raises(ValueError, match="duplicate idx"):
        stats.discordant_counts(a + a[:1], b + b[:1])


# -- mcnemar ---------------------------------------------------------------

def test_mcnemar_matches_a_hand_computed_exact_binomial():
    """b=9, c=1: under H0 the 10 discordant pairs are fair coins, so

        p = 2 x P(X <= 1) = 2 x (C(10,0) + C(10,1)) / 2**10 = 2 x 11/1024
          = 22/1024 = 0.021484375

    The three values it must NOT be: 0.0107421875 (the same tail, one-sided),
    0.0268566955 (chi-square with continuity correction, (|9-1|-1)**2/10 = 4.9) and
    0.0114120364 (chi-square without it, 6.4).
    """
    assert stats.mcnemar(9, 1) == pytest.approx(0.021484375, rel=1e-12)
    assert stats.mcnemar(9, 1) == stats.mcnemar(1, 9)            # two-sided is symmetric
    assert stats.mcnemar(9, 1) != pytest.approx(0.0107421875, rel=1e-6)
    assert stats.mcnemar(9, 1) != pytest.approx(0.0268566955, rel=1e-6)
    assert stats.mcnemar(9, 1) != pytest.approx(0.0114120364, rel=1e-6)


def test_identical_em_is_not_significant():
    a, b = TIE
    assert sum(l["em"] for l in a) == sum(l["em"] for l in b) == 65
    assert stats.mcnemar(*stats.discordant_counts(a, b)) == 1.0


def test_dominance_is_significant_and_one_sided_is_not_the_answer():
    a, b = DOM
    disc_b, disc_c = stats.discordant_counts(a, b)
    assert (disc_b, disc_c) == (9, 0)
    assert stats.mcnemar(disc_b, disc_c) == pytest.approx(2 / 2 ** 9)  # 0.00390625
    assert stats.mcnemar(disc_b, disc_c) < 0.05


def test_no_disagreement_is_p_one_and_not_a_zero_division():
    a, b = SAME
    assert stats.discordant_counts(a, b) == (0, 0)
    assert stats.mcnemar(0, 0) == 1.0


# -- paired_bootstrap_ci ---------------------------------------------------

def test_the_bootstrap_resamples_questions_not_conditions():
    """SAME agrees question by question, so every PAIRED resample has difference
    exactly 0 and the interval collapses. Drawing the two conditions independently
    would put ~0.25/n of variance back in and give roughly (-0.14, +0.14)."""
    a, b = SAME
    assert stats.paired_bootstrap_ci(a, b) == (0.0, 0.0)


def test_the_interval_is_the_percentile_interval_and_is_reproducible():
    a, b = DOM
    ci = stats.paired_bootstrap_ci(a, b)
    assert ci == (0.04, 0.15)                     # basic/reversed would be (0.03, 0.14)
    assert stats.paired_bootstrap_ci(a, b) == ci  # same seed, same interval
    # The seed is really used: at 10,000 resamples the interval has converged onto the
    # 1/100 grid these EM differences live on, so seed sensitivity only shows at 200.
    assert (stats.paired_bootstrap_ci(a, b, n_resamples=200, seed=1)
            != stats.paired_bootstrap_ci(a, b, n_resamples=200, seed=2))
    assert ci[0] > 0  # A dominates, so the interval excludes 0 — same story as p < 0.05


def test_a_tied_pair_gives_an_interval_straddling_zero():
    a, b = TIE
    lo, hi = stats.paired_bootstrap_ci(a, b)
    assert (lo, hi) == (-0.13, 0.12)
    assert lo < 0 < hi


def test_the_interval_does_not_depend_on_the_order_the_lines_were_written(): 
    """Rule 6: the interval is a function of (data, seed, n_resamples) and of nothing
    else — least of all the order the lines happen to sit in the JSONL, which
    src/run.py's own resume path scrambles (it appends in whatever order it filled its
    gaps). Dropping the `sorted` in `paired_bootstrap_ci` publishes (-0.12, 0.12) from
    this identical data at the shipped 10,000 resamples: same seed, same rows, a
    different published CI. TIE, not DOM — DOM's interval is order-insensitive here."""
    a, b = TIE
    gap_fill = sorted(a, key=lambda l: (l["idx"] % 7, l["idx"]))  # a resumed run's order
    assert stats.paired_bootstrap_ci(gap_fill, b) == (-0.13, 0.12)
    assert stats.paired_bootstrap_ci(list(reversed(a)), list(reversed(b))) == (-0.13, 0.12)


def test_the_seed_reaches_the_generator(monkeypatch):
    """A published interval must reproduce (rule 6), and NO assertion on the returned
    value can prove the seed is used: at 10,000 resamples the interval has converged
    onto the 1/100 grid these EM differences live on, so an implementation with no seed
    at all returns (0.04, 0.15) every time. Assert the OUTBOUND argument (rule 11)."""
    seen, real = [], random.Random  # bound BEFORE the patch, or the spy calls itself
    monkeypatch.setattr(stats.random, "Random",
                        lambda s=None: seen.append(s) or real(s))
    stats.paired_bootstrap_ci(*DOM, n_resamples=10, seed=4242)
    assert seen == [4242]  # not [None] from an unseeded Random(), not [stats.SEED]
    # ...backed by a fixture small enough that the interval has NOT converged: an
    # unseeded implementation reproduces this exact pair ~1.6% of the time.
    assert stats.paired_bootstrap_ci(*DOM, n_resamples=97, seed=3) == (0.05, 0.156)


def test_the_percentile_interval_uses_the_inclusive_method():
    """`method="inclusive"` is a deliberate argument that nothing else defends. The two
    methods agree at the shipped 10,000 resamples (both converged), so the only way to
    pin it is a small n_resamples: "exclusive" — quantiles' DEFAULT — gives
    (-0.13975, 0.14975) here, about a point wider on each side, and raises
    StatisticsError below 39 data points where "inclusive" still answers."""
    assert stats.paired_bootstrap_ci(*TIE, n_resamples=200) == pytest.approx(
        (-0.13025, 0.14025))
    assert stats.paired_bootstrap_ci(*TIE, n_resamples=20)  # "exclusive" would raise


def test_compare_records_the_seed_so_the_interval_can_be_reproduced():
    a, b = DOM
    out = stats.compare(a, b, "react", "act")
    assert out["seed"] == stats.SEED and out["n_resamples"] == 10000
    assert (out["a"], out["b"], out["n"]) == ("react", "act", 100)
    assert (out["em_a"], out["em_b"], out["em_diff"]) == (0.39, 0.30, 0.09)
    assert (out["discordant_b"], out["discordant_c"]) == (9, 0)
    assert out["p_mcnemar_exact"] == pytest.approx(0.00390625)
    assert (out["ci_lo"], out["ci_hi"]) == (0.04, 0.15)


def test_compare_computes_with_the_arguments_it_records_not_the_defaults():
    """Called ON the defaults, a recorded argument and a hardcoded constant are the same
    number, so the test above cannot tell them apart. Off the defaults they separate,
    and rule 6 lives exactly there: the seed/n_resamples columns of a results row must
    regenerate the ci_lo/ci_hi printed beside them, or a borderline
    SUPPORTED/INCONCLUSIVE call cannot be audited from the table it was made from."""
    a, b = DOM
    out = stats.compare(a, b, "react", "act", n_resamples=200, seed=7)
    assert (out["n_resamples"], out["seed"]) == (200, 7)  # not 10000 / stats.SEED
    assert (out["ci_lo"], out["ci_hi"]) == stats.paired_bootstrap_ci(a, b, 200, 7)
    assert out["ci_hi"] == 0.15025  # the default 10,000 resamples would say 0.15
