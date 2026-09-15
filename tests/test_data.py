"""C2 of paper/notes.md section 6. Acceptance values from tests/EXPECTED.md."""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    eval_indices,
    eval_indices_for,
    exact_match,
    f1,
    fever_score,
    load_fever,
    load_hotpotqa,
    normalize_answer,
)


def test_eval_indices_first_five():
    idxs = eval_indices(7405)
    assert idxs[:5] == [3687, 6238, 5388, 3522, 3824]
    assert len(idxs) == 500
    assert len(set(idxs)) == 500


def test_eval_indices_fever_uses_the_7405_bound():
    # The FEVER sampler must never see len(load_fever()) == 9999.
    assert eval_indices_for("fever") == eval_indices(7405)
    assert max(eval_indices_for("fever")) == 7390
    with pytest.raises(AssertionError):  # a typo'd task must not silently return indices
        eval_indices_for("hotpot")


def test_load_hotpotqa():
    data = load_hotpotqa()
    assert len(data) == 7405
    assert all(len(row) == 3 for row in data)
    assert Counter(t for _, _, t in data) == {"bridge": 5918, "comparison": 1487}
    # Field identity, not just arity: pins question vs answer against any swap.
    assert data[0] == (
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "yes",
        "comparison",
    )


def test_load_fever():
    data = load_fever()
    assert len(data) == 9999
    assert Counter(label for _, label in data) == {
        "SUPPORTS": 3333,
        "REFUTES": 3333,
        "NOT ENOUGH INFO": 3333,
    }
    assert data[0] == (
        "Colin Kaepernick became a starting quarterback during the 49ers 63rd season"
        " in the National Football League.",
        "NOT ENOUGH INFO",
    )


def test_exact_match_strips_case_and_articles():
    assert exact_match("The Chief of Protocol", "chief of protocol") == 1
    assert exact_match("a film", "film") == 1
    assert exact_match("Chief of Protocol", "chief") == 0
    assert exact_match("chief of protocol", "The Chief of Protocol") == 1  # gold normalized too
    assert type(exact_match("a film", "film")) is int  # not bool: rule 6 writes "em": 1


def test_normalize_answer_strips_punctuation_before_articles():
    # The composition order is observable here: with articles removed first the hyphen
    # gives "the" a word boundary and the result would be "film".
    assert normalize_answer("the-film") == "thefilm"
    assert normalize_answer("  The, Answer!  ") == "answer"
    assert normalize_answer("an apple") == "apple"  # all three articles, not just a|the
    assert normalize_answer("x—the—y") == "x— —y"  # articles become a space, not ""
    # Whitespace is COLLAPSED, not just stripped: removing a mid-string article leaves a
    # double space, and a prediction can carry a newline out of finish[].
    assert normalize_answer("Kingdom of the Isles") == "kingdom of isles"
    assert normalize_answer("Richard  Nixon\n") == "richard nixon"


def test_f1_partial_overlap_is_strictly_between_zero_and_one():
    score = f1("Richard Milhous Nixon", "Richard Nixon")
    assert isinstance(score, float)
    assert 0 < score < 1
    assert score == 0.8
    assert f1("b b", "b b c") == 0.8  # multiset intersection: repeated tokens count twice
    assert f1("1985", "Chief of Protocol") == 0.0  # zero overlap: every wrong answer
    assert f1("", "Kiss and Tell") == 0.0  # empty prediction: the forced finish[]


def test_f1_yes_no_short_circuits():
    # Prediction side: without the guard this overlaps and scores 0.5.
    assert f1("yes", "yes it is") == 0.0
    # Ground-truth side: without the guard this scores ~0.667.
    assert f1("the yes man", "yes") == 0.0
    assert f1("yes", "yes") == 1.0
    assert f1("no", "no he did not") == 0.0  # 0.4 without "no" in YESNO
    assert f1("noanswer", "noanswer given") == 0.0  # 0.667 without "noanswer"


def test_fever_score_accepts_only_the_three_labels():
    assert fever_score("supports.", "SUPPORTS") == 1
    assert fever_score("supports", "SUPPORTS.") == 1  # gold is normalized, not lowercased
    assert fever_score("NOT ENOUGH INFO", "not enough info") == 1
    assert fever_score("", "SUPPORTS") == 0  # forced finish[]
    assert fever_score("maybe", "SUPPORTS") == 0
    assert fever_score("probably true", "SUPPORTS") == 0
    assert fever_score("refutes", "SUPPORTS") == 0
    # The whitelist is on the label space, not on agreement with the gold.
    assert fever_score("maybe", "maybe") == 0
