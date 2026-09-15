"""C7 of paper/notes.md §6 — the two combination conditions.

Every test here is OFFLINE, and more than offline: `src.llm.complete` is replaced by a
function that RAISES, so any combination rule that tried to ask the model a question
fails loudly instead of silently costing money (D6/D40: these rules are pure
post-processing over two `runs/` files).

The fixture pair is C7's, literally: `tests/fixtures/combine_react.jsonl` and
`combine_cotsc.jsonl`, 5 rows joined on `idx`.

| row | idx  | react hit_step_limit | react em | cotsc winner_votes | cotsc empty_samples |
|-----|------|----------------------|----------|--------------------|---------------------|
| 0   | 3687 | false                | 0        | 15                 | 0                   |
| 1   | 6238 | true                 | 0        | 15                 | 0                   |
| 2   | 5388 | false                | 1        | 10                 | 3                   |
| 3   | 3522 | true                 | 0        | 10                 | 11                  |
| 4   | 3824 | false                | 1        | 11                 | 2                   |

Row 0 is the one that separates `hit_step_limit` from `em` (a finished-but-wrong ReAct
answer); rows 2 and 3 sit exactly ON the threshold (10 < 10.5, so both back off); row 4
sits one vote above it; and row 3's 11 empty samples make its 10 votes a *unanimous*
10-of-10 among the samples that parsed — which still backs off, because the
denominator is the paper's n = 21 and never the non-empty count (D13).

Rule 12 governs the two files' LAYOUT as much as their fields:

* `idx` holds the real first five evaluation indices, so the react file is in evaluation
  order and NOT in ascending-idx order. A `_combine` that sorted its output by idx would
  return the rows in a different order and is separated here.
* the cotsc file lists the same five rows in ascending-idx order, i.e. a DIFFERENT order
  from the react file, so a `_combine` that zipped the two files positionally instead of
  joining on `idx` mis-joins every row and trips the gold assertion.
* the cotsc `cost` column is five distinct values (sum 0.0083, max 0.0023, mean 0.00166,
  first 0.0019, last 0.0013), because the combination rows are all 0.0 by construction
  and an all-zero column cannot tell `total_cost` = sum from max / mean / first / last.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import baselines, combine, llm

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REACT = FIXTURES / "combine_react.jsonl"
COTSC = FIXTURES / "combine_cotsc.jsonl"
# The five rows' idx, in the react file's (= evaluation) order; row n is IDX[n].
IDX = [3687, 6238, 5388, 3522, 3824]


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Rule 11, inverted: the outbound request that must never happen."""
    def boom(*a, **k):
        raise AssertionError("combine issued an LLM call")
    monkeypatch.setattr(llm, "complete", boom)


def run(rule, tmp_path):
    out = tmp_path / f"hotpotqa_{rule}_gpt-4o-mini.jsonl"
    lines = getattr(combine, rule)(REACT, COTSC, out)
    assert lines == [json.loads(l) for l in out.read_text().splitlines()]
    return lines


# -- D40: the two rules ----------------------------------------------------

def test_react_to_cotsc_backs_off_exactly_where_react_hit_the_step_limit(tmp_path):
    lines = run("react_to_cotsc", tmp_path)
    assert [l["prediction"] for l in lines] == ["R0", "S1", "R2", "S3", "R4"]
    assert [l["source"] for l in lines] == ["react", "cotsc", "react", "cotsc", "react"]


def test_react_to_cotsc_keys_off_hit_step_limit_and_not_off_em(tmp_path):
    """idx 0: ReAct finished and was WRONG (em 0). The paper backs off when ReAct
    "fails to return an answer within given steps", not when it answers badly."""
    lines = run("react_to_cotsc", tmp_path)
    assert (lines[0]["em"], lines[0]["hit_step_limit"]) == (0, False)
    assert lines[0]["source"] == "react"
    assert lines[0]["prediction"] == "R0"  # an em-keyed rule returns S0 here


def test_cotsc_to_react_backs_off_below_ten_point_five_votes(tmp_path):
    lines = run("cotsc_to_react", tmp_path)
    assert [l["prediction"] for l in lines] == ["S0", "S1", "R2", "R3", "S4"]
    assert [l["source"] for l in lines] == ["cotsc", "cotsc", "react", "react", "cotsc"]


def test_exactly_ten_votes_falls_back_and_eleven_does_not(tmp_path):
    """The boundary, pinned from both sides. 10 < 10.5 -> ReAct; 11 -> CoT-SC.
    A `n // 2` threshold (10) keeps CoT-SC on the 10-vote rows and fails here."""
    lines = run("cotsc_to_react", tmp_path)
    votes = {l["idx"]: l for l in combine.read_jsonl(COTSC)}
    assert votes[IDX[2]]["winner_votes"] == 10 and lines[2]["source"] == "react"
    assert votes[IDX[3]]["winner_votes"] == 10 and lines[3]["source"] == "react"
    assert votes[IDX[4]]["winner_votes"] == 11 and lines[4]["source"] == "cotsc"
    assert combine.N_SAMPLES / 2 == 10.5


def test_the_denominator_is_21_and_never_the_non_empty_sample_count(tmp_path):
    """idx 3: 10 votes out of 10 samples that parsed, 11 that did not. Unanimous among
    the valid ones — and still a back-off, because n is 21 (D13)."""
    cotsc = {l["idx"]: l for l in combine.read_jsonl(COTSC)}[IDX[3]]
    assert (cotsc["winner_votes"], cotsc["empty_samples"]) == (10, 11)
    n_valid = combine.N_SAMPLES - cotsc["empty_samples"]
    assert cotsc["winner_votes"] > n_valid / 2  # an n_valid denominator keeps CoT-SC
    assert run("cotsc_to_react", tmp_path)[3]["source"] == "react"


def test_a_clearly_unconfident_vote_also_backs_off(tmp_path):
    """The other end of "below n/2": 0 votes, all 21 samples empty (D4's last bullet)."""
    assert combine._pick_cotsc_to_react({}, {"winner_votes": 0}) == "react"
    assert combine._pick_cotsc_to_react({}, {"winner_votes": 3}) == "react"
    assert combine._pick_cotsc_to_react({}, {"winner_votes": 21}) == "cotsc"


def test_the_threshold_is_pinned_at_every_vote_count_a_question_can_have():
    """Exhaustive over 0..21 — the only values `winner_votes` can take. The fallback
    set is exactly {0..10}, which kills `n // 2` (it keeps CoT-SC at 10).

    It cannot kill `<=` in place of `<`, and no test can: for integer votes and the
    paper's odd n, `votes < 10.5`, `votes <= 10.5` and `votes <= 10` are the same
    predicate. That is a property of n = 21, not a hole in the suite — and it is why
    the boundary is pinned from both sides (10 -> react, 11 -> cotsc) rather than by
    reading the operator off the source.
    """
    fallback = {v for v in range(combine.N_SAMPLES + 1)
                if combine._pick_cotsc_to_react({}, {"winner_votes": v}) == "react"}
    assert fallback == set(range(11))


def test_n_samples_agrees_with_the_baseline_that_produced_the_votes():
    assert combine.N_SAMPLES == baselines.N_SAMPLES == 21


# -- D15/D25: what a combination line carries ------------------------------

def test_each_line_inherits_steps_limit_and_trajectory_from_its_source(tmp_path):
    react = {l["idx"]: l for l in combine.read_jsonl(REACT)}
    cotsc = {l["idx"]: l for l in combine.read_jsonl(COTSC)}
    for line in run("react_to_cotsc", tmp_path):
        src = (react if line["source"] == "react" else cotsc)[line["idx"]]
        for field in ("prediction", "em", "f1", "n_steps", "hit_step_limit",
                      "trajectory", "n_calls", "n_badcalls", "gold"):
            assert line[field] == src[field], field
        assert line["condition"] == "react_to_cotsc"  # not the source's condition


def test_no_combination_line_is_charged_for_anything(tmp_path):
    path = tmp_path / "results.csv"
    path.touch()  # exists, size 0: an editor truncation, or a killed write (H-2's shape)
    for rule in ("react_to_cotsc", "cotsc_to_react"):
        lines = run(rule, tmp_path)
        assert [l["cost"] for l in lines] == [0.0] * 5
        row = combine.write_results_row("hotpotqa", rule, "gpt-4o-mini", lines, path=path)
        assert row[-1] == 0.0
    # results.csv is APPENDED to across runs, so the header is written exactly once, and
    # written even though the file already existed: `write_header = True` repeats it
    # between the two rows and `not path.exists()` omits it from a touched file, and
    # either leaves every later DictReader of results.csv reading garbage.
    written = path.read_text().splitlines()
    assert written[0] == ",".join(combine.RESULTS_HEADER)
    assert len(written) == 3
    assert written[1].startswith("hotpotqa,react_to_cotsc,")
    assert written[2].startswith("hotpotqa,cotsc_to_react,")


def test_a_mis_joined_pair_is_refused_rather_than_silently_combined(tmp_path):
    other = tmp_path / "other.jsonl"
    rows = combine.read_jsonl(COTSC)
    rows[2]["gold"] = "a different gold"
    combine.write_jsonl(other, rows)
    with pytest.raises(AssertionError):
        combine.react_to_cotsc(REACT, other, tmp_path / "out.jsonl")

    missing = tmp_path / "missing.jsonl"
    combine.write_jsonl(missing, combine.read_jsonl(COTSC)[:3])
    with pytest.raises(KeyError):
        combine.react_to_cotsc(REACT, missing, tmp_path / "out2.jsonl")


# -- results.csv (rule 6, D15) ---------------------------------------------

def test_the_results_row_carries_the_model_string_and_inherited_step_stats(tmp_path):
    path = tmp_path / "results.csv"
    lines = run("react_to_cotsc", tmp_path)
    combine.write_results_row("hotpotqa", "react_to_cotsc", "gpt-4o-mini", lines, path)
    rows = list(csv.DictReader(path.open()))
    assert list(rows[0]) == combine.RESULTS_HEADER
    assert rows[0]["model"] == "gpt-4o-mini"
    assert rows[0]["n"] == "5"
    # picks R0,S1,R2,S3,R4 -> em 0,1,1,0,1 -> 0.6; steps 4,0,3,0,5 -> 2.4
    assert rows[0]["metric"] == "0.6"
    assert rows[0]["mean_steps"] == "2.4"
    assert rows[0]["pct_hit_step_limit"] == "0.0"  # the limit rows were replaced
    assert rows[0]["total_cost"] == "0.0"


def test_a_no_loop_condition_writes_empty_step_columns_not_zero(tmp_path):
    path = tmp_path / "results.csv"
    lines = combine.read_jsonl(COTSC)
    combine.write_results_row("hotpotqa", "cotsc", "gpt-4o-mini", lines, path)
    row = list(csv.DictReader(path.open()))[0]
    assert (row["mean_steps"], row["pct_hit_step_limit"]) == ("", "")
    assert row["metric"] == "0.4"  # D15: "" means not applicable, 0 would be a value
    # rule 12: the only results row written over lines with DISTINCT, non-zero costs, so
    # `total_cost` is the SUM (0.0083) and not max 0.0023, mean 0.00166, first 0.0019 or
    # last 0.0013. Mean EM 0.4 and mean F1 0.63 also differ on these same five rows.
    assert row["total_cost"] == "0.0083"
    assert sum(l["f1"] for l in lines) / len(lines) == 0.63
