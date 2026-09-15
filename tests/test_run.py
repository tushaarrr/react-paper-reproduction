"""C8 of paper/notes.md §6 — the runner.

Every test here is OFFLINE. `src.llm.complete` (the one function both `baselines.py`
and `graph_react.py` call) is replaced by a fake that RECORDS every outbound request,
`wiki_env.WikiEnv` by a fake that records resets and actions, and `runs/`,
`results/results.csv` and `results/calls.csv` are all redirected into tmp_path — the
real ones are never opened, let alone written.

The ordering assertion matters as much as the field assertions: rule 8's gate is
useless if it runs after the first call, so the fakes share one `events` list and the
test asserts `events[0] == "budget"`.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import combine, data, llm, run

Q0 = 3687  # tests/EXPECTED.md: the first five eval indices are fixed
FINISH = "I know this.\nAction 1: Finish[Arthur's Magazine]"
SEARCH = [f"looking.\nAction {i}: Search[Arthur's Magazine]" for i in range(1, 9)]

# The vote fixture, shared with tests/test_baselines.py: 21 samples, of which 4 parse to
# nothing. Every count it could be confused with is a different number (rule 12) —
# winner 9, runner-up 5, third 3, empty 4, non-empty 17, distinct answers 3, total 21 —
# and the first answer to appear ("Bravo") is deliberately not the winner.
SAMPLES_9_5_3_4EMPTY = ["Bravo"] * 5 + ["Alpha"] * 9 + ["Charlie"] * 3 + [""] * 4

# Two questions whose loops differ in every number rule 12 cares about: question 1 needs
# a parse-failure retry and finishes at step 2 (n_steps 2, n_calls 3 — so `n_steps` can
# never be read as `n_calls`), question 2 runs to the step limit (n_steps 7). Mean steps
# 4.5 is none of max 7 / first 2 / last 7 / sum 9 / mean n_calls 5, and one of the two
# hitting the limit makes pct_hit_step_limit 50.0 — neither the fraction 0.5, nor the
# count 1, nor the 0.0/100.0 that an all-or-nothing fixture leaves indistinguishable.
RETRY_THEN_LIMIT = [
    "I rambled without an action label.",   # no "\nAction 1: " -> a bad call and a retry
    "Search[Arthur's Magazine]",            # the retry answers with the action alone
    "I know this.\nAction 2: Finish[Arthur's Magazine]",
] + SEARCH[:7]


class FakeEnv:
    """The WikiEnv contract (D7), plus the reset spy C8 asks for (D21)."""

    def __init__(self):
        self.resets = 0
        self.actions = []

    def reset(self):
        self.resets += 1
        self.actions = []

    def step(self, action):
        self.actions.append(action)
        if action.startswith("finish["):
            return "Episode finished, reward = 0\n", True, {
                "steps": len(self.actions), "answer": action[len("finish["):-1]}
        return "Arthur's Magazine is a magazine.", False, {
            "steps": len(self.actions), "answer": None}


class FakeLLM:
    """Records every outbound request (rule 11). `texts` is a list consumed one per
    request, or a callable (prompt, n) -> list[str]."""

    def __init__(self, texts):
        self.texts = texts
        self.calls = []

    def complete(self, prompt, stop, temperature=None, max_tokens=None, n=1):
        self.calls.append({"prompt": prompt, "stop": stop, "temperature": temperature,
                           "max_tokens": max_tokens, "n": n})
        if callable(self.texts):
            return self.texts(prompt, n)
        return [self.texts[(len(self.calls) - 1) % len(self.texts)]] * n


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """-> a factory: harness(texts) installs the fakes and returns the recorder."""
    monkeypatch.setattr(run, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(combine, "RESULTS_CSV", tmp_path / "results.csv")
    # The spend ledger starts NON-EMPTY (rule 12): with an empty one, a per-question
    # `cost` that logged the cumulative spend instead of this question's delta would
    # still read 0.0 on every line and be indistinguishable from the correct value.
    # $0.0031 of someone else's prior run is on it, and every line below must read 0.0.
    ledger = tmp_path / "calls.csv"
    ledger.write_text(
        ",".join(llm.CALLS_HEADER) + "\n"
        "2026-09-15T00:00:00,gpt-4o-mini,0,20000,500,0.0031,False\n"
    )
    monkeypatch.setattr(llm, "CALLS_CSV", ledger)
    monkeypatch.setattr(llm, "MODEL", "gpt-4o-mini")
    monkeypatch.setattr(run.wiki_env, "WikiEnv", FakeEnv)

    def factory(texts, fits=True):
        fake = FakeLLM(texts)
        fake.events = []
        original = fake.complete

        def complete(*a, **k):
            fake.events.append("call")
            return original(*a, **k)

        def budget(n_questions, calls_per_question, prompt_chars, max_tokens=100):
            fake.events.append("budget")
            fake.budget = (n_questions, calls_per_question, prompt_chars)
            return fits

        monkeypatch.setattr(llm, "complete", complete)
        monkeypatch.setattr(llm, "check_budget_for_batch", budget)
        return fake
    return factory


def lines_of(tmp_path, name):
    return [json.loads(l)
            for l in (tmp_path / "runs" / name).read_text().splitlines()]


# -- CLAUDE.md rule 8 ------------------------------------------------------

def test_the_budget_is_checked_before_the_first_call(harness, tmp_path):
    fake = harness(["Arthur's Magazine"])
    run.main(["--task", "hotpotqa", "--condition", "standard", "--n", "3"])
    assert fake.events[0] == "budget"
    assert fake.events.count("budget") == 1
    assert fake.events[1:] == ["call"] * 3
    n_questions, calls_per_question, prompt_chars = fake.budget
    assert (n_questions, calls_per_question) == (3, 1)
    assert prompt_chars > 700  # the real Standard prompt, not a guess


def test_a_refused_budget_stops_the_run_before_any_call(harness, tmp_path):
    fake = harness(["Arthur's Magazine"], fits=False)
    with pytest.raises(SystemExit):
        run.main(["--task", "hotpotqa", "--condition", "cot", "--n", "3"])
    assert fake.events == ["budget"]
    assert not (tmp_path / "runs").exists()


def test_more_than_100_questions_needs_an_explicit_ask(harness, tmp_path):
    fake = harness(["Arthur's Magazine"])
    with pytest.raises(SystemExit):
        run.main(["--task", "hotpotqa", "--condition", "standard", "--n", "101"])
    assert fake.events == []  # rule 8's second trigger fires before the first one


def test_cotsc_is_budgeted_at_21_calls_a_question(harness, tmp_path):
    fake = harness(lambda prompt, n: [f"...\nAnswer: {p}" for p in SAMPLES_9_5_3_4EMPTY])
    run.main(["--task", "hotpotqa", "--condition", "cotsc", "--n", "1"])
    assert fake.budget[1] == 21


# -- the JSONL (rule 6, D25, D39, D41) -------------------------------------

def test_a_run_writes_one_line_per_question_in_evaluation_order(harness, tmp_path):
    harness(["Arthur's Magazine"])
    run.main(["--task", "hotpotqa", "--condition", "standard", "--n", "5"])
    lines = lines_of(tmp_path, "hotpotqa_standard_gpt-4o-mini.jsonl")
    assert [l["idx"] for l in lines] == [3687, 6238, 5388, 3522, 3824]
    assert [l["question"] for l in lines] == [
        q for q, _, _ in [data.load_hotpotqa()[i] for i in [3687, 6238, 5388, 3522, 3824]]
    ]


def test_every_line_carries_the_whole_schema_including_f1(harness, tmp_path):
    harness(["Arthur's Magazine"])
    run.main(["--task", "hotpotqa", "--condition", "standard", "--n", "2"])
    for line in lines_of(tmp_path, "hotpotqa_standard_gpt-4o-mini.jsonl"):
        assert set(line) == {
            "idx", "question", "gold", "prediction", "em", "f1", "n_steps", "n_calls",
            "n_badcalls", "hit_step_limit", "empty_samples", "winner_votes", "cost",
            "condition", "trajectory",
        }
        assert line["n_steps"] == 0 and line["hit_step_limit"] is False  # D15
        assert line["winner_votes"] is None and line["empty_samples"] is None
        assert line["trajectory"].startswith("Question: ")  # D25
        # This question's own spend, not the ledger total: the harness seeded $0.0031 of
        # prior spend, and no call here is billed, so the delta is 0.0 (rule 12).
        assert line["cost"] == 0.0
    assert llm._spend_so_far() == 0.0031


def test_f1_is_logged_because_em_rejects_answers_that_are_not_wrong(harness):
    """D41, on question 5388 (gold `torpedoes`) — with D41's own example CORRECTED.

    SQuAD normalization does not stem, so `torpedo boats and submarines` shares no
    token with `torpedoes` and scores F1 0.0, not "F1 > 0" as D41 states. The decision
    stands; the pair that demonstrates it on this question is the one below.
    """
    assert run.score("hotpotqa", "torpedo boats and submarines", "torpedoes") == (0, 0.0)
    em, f1 = run.score("hotpotqa", "torpedoes and submarines", "torpedoes")
    assert (em, f1) == (0, 0.5)  # EM rejects it; F1 says how close it was
    assert run.score("fever", "REFUTES", "REFUTES") == (1, 1.0)
    # rule 12: the gold is "maybe" too, so plain exact_match would score this 1. FEVER's
    # `em` column is §3's fever_score — a label whitelist — and nothing else separates the
    # two on a pair whose prediction and gold disagree.
    assert run.score("fever", "maybe", "maybe")[0] == 0  # not one of the 3 labels


def test_the_cotsc_line_records_the_winners_votes_never_the_sample_count(harness, tmp_path):
    harness(lambda prompt, n: [f"...\nAnswer: {p}" for p in SAMPLES_9_5_3_4EMPTY])
    run.main(["--task", "hotpotqa", "--condition", "cotsc", "--n", "1"])
    line = lines_of(tmp_path, "hotpotqa_cotsc_gpt-4o-mini.jsonl")[0]
    assert line["prediction"] == "Alpha"
    assert line["winner_votes"] == 9  # not 21, not 17 non-empty, not the first answer's 5
    assert line["empty_samples"] == 4
    assert line["n_badcalls"] == 4  # the empty samples, never the 21 that were issued
    assert line["n_calls"] == 21
    assert len(line["trajectory"]["samples"]) == 21


def test_fever_runs_score_labels_and_use_the_claim_as_the_question(harness, tmp_path):
    harness(["REFUTES"])
    run.main(["--task", "fever", "--condition", "standard", "--n", "2"])
    lines = lines_of(tmp_path, "fever_standard_gpt-4o-mini.jsonl")
    assert [l["idx"] for l in lines] == [3687, 6238]  # same sampler for both tasks
    assert all(l["gold"] in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO") for l in lines)
    assert all(l["em"] in (0, 1) for l in lines)


# -- the loop conditions (D15, D21, D24) -----------------------------------

def test_react_resets_the_env_once_per_question_and_logs_the_loop_index(harness, tmp_path):
    harness([FINISH])
    lines = run.main(["--task", "hotpotqa", "--condition", "react", "--n", "3"])
    assert [l["n_steps"] for l in lines] == [1, 1, 1]  # the loop index, not env.steps
    assert [l["hit_step_limit"] for l in lines] == [False] * 3
    assert all(l["trajectory"].endswith(
        "Action 1: Finish[Arthur's Magazine]\nObservation 1: Episode finished, reward = 0\n\n")
        for l in lines)
    assert all(l["prediction"] == "Arthur's Magazine" for l in lines)


def test_a_step_limited_episode_logs_seven_steps_and_an_empty_prediction(harness, tmp_path):
    harness(SEARCH[:7] * 2)
    lines = run.main(["--task", "hotpotqa", "--condition", "react", "--n", "1"])
    assert lines[0]["n_steps"] == 7
    assert lines[0]["hit_step_limit"] is True
    assert lines[0]["prediction"] == ""
    assert lines[0]["em"] == 0
    assert lines[0]["trajectory"].endswith("Observation 7: Arthur's Magazine is a magazine.\n")


def test_one_env_serves_the_whole_run_and_is_reset_per_question(harness, monkeypatch):
    envs = []
    monkeypatch.setattr(run.wiki_env, "WikiEnv", lambda: envs.append(FakeEnv()) or envs[-1])
    harness([FINISH])
    run.main(["--task", "hotpotqa", "--condition", "act", "--n", "4"])
    assert len(envs) == 1  # D21: one env for the run...
    assert envs[0].resets == 4  # ...reset once per question, by the runner


# -- results.csv (rule 6, D15) ---------------------------------------------

def test_the_summary_row_carries_the_model_string(harness, tmp_path):
    # One scripted answer PER QUESTION, against the five real golds, chosen so that every
    # aggregate differs (rule 12): EM 1,0,1,0,0 -> mean 0.4, which is not max 1, not the
    # first row's 1, not the last row's 0 and not the sum 2 — and mean F1 is 0.6333, so
    # the metric column cannot silently become the F1 it would be systematically higher.
    harness([
        "Beyond the Clouds",  # 3687 gold "Beyond the Clouds"  -> em 1, f1 1.0
        "Seth",               # 6238 gold "Seth MacFarlane"    -> em 0, f1 0.667
        "torpedoes",          # 5388 gold "torpedoes"          -> em 1, f1 1.0
        "Michael Jordan",     # 3522 gold "James Worthy"       -> em 0, f1 0.0
        "Hillary Clinton",    # 3824 gold "Bill Clinton"       -> em 0, f1 0.5
    ])
    run.main(["--task", "hotpotqa", "--condition", "standard", "--n", "5"])
    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert list(rows[0]) == combine.RESULTS_HEADER
    assert [rows[0][k] for k in ("task", "condition", "model", "n")] == \
        ["hotpotqa", "standard", "gpt-4o-mini", "5"]
    assert (rows[0]["mean_steps"], rows[0]["pct_hit_step_limit"]) == ("", "")  # D15
    lines = [json.loads(l) for l in
             (tmp_path / "runs" / "hotpotqa_standard_gpt-4o-mini.jsonl").read_text().splitlines()]
    assert [l["em"] for l in lines] == [1, 0, 1, 0, 0]
    mean_em = sum(l["em"] for l in lines) / len(lines)
    mean_f1 = sum(l["f1"] for l in lines) / len(lines)
    assert mean_f1 != mean_em  # the fixture separates the two columns, unconditionally
    assert float(rows[0]["metric"]) == pytest.approx(mean_em) == pytest.approx(0.4)
    assert float(rows[0]["metric"]) != pytest.approx(mean_f1)


def test_a_react_summary_row_reports_steps_and_the_step_limit_rate(harness, tmp_path):
    harness(RETRY_THEN_LIMIT)
    lines = run.main(["--task", "hotpotqa", "--condition", "react", "--n", "2"])
    # Question 1 retried once (3 calls for 2 steps); question 2 ran out of steps.
    assert [(l["n_steps"], l["n_calls"]) for l in lines] == [(2, 3), (7, 7)]
    assert [l["hit_step_limit"] for l in lines] == [False, True]
    row = list(csv.DictReader((tmp_path / "results.csv").open()))[0]
    assert row["mean_steps"] == "4.5"  # not max 7, first 2, last 7, sum 9, mean calls 5
    assert row["pct_hit_step_limit"] == "50.0"  # a PERCENTAGE: not 0.5, not the count 1


# -- the seam between the runner and src/combine.py ------------------------

def test_runner_output_feeds_the_combination_rules_unchanged(harness, tmp_path):
    """The two modules share a schema, and nothing else checks that they agree: a
    runner that wrote `votes` instead of `winner_votes` would KeyError in step 07,
    after the 500-question runs were paid for."""
    harness(SEARCH[:7] * 2)  # every question exhausts the 7 steps
    run.main(["--task", "hotpotqa", "--condition", "react", "--n", "2"])
    harness(lambda prompt, n: [f"...\nAnswer: {p}" for p in SAMPLES_9_5_3_4EMPTY])
    run.main(["--task", "hotpotqa", "--condition", "cotsc", "--n", "2"])

    react = tmp_path / "runs" / "hotpotqa_react_gpt-4o-mini.jsonl"
    cotsc = tmp_path / "runs" / "hotpotqa_cotsc_gpt-4o-mini.jsonl"
    a = combine.react_to_cotsc(react, cotsc, tmp_path / "a.jsonl")
    b = combine.cotsc_to_react(react, cotsc, tmp_path / "b.jsonl")
    # ReAct hit the limit on both -> A backs off; CoT-SC won with 9 of 21 -> B backs off.
    assert [l["source"] for l in a] == ["cotsc", "cotsc"]
    assert [l["prediction"] for l in a] == ["Alpha", "Alpha"]
    assert [l["source"] for l in b] == ["react", "react"]
    assert [l["n_steps"] for l in b] == [7, 7]  # inherited from ReAct (D15)
    assert [l["cost"] for l in a + b] == [0.0] * 4
