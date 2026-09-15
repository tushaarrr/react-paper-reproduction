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
    # One earlier run's row is on it — 20,000 + 500 tokens, $0.0033 at the section-5
    # rates — and every line written below must still cost 0.0.
    ledger = tmp_path / "calls.csv"
    ledger.write_text(
        ",".join(llm.CALLS_HEADER) + "\n"
        "2026-09-15T00:00:00,gpt-4o-mini,0,20000,500,0.0033,False\n"
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
        # This question's own spend, not the ledger total: the harness seeded $0.0033 of
        # prior spend, and no call here is billed, so the delta is 0.0 (rule 12).
        assert line["cost"] == 0.0
    assert llm._spend_so_far() == 0.0033


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


# -- resumability (rule 6; a 500-question run is long) ----------------------
#
# Rule 12 — the wrong implementations these three tests separate:
#   * a resume that re-runs questions already on disk  -> 15 JSONL rows, 15 ledger rows
#   * a resume that appends duplicate lines            -> 10 idx values but 15 rows
#   * a resume that appends a SECOND summary row       -> 2 rows, total_cost 0.0042
#   * a rewrite instead of an append                   -> the first 5 lines change
#   * a budget pre-flight sized on --n, not the gap    -> budget[0] == 10, not 5

def ledgered(monkeypatch):
    """Make the fake client append a real results/calls.csv row per request (1,000 +
    100 tokens = $0.00021 on gpt-4o-mini), so these tests can watch rule 8's ledger the
    way rule 8 does. Without it every cost here is 0.0 and a double-count is invisible."""
    inner = llm.complete

    def logged(*a, **k):
        out = inner(*a, **k)
        llm._log_call({"prompt_tokens": 1000, "completion_tokens": 100}, 0)
        return out

    monkeypatch.setattr(llm, "complete", logged)


def ledger_rows(tmp_path):
    """The calls.csv rows THIS run wrote. The header and the harness's seeded row (one
    earlier run's $0.0033, there so a cumulative-vs-delta cost bug cannot hide) are the
    first two lines and belong to neither invocation under test."""
    return (tmp_path / "calls.csv").read_text().splitlines()[2:]


def killing_after(k):
    """A fake that answers, then dies on its k-th request — a run killed mid-flight."""
    box = {"i": 0}

    def texts(prompt, n):
        box["i"] += 1
        if box["i"] == k:
            raise RuntimeError("process killed mid-run")
        return ["Arthur's Magazine"] * n
    return texts


def test_a_killed_run_resumes_from_the_jsonl_without_double_counting(harness, tmp_path,
                                                                    monkeypatch):
    fake = harness(killing_after(6))
    ledgered(monkeypatch)
    with pytest.raises(RuntimeError):
        run.main(["--task", "hotpotqa", "--condition", "standard", "--n", "10"])

    # The five questions that finished are on disk — the whole point of appending per
    # question rather than collecting the run and writing at the end.
    first_five = (tmp_path / "runs" / "hotpotqa_standard_gpt-4o-mini.jsonl").read_text()
    assert len(first_five.splitlines()) == 5
    assert len(ledger_rows(tmp_path)) == 5          # 6th request raised before billing
    assert not (tmp_path / "results.csv").exists()  # a killed run summarises nothing

    fake = harness(killing_after(None))
    ledgered(monkeypatch)
    run.main(["--task", "hotpotqa", "--condition", "standard", "--n", "10"])

    lines = lines_of(tmp_path, "hotpotqa_standard_gpt-4o-mini.jsonl")
    assert len(lines) == 10
    assert len({l["idx"] for l in lines}) == 10  # no duplicate question, in any order
    assert [l["idx"] for l in lines[:5]] == [3687, 6238, 5388, 3522, 3824]
    # Appended, not rewritten: the five surviving lines are byte-identical.
    first_five_now = "\n".join(
        (tmp_path / "runs" / "hotpotqa_standard_gpt-4o-mini.jsonl")
        .read_text().splitlines()[:5]) + "\n"
    assert first_five_now == first_five

    # The resume issued 5 requests, not 10: the ledger grew by 5 rows and the budget
    # pre-flight was sized on the GAP, so a resumed run cannot be refused for spend it
    # is not about to make.
    assert len(fake.calls) == 5
    assert fake.budget[0] == 5
    assert len(ledger_rows(tmp_path)) == 10
    assert llm._spend_so_far() == pytest.approx(0.0033 + 10 * 0.00021)
    assert sum(l["cost"] for l in lines) == pytest.approx(10 * 0.00021)

    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert len(rows) == 1  # one row for the condition, not one per invocation
    assert rows[0]["n"] == "10"
    assert float(rows[0]["total_cost"]) == pytest.approx(10 * 0.00021)


def test_re_running_a_finished_run_costs_nothing_and_leaves_one_summary_row(
        harness, tmp_path, monkeypatch):
    harness(killing_after(None))
    ledgered(monkeypatch)
    run.main(["--task", "hotpotqa", "--condition", "standard", "--n", "10"])
    jsonl = (tmp_path / "runs" / "hotpotqa_standard_gpt-4o-mini.jsonl").read_text()
    ledger = ledger_rows(tmp_path)

    fake = harness(killing_after(None))
    ledgered(monkeypatch)
    run.main(["--task", "hotpotqa", "--condition", "standard", "--n", "10"])

    assert fake.calls == []  # every question was already answered
    assert (tmp_path / "runs" / "hotpotqa_standard_gpt-4o-mini.jsonl").read_text() == jsonl
    assert ledger_rows(tmp_path) == ledger  # rule 8's ledger did not double-count
    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert [r["condition"] for r in rows] == ["standard"]
    assert float(rows[0]["total_cost"]) == pytest.approx(10 * 0.00021)  # not 0.0042


def test_a_smaller_n_on_a_resume_deletes_nothing_and_does_not_shrink_the_summary(
        harness, tmp_path, monkeypatch):
    """The documented answer to "--n differs from the original" (src/run.py docstring):
    --n selects a prefix, a resume only ever ADDS, and the summary row describes the
    whole file — so its n is the file's line count and never contradicts the file."""
    harness(killing_after(None))
    ledgered(monkeypatch)
    run.main(["--task", "hotpotqa", "--condition", "standard", "--n", "3"])
    assert list(csv.DictReader((tmp_path / "results.csv").open()))[0]["n"] == "3"

    fake = harness(killing_after(None))  # grow: 3 done, 5 asked for -> 2 new
    ledgered(monkeypatch)
    run.main(["--task", "hotpotqa", "--condition", "standard", "--n", "5"])
    assert len(fake.calls) == 2
    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert [r["n"] for r in rows] == ["5"]

    fake = harness(killing_after(None))  # shrink: nothing run, nothing deleted
    ledgered(monkeypatch)
    run.main(["--task", "hotpotqa", "--condition", "standard", "--n", "2"])
    assert fake.calls == []
    assert len(lines_of(tmp_path, "hotpotqa_standard_gpt-4o-mini.jsonl")) == 5
    rows = list(csv.DictReader((tmp_path / "results.csv").open()))
    assert [r["n"] for r in rows] == ["5"]  # not 2, and still exactly one row


# -- resuming a run that is NOT a clean prefix ------------------------------
#
# Rule 12 — the wrong implementations the four tests below separate, none of which any
# test above can see because every fixture there is a prefix of the evaluation order,
# is HotpotQA (whose questions are all distinct) and is `--condition standard` (the
# branch with no env at all):
#   * a resume that skips POSITIONALLY (items[len(done):])  -> 10 rows, 7 distinct idx
#   * a resume that matches on question TEXT, not idx       -> FEVER tops out at 237
#   * rule 8's >100 gate read off the REMAINING work        -> 101 runs unauthorised
#   * an env lifecycle that counts items instead of todo    -> resets 5, not 2

def seed_jsonl(tmp_path, name, condition, items):
    """Write finished-looking lines for `items` — a JSONL left by an earlier run. Only
    `idx` (the resume) and `em`/`n_steps`/`hit_step_limit`/`cost` (the summary row) are
    read back, but the whole schema is written so nothing here can pass by omission."""
    path = tmp_path / "runs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({
        "idx": i, "question": q, "gold": g, "prediction": g, "em": 1, "f1": 1.0,
        "n_steps": 0, "n_calls": 1, "n_badcalls": 0, "hit_step_limit": False,
        "empty_samples": None, "winner_votes": None, "cost": 0.0,
        "condition": condition, "trajectory": f"Question: {q}\n",
    }) + "\n" for i, q, g in items))
    return path


def test_a_resume_fills_the_gaps_by_idx_never_by_position(harness, tmp_path):
    """The file on disk is NOT a prefix of the evaluation order — two shards merged, or
    three lines lost from the middle of a killed run. `items[len(done):]` issues the
    same FOUR requests and leaves the same TEN rows, so a row count cannot see it: it
    re-runs (and re-bills) idx 2544/1557/5762 and never asks 3522/3824/2866, leaving
    results.csv reporting n=10 over 7 questions, three of them counted twice."""
    items = run.load("hotpotqa", "dev")[:10]
    assert [i for i, _, _ in items] == [3687, 6238, 5388, 3522, 3824,
                                        2866, 1551, 2544, 1557, 5762]
    seed_jsonl(tmp_path, "hotpotqa_standard_gpt-4o-mini.jsonl", "standard",
               [items[i] for i in (0, 1, 2, 7, 8, 9)])

    fake = harness(["Arthur's Magazine"])
    run.main(["--task", "hotpotqa", "--condition", "standard", "--n", "10"])

    lines = lines_of(tmp_path, "hotpotqa_standard_gpt-4o-mini.jsonl")
    assert len(fake.calls) == 4
    assert fake.budget[0] == 4                       # the gap, not the 6 already done
    assert [l["idx"] for l in lines[6:]] == [3522, 3824, 2866, 1551]  # exactly the gap
    assert len(lines) == 10 and len({l["idx"] for l in lines}) == 10
    assert sorted(l["idx"] for l in lines) == sorted(i for i, _, _ in items)


def test_a_fever_resume_skips_by_idx_not_by_the_claim_text(harness, tmp_path):
    """Invisible on HotpotQA — all 7,405 dev questions are distinct — so every test
    above passes against a resume keyed on question text. FEVER's 500-item sample
    contains the same claim twice under two different idx, and a text-keyed resume then
    tops out at 237 lines for --n 238: results.csv reports the wrong n, and any paired
    comparison against a condition that was NOT resumed dies in stats._em_by_idx."""
    items = run.load("fever", "dev")[:238]
    seeded, dup = items[114], items[237]
    assert seeded[1] == dup[1] == "AMGTV has programming."  # same claim...
    assert (seeded[0], dup[0]) == (5274, 5199)              # ...two different idx
    seed_jsonl(tmp_path, "fever_standard_gpt-4o-mini.jsonl", "standard", [seeded])

    fake = harness(["SUPPORTS"])
    run.main(["--task", "fever", "--condition", "standard", "--n", "238",
              "--yes-over-100"])

    lines = lines_of(tmp_path, "fever_standard_gpt-4o-mini.jsonl")
    assert len(fake.calls) == 237          # everything except the one already on disk
    assert [l["idx"] for l in lines] == [5274] + [i for i, _, _ in items if i != 5274]
    assert {5274, 5199} <= {l["idx"] for l in lines}
    assert len(lines) == 238


def test_rule_8s_gate_reads_the_whole_run_not_what_is_left_of_it(harness, tmp_path):
    """CLAUDE.md rule 8 authorises a RUN, not a chunk. Reading the gate off `todo`
    defeats it by salami-slicing: --n 500 is refused on an empty file, then --n 150,
    --n 300, --n 500 each present fewer than 100 remaining and complete the same
    unauthorised 500-question run."""
    items = run.load("hotpotqa", "dev")[:100]
    seed_jsonl(tmp_path, "hotpotqa_standard_gpt-4o-mini.jsonl", "standard", items)

    fake = harness(["Arthur's Magazine"])
    with pytest.raises(SystemExit, match="rule 8"):
        run.main(["--task", "hotpotqa", "--condition", "standard", "--n", "101"])

    assert fake.events == []  # no call, and the budget pre-flight was never reached
    assert len(lines_of(tmp_path, "hotpotqa_standard_gpt-4o-mini.jsonl")) == 100
    assert not (tmp_path / "results.csv").exists()


def test_a_resumed_react_run_resets_the_env_once_per_REMAINING_question(
        harness, tmp_path, monkeypatch):
    """The three resume tests above are all --condition standard, the branch that never
    builds a WikiEnv or a graph — so D21's lifecycle across a resume boundary is
    otherwise unexercised, and the 500-question runs that feed the claims table are
    ReAct and Act."""
    envs = []
    monkeypatch.setattr(run.wiki_env, "WikiEnv", lambda: envs.append(FakeEnv()) or envs[-1])
    harness([FINISH])
    run.main(["--task", "hotpotqa", "--condition", "react", "--n", "3"])
    assert envs[-1].resets == 3

    fake = harness([FINISH])
    lines = run.main(["--task", "hotpotqa", "--condition", "react", "--n", "5"])

    assert len(envs) == 2        # D21: one env per invocation, never one per question
    assert envs[-1].resets == 2  # the two REMAINING questions, not all five
    assert len(fake.calls) == 2
    assert [l["idx"] for l in lines] == [3687, 6238, 5388, 3522, 3824]
    assert [l["n_steps"] for l in lines] == [1] * 5          # D15, across the boundary
    assert [l["hit_step_limit"] for l in lines] == [False] * 5
    row = list(csv.DictReader((tmp_path / "results.csv").open()))[0]
    assert (row["n"], row["mean_steps"], row["pct_hit_step_limit"]) == ("5", "1.0", "0.0")
