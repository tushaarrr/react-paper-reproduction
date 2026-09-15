"""C6 of paper/notes.md §6 — Standard, CoT and CoT-SC.

Every test here is OFFLINE. Two fake layers are used, on purpose:

* `FakeLLM` replaces `llm.complete` and RECORDS every outbound request (rule 11) —
  prompt, stop, temperature, max_tokens, n.
* `install()` goes one layer deeper and replaces `llm._request`, the function that
  actually talks to the API, with the real cache and budget code still in the path.
  Only there is `sample_index` observable, and "21 samples must be 21 DISTINCT
  sample_index values" is exactly the bug that a `complete`-level fake cannot see: 21
  separate `n=1` calls all key on `sample_index=0`, so one request is made and its
  single answer votes 21 times, unanimously.

No test constructs a client, touches the network, the real LLM cache or
results/calls.csv.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import baselines, data, graph_react, llm

Q = "Which magazine was started first Arthur's Magazine or First for Women?"
C = "Stranger Things is set in Bloomington, Indiana."
HEADER_OPENING = "Solve a question answering task"  # D1: never on a baseline prompt

# The 9/7/5 fixture. Every aggregate is a DISTINCT number, so no two of them can be
# confused by a passing test (rule 12): total 21, winner 9, runner-up 7, third 5, and
# the FIRST answer to appear ("Bravo", 7) is deliberately not the winner.
NINE_SEVEN_FIVE = ["Bravo"] * 7 + ["Alpha"] * 9 + ["Charlie"] * 5


class FakeLLM:
    """Records every outbound request (rule 11) and serves scripted completions."""

    def __init__(self, *completions):
        self.completions = list(completions)
        self.calls = []

    def complete(self, prompt, stop, temperature=None, max_tokens=None, n=1):
        self.calls.append({"prompt": prompt, "stop": stop, "temperature": temperature,
                           "max_tokens": max_tokens, "n": n})
        if len(self.completions) == 1 and n > 1:  # one script, n copies
            return self.completions * n
        return [self.completions.pop(0) for _ in range(n)]


@pytest.fixture
def fake(monkeypatch):
    def factory(*completions):
        f = FakeLLM(*completions)
        monkeypatch.setattr(llm, "complete", f.complete)
        return f
    return factory


def install(monkeypatch, tmp_path, texts):
    """The deeper fake: real `complete`, real cache, real budget, fake `_request`."""
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "CALLS_CSV", tmp_path / "calls.csv")
    monkeypatch.setattr(llm, "MODEL", "gpt-4o-mini")
    monkeypatch.setattr(llm, "MAX_SPEND_USD", 5.00)
    requests = []

    def _request(prompt, stop, temperature, max_tokens):
        requests.append({"prompt": prompt, "stop": stop, "temperature": temperature,
                         "max_tokens": max_tokens})
        text = texts[len(requests) - 1] if isinstance(texts, list) else texts
        return text, {"prompt_tokens": 500, "completion_tokens": 20}

    monkeypatch.setattr(llm, "_request", _request)
    return requests


def cache_keys():
    return [json.loads(p.read_text())["key"] for p in llm.CACHE_DIR.glob("*.json")]


# -- D38: prompt construction, measured against the exemplars ---------------

def test_hotpotqa_standard_prompt_is_the_exemplars_plus_question_answer(fake):
    f = fake("1,800 to 7,000 ft")
    baselines.standard(Q, "hotpotqa")
    prompt = f.calls[0]["prompt"]
    assert prompt.endswith(f"Question: {Q}\nAnswer:")
    assert prompt.startswith("Question: What is the elevation range")  # no header (D1)
    assert HEADER_OPENING not in prompt
    assert prompt.count("Question: ") == 7  # 6 exemplars + the live one


def test_hotpotqa_cot_prompt_ends_at_thought_and_carries_no_header(fake):
    f = fake("Let's think step by step. ...\nAnswer: Arthur's Magazine")
    baselines.cot(Q, "hotpotqa")
    prompt = f.calls[0]["prompt"]
    assert prompt.endswith(f"Question: {Q}\nThought:")
    assert HEADER_OPENING not in prompt
    assert prompt.count("Let's think step by step. ") == 6  # §4: 6/6, HotpotQA only


def test_fever_prompts_restore_the_blank_line_between_exemplars(fake):
    f = fake("REFUTES")
    baselines.standard(C, "fever")
    assert f.calls[0]["prompt"].endswith(f"\n\nClaim: {C}\nAnswer:")

    f = fake("Stranger Things is in Hawkins.\nAnswer:REFUTES")
    baselines.cot(C, "fever")
    prompt = f.calls[0]["prompt"]
    assert prompt.endswith(f"\n\nClaim: {C}\nThought:")
    assert "Let's think step by step" not in prompt  # §4: 0/3 on FEVER


@pytest.mark.parametrize("task,condition,stop", [
    ("hotpotqa", "standard", ["\n"]),
    ("fever", "standard", ["\n"]),
    ("hotpotqa", "cot", ["\nQuestion:"]),
    ("fever", "cot", ["\nClaim:"]),
])
def test_stop_strings_per_task_and_condition(fake, task, condition, stop):
    f = fake("Answer: x")
    getattr(baselines, condition)(Q if task == "hotpotqa" else C, task)
    assert f.calls[0]["stop"] == stop


# -- D37: the parse rule ---------------------------------------------------

@pytest.mark.parametrize("completion,expected", [
    (" SUPPORTS", "SUPPORTS"),        # the spelling WITH a space, 2/3 of fever.json
    ("REFUTES", "REFUTES"),           # `Answer:REFUTES` — the no-space spelling
    (" NOT ENOUGH INFO", "NOT ENOUGH INFO"),
])
def test_fever_exemplar_tails_parse_under_both_spellings(completion, expected):
    # The exact text a FEVER CoT completion carries, for both exemplar spellings.
    assert baselines.parse_answer(f" ...thought.\nAnswer:{completion}") == expected
    assert baselines.parse_answer(f"Answer:{completion}") == expected


def test_both_spellings_are_really_in_the_exemplars():
    """The reason D37 splits on "Answer:" and not "Answer: " — measured, not assumed."""
    fever = json.loads((Path(__file__).resolve().parents[1] / "prompts" / "fever.json")
                       .read_text(encoding="utf-8"))
    for key in ("webqa_simple3", "cotqa_simple3"):
        assert "Answer:REFUTES" in fever[key]
        assert fever[key].count("Answer: ") == 2  # the other two DO have the space


def test_every_hotpotqa_cot_exemplar_tail_parses_to_its_gold_answer():
    text = graph_react._exemplars("prompts_naive.json", "cotqa_simple6")
    golds = ["1,800 to 7,000 ft", "Richard Nixon", "The Saimaa Gesture",
             "director, screenwriter, actor", "Arthur's Magazine", "Yes"]
    blocks = text.split("Question: ")[1:]
    assert len(blocks) == 6
    for block, gold in zip(blocks, golds):
        assert baselines.parse_answer(block) == gold


def test_the_answer_is_taken_from_the_first_occurrence_not_the_last():
    """D37, which CORRECTS D3. If the stop list fails, the completion runs on into the
    next exemplar's question; LAST would return that other question's answer."""
    runaway = (" Arthur's Magazine\nQuestion: Who is Milhouse named after?\n"
               "Thought: ...\nAnswer: Richard Nixon")
    got = baselines.parse_answer(f" ...\nAnswer:{runaway}")
    assert got.startswith("Arthur's Magazine")
    assert got != "Richard Nixon"  # what a LAST-occurrence split returns


def test_a_completion_with_no_answer_marker_is_empty_and_counts_as_a_bad_call(fake):
    assert baselines.parse_answer("Let's think step by step. I have no idea") == ""
    fake("Let's think step by step. I have no idea")
    out = baselines.cot(Q, "hotpotqa")
    assert out["prediction"] == ""
    assert out["n_badcalls"] == 1


def test_standard_does_not_reparse_its_own_completion(fake):
    """The Standard prompt already ends in `Answer:`, so D2 strips (a parse_answer
    here would return "" for every well-formed completion)."""
    fake(" Arthur's Magazine")
    assert baselines.standard(Q, "hotpotqa")["prediction"] == "Arthur's Magazine"


# -- CLAUDE.md rule 4: the decoding parameters that actually go out --------

def test_standard_and_cot_are_greedy(fake):
    for condition in ("standard", "cot"):
        f = fake("Answer: x")
        getattr(baselines, condition)(Q, "hotpotqa")
        assert f.calls[0]["temperature"] == 0.0
        assert f.calls[0]["max_tokens"] == 100
        assert f.calls[0]["n"] == 1


def test_cot_sc_samples_21_times_at_temperature_0_7(fake):
    f = fake("...\nAnswer: Arthur's Magazine")
    baselines.cot_sc(Q, "hotpotqa")
    assert len(f.calls) == 1  # one complete(), which fans out into n requests
    assert f.calls[0]["temperature"] == 0.7
    assert f.calls[0]["n"] == 21
    assert f.calls[0]["max_tokens"] == 100


def test_cot_sc_issues_21_requests_under_21_distinct_sample_indices(monkeypatch, tmp_path):
    """Rule 11, at the layer where the bug lives: `sample_index` is what makes the 21
    samples 21 different samples. 21 `complete(n=1)` calls would collide on index 0."""
    texts = [f"...\nAnswer: A{i}" for i in range(21)]
    requests = install(monkeypatch, tmp_path, texts)
    out = baselines.cot_sc(Q, "hotpotqa")

    assert len(requests) == 21
    assert {r["temperature"] for r in requests} == {0.7}
    assert {r["max_tokens"] for r in requests} == {100}
    assert len({r["prompt"] for r in requests}) == 1  # same prompt, 21 samples
    keys = cache_keys()
    assert sorted(k["sample_index"] for k in keys) == list(range(21))
    assert len(keys) == 21
    assert out["n_calls"] == 21
    assert len(out["trajectory"]["samples"]) == 21


def test_cot_sc_is_free_on_a_warm_cache(monkeypatch, tmp_path):
    requests = install(monkeypatch, tmp_path, [f"...\nAnswer: A{i}" for i in range(21)])
    first = baselines.cot_sc(Q, "hotpotqa")
    second = baselines.cot_sc(Q, "hotpotqa")
    assert len(requests) == 21  # no second round of requests
    assert first == second


# -- D4 / D13: the vote ----------------------------------------------------

def test_majority_vote_returns_the_winner_and_the_winners_own_count():
    winner, votes, empty = baselines.majority_vote(NINE_SEVEN_FIVE)
    assert (winner, votes, empty) == ("Alpha", 9, 0)
    assert len(NINE_SEVEN_FIVE) == 21
    # Every aggregate this could be confused with is a different number (rule 12):
    assert votes != len(NINE_SEVEN_FIVE)      # not the sample total (mutation 6)
    assert votes != NINE_SEVEN_FIVE.count("Bravo")  # not the first answer's count
    assert votes != len(set(NINE_SEVEN_FIVE))   # not the number of distinct answers


def test_cot_sc_reports_the_winners_votes_not_the_sample_count(fake):
    fake(*[f"...\nAnswer: {p}" for p in NINE_SEVEN_FIVE])
    out = baselines.cot_sc(Q, "hotpotqa")
    assert out["prediction"] == "Alpha"
    assert out["winner_votes"] == 9
    assert out["empty_samples"] == 0
    assert out["trajectory"]["votes"] == {"bravo": 7, "alpha": 9, "charlie": 5}


def test_the_vote_is_over_normalized_answers_but_the_winner_stays_raw():
    preds = ["Richard Nixon.", "richard nixon", "Richard Nixon."]
    winner, votes, _ = baselines.majority_vote(preds)
    assert data.normalize_answer(winner) == "richard nixon"
    assert (winner, votes) == ("Richard Nixon.", 3)  # raw, lowest index of the key


@pytest.mark.parametrize("first,second", [("Y", "Z"), ("Z", "Y")])
def test_ties_go_to_the_lowest_first_occurrence_and_never_move(first, second):
    """The RULE, not a constant: swapping which tied answer appears first swaps the
    winner. 10/10/1, and the answer with 1 vote can never win."""
    preds = [first, second] * 10 + ["W"]
    assert len(preds) == 21
    assert {baselines.majority_vote(preds)[:2] for _ in range(100)} == {(first, 10)}


def test_empty_samples_do_not_vote_and_the_denominator_stays_21():
    preds = [""] * 11 + ["Richard Nixon"] * 10
    winner, votes, empty = baselines.majority_vote(preds)
    assert (winner, votes, empty) == ("Richard Nixon", 10, 11)
    # 10 non-empty samples agreed unanimously, yet the paper's n is still 21, so this
    # question is below the 10.5 threshold and backs off to ReAct (D13 -> D6).
    assert votes < baselines.N_SAMPLES / 2
    assert votes > (baselines.N_SAMPLES - empty) / 2  # what an n_valid denominator says


def test_all_samples_empty_gives_an_empty_prediction_and_zero_votes(fake):
    fake("no answer marker here")
    out = baselines.cot_sc(Q, "hotpotqa")
    assert (out["prediction"], out["winner_votes"], out["empty_samples"]) == ("", 0, 21)
    assert out["n_badcalls"] == 21
    assert data.exact_match(out["prediction"], "Arthur's Magazine") == 0


def test_the_trajectory_carries_all_21_raw_samples_and_the_winning_one(fake):
    fake(*[f"thought {i}.\nAnswer: {p}" for i, p in enumerate(NINE_SEVEN_FIVE)])
    traj = baselines.cot_sc(Q, "hotpotqa")["trajectory"]
    assert len(traj["samples"]) == 21
    assert traj["winner"] == "thought 7.\nAnswer: Alpha"  # the first Alpha, index 7
    assert sum(traj["votes"].values()) == 21


# -- N50/N51: the CoT-SC PROMPT itself was never asserted ---------------------

@pytest.mark.parametrize("task,seed", [("hotpotqa", "Question"), ("fever", "Claim")])
def test_cot_sc_is_built_on_the_cot_prompt_not_standard(fake, task, seed):
    """Nothing asserted what CoT-SC actually SENDS, only its temperature and n.
    So the lead could become `Answer:` or the key could fall back to the Standard
    exemplars, and CoT-SC would quietly become `Standard sampled at 0.7` while
    still returning 21 parseable answers and healthy-looking winner_votes — and
    both combination rows would inherit it through the vote counts."""
    q = Q if task == "hotpotqa" else C
    f = fake(*(["Thought x. Answer: yes"] * 21))   # `fake` is a FACTORY, not the instance
    baselines.cot_sc(q, task)
    sent = f.calls[0]["prompt"]
    assert sent == baselines.build_prompt(q, task, "cot")
    assert sent.endswith(f"{seed}: {q}\nThought:")          # never "\nAnswer:"
    assert sent != baselines.build_prompt(q, task, "standard")
    # the CoT exemplars really are in there: thoughts, not bare answers
    assert "Thought:" in baselines.build_prompt(q, task, "cot")
    assert "Thought:" not in baselines.build_prompt(q, task, "standard")
