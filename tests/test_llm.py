"""C3 of paper/notes.md section 6.

Every test here is OFFLINE. No test constructs a real client (`llm._client` is
patched in every test that can reach it) and no test makes a network call; the cache
and results/calls.csv are redirected to tmp_path so the real data/ and results/ are
never touched.
"""

import csv
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import llm

# The C3 canned response, and what cutting at the first stop string must leave.
CANNED = "I need to search X.\nAction 1: Search[X]\nObservation 1: leak"
CUT = "I need to search X.\nAction 1: Search[X]"
STOP = ["\nObservation 1:"]
USAGE = {"prompt_tokens": 1000, "completion_tokens": 100}
COST = 0.00021  # 1000/1e6*0.15 + 100/1e6*0.60, notes.md section 5 price table
BIG = {"prompt_tokens": 1_000_000, "completion_tokens": 0}  # exactly $0.15 a call
LOCAL = "qwen2.5-3b-react-lora"  # D27's phase-3 student, priced 0.0/0.0 on purpose


class FakeClient:
    """Mimics ChatOpenAI.generate: LLMResult.generations[0][0].text + llm_output."""

    def __init__(self, texts=CANNED, usage=USAGE, errors=()):
        self.texts = texts  # one str reused, or a list consumed one per request
        self.usage = usage
        self.errors = list(errors)
        self.calls = []

    def generate(self, messages, stop=None, **kwargs):
        self.calls.append({"messages": messages[0], "stop": stop, **kwargs})
        if self.errors:
            raise self.errors.pop(0)
        text = self.texts
        if not isinstance(text, str):
            text = self.texts[len(self.calls) - 1]
        return SimpleNamespace(
            generations=[[SimpleNamespace(text=text)]],
            llm_output={"token_usage": self.usage},
        )


def install(monkeypatch, tmp_path, fake=None, max_spend=5.00, model="gpt-4o-mini"):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "CALLS_CSV", tmp_path / "calls.csv")
    monkeypatch.setattr(llm, "MODEL", model)
    monkeypatch.setattr(llm, "MAX_SPEND_USD", max_spend)
    fake = fake or FakeClient()
    monkeypatch.setattr(llm, "_client", lambda: fake)
    return fake


@pytest.fixture
def client(monkeypatch, tmp_path):
    return install(monkeypatch, tmp_path)


def rows():
    if not llm.CALLS_CSV.exists():
        return []
    with open(llm.CALLS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def records():
    return [json.loads(p.read_text()) for p in sorted(llm.CACHE_DIR.glob("*.json"))]


# -- chat adaptation and stop cutting --------------------------------------

def test_prompt_is_one_user_message_plus_the_fixed_system_message(client):
    llm.complete("Question: who?\nThought 1:", STOP)
    system, user = client.calls[0]["messages"]
    assert system.content == llm.SYSTEM_MESSAGE
    assert user.content == "Question: who?\nThought 1:"
    assert client.calls[0]["temperature"] == 0.0
    assert client.calls[0]["max_tokens"] == 100


def test_system_message_is_the_one_recorded_in_notes(client):
    # The wording is a deviation row in paper/notes.md; it is also in the cache key,
    # so a silent edit here would strand every cached completion.
    notes = (Path(__file__).resolve().parents[1] / "paper" / "notes.md").read_text()
    assert llm.SYSTEM_MESSAGE in notes


def test_stop_is_sent_to_the_api_and_the_canned_response_is_cut(client):
    assert llm.complete("p", STOP) == [CUT]
    assert client.calls[0]["stop"] == STOP  # enforced twice: API list...


def test_cut_is_at_the_first_of_two_stop_occurrences(monkeypatch, tmp_path):
    twice = CANNED + "\nObservation 1: and again"
    install(monkeypatch, tmp_path, FakeClient(texts=twice))
    assert llm.complete("p", STOP) == [CUT]  # ...and client-side, at the first hit


def test_cut_takes_the_earliest_of_several_stop_strings(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, FakeClient(texts="a\nQuestion: b\nObservation 1: c"))
    assert llm.complete("p", ["\nObservation 1:", "\nQuestion:"]) == ["a"]


def test_a_response_without_the_stop_string_is_returned_whole(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, FakeClient(texts=CUT))
    assert llm.complete("p", STOP) == [CUT]


def test_a_bare_string_stop_is_one_stop_string_not_one_per_character(client):
    # Iterating a str yields characters, so an unwrapped stop would cut every
    # completion at its first letter — EM 0, cached permanently, and the symptom
    # points at the graph rather than here.
    assert llm.complete("p", "\nObservation 1:") == [CUT]
    assert llm.complete("p", STOP) == [CUT]
    assert len(client.calls) == 1  # same call, so the list spelling is a cache hit
    assert client.calls[0]["stop"] == STOP


def test_the_completion_is_returned_byte_exact_never_stripped(monkeypatch, tmp_path):
    # D25 assembles trajectories as f"Thought {i}:{completion}" with no separator, so
    # the leading space is load-bearing; the trailing newline ends the Action line.
    raw = " I need to search X.\nAction 1: Search[X]\n\nObservation 1: leak"
    install(monkeypatch, tmp_path, FakeClient(texts=raw))
    assert llm.complete("p", STOP) == [" I need to search X.\nAction 1: Search[X]\n"]


def test_temperature_and_max_tokens_reach_the_api_on_every_sample(monkeypatch, tmp_path):
    # The cache key carrying them is not enough: if they are not forwarded, CoT-SC's
    # 21 samples are 21 greedy duplicates and its vote is meaningless.
    fake = install(monkeypatch, tmp_path)
    llm.complete("p", STOP, temperature=0.7, max_tokens=50, n=3)
    assert [c["temperature"] for c in fake.calls] == [0.7] * 3
    assert [c["max_tokens"] for c in fake.calls] == [50] * 3


# -- cache (D14, D28) ------------------------------------------------------

def test_second_identical_call_is_a_free_cache_hit(client):
    first = llm.complete("p", STOP)
    spent = llm._spend_so_far()
    assert llm.complete("p", STOP) == first
    assert len(client.calls) == 1  # zero network calls
    assert len(rows()) == 1  # zero added calls.csv rows
    assert llm._spend_so_far() == spent > 0  # zero added spend


def test_cache_hits_never_count_toward_the_budget(monkeypatch, tmp_path):
    # Ceiling set to exactly the spend after one real call: further real calls are
    # refused, cached ones must still be served, forever and for free.
    install(monkeypatch, tmp_path)
    llm.complete("p", STOP)
    monkeypatch.setattr(llm, "MAX_SPEND_USD", llm._spend_so_far())
    for _ in range(5):
        assert llm.complete("p", STOP) == [CUT]
    assert llm._spend_so_far() == COST
    assert len(rows()) == 1
    with pytest.raises(llm.BudgetExceeded):
        llm.complete("a different prompt", STOP)


def test_a_cache_entry_appears_only_once_it_is_complete(client, monkeypatch):
    # Written to a temp path and renamed, so an interrupted run leaves no truncated
    # JSON behind — that key would otherwise raise on every later run, forever.
    def interrupted(*args):
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(llm.os, "replace", interrupted)
    with pytest.raises(RuntimeError):
        llm.complete("p", STOP)
    assert records() == []  # no half-written entry under the cache path


def test_cache_record_keeps_both_raw_and_cut_text(client):
    llm.complete("p", STOP)
    (record,) = records()
    assert record["raw"] == CANNED  # 08-verify.md:10's evidence, no fresh call
    assert record["text"] == CUT


def test_temperature_is_part_of_the_key(monkeypatch, tmp_path):
    fake = install(monkeypatch, tmp_path, FakeClient(texts=["greedy", "sampled"]))
    assert llm.complete("p", STOP) == ["greedy"]
    assert llm.complete("p", STOP, temperature=0.7) == ["sampled"]
    assert len(fake.calls) == 2 and len(records()) == 2


@pytest.mark.parametrize(
    "kwargs",
    [{"prompt": "other"}, {"stop": ["\n"]}, {"max_tokens": 50}, {"temperature": 0.7}],
)
def test_every_key_field_changes_the_cache_entry(monkeypatch, tmp_path, kwargs):
    fake = install(monkeypatch, tmp_path)
    call = {"prompt": "p", "stop": STOP, "temperature": 0.0, "max_tokens": 100}
    llm.complete(**call)
    llm.complete(**{**call, **kwargs})
    assert len(fake.calls) == 2 and len(records()) == 2


def test_system_message_is_part_of_the_key(monkeypatch, tmp_path):
    # D28: one field more than 04-llm-client.md:4 lists.
    fake = install(monkeypatch, tmp_path)
    llm.complete("p", STOP)
    monkeypatch.setattr(llm, "SYSTEM_MESSAGE", "Continue the text. Differently.")
    llm.complete("p", STOP)
    assert len(fake.calls) == 2 and len(records()) == 2 and len(rows()) == 2
    llm.complete("p", STOP)
    assert len(fake.calls) == 2  # the repeat under the same system message is cached


def test_21_samples_are_21_distinct_cache_entries(monkeypatch, tmp_path):
    fake = install(monkeypatch, tmp_path, FakeClient(texts=[str(i) for i in range(21)]))
    assert llm.complete("p", STOP, temperature=0.7, n=21) == [str(i) for i in range(21)]
    assert len(fake.calls) == 21  # D4: 21 independent requests
    keys = [r["key"] for r in records()]
    assert sorted(k["sample_index"] for k in keys) == list(range(21))
    assert len({json.dumps({**k, "sample_index": 0}, sort_keys=True) for k in keys}) == 1
    assert [r["sample_index"] for r in rows()] == [str(i) for i in range(21)]
    fake.calls.clear()
    assert llm.complete("p", STOP, temperature=0.7, n=21) == [str(i) for i in range(21)]
    assert fake.calls == [] and len(rows()) == 21  # warm: 0 calls, 0 new rows


def test_model_is_part_of_the_key(monkeypatch, tmp_path):
    # notes.md D16: a model change must MISS the cache rather than be served the old
    # model's completions. Phase 3 evaluates a second model through this function.
    fake = install(monkeypatch, tmp_path)
    llm.complete("p", STOP)
    monkeypatch.setattr(llm, "MODEL", LOCAL)
    llm.complete("p", STOP)
    assert len(fake.calls) == 2 and len(records()) == 2 and len(rows()) == 2
    assert [r["model"] for r in rows()] == ["gpt-4o-mini", LOCAL]  # rule 6
    monkeypatch.setattr(llm, "MODEL", "gpt-4o-mini")
    llm.complete("p", STOP)
    assert len(fake.calls) == 2  # the repeat under the first model is served cached


def test_a_cached_empty_completion_is_still_a_hit(monkeypatch, tmp_path):
    # D13/D3: an empty sample is an expected outcome, not a missing one. If "" did not
    # count as cached, every rerun would re-buy it and C8's free rerun would be false.
    fake = install(monkeypatch, tmp_path, FakeClient(texts=""))
    assert llm.complete("p", STOP) == [""]
    assert llm.complete("p", STOP) == [""]
    assert len(fake.calls) == 1 and len(rows()) == 1


# -- cost and budget (rules 6 and 8) ---------------------------------------

def test_row_uses_the_api_usage_numbers_not_a_local_estimate(client):
    # The prompt is 1 character: any local estimate is nowhere near 1000 tokens.
    llm.complete("p", STOP)
    (row,) = rows()
    assert row["model"] == "gpt-4o-mini"
    assert (row["prompt_tokens"], row["completion_tokens"]) == ("1000", "100")
    assert float(row["cost_usd"]) == COST
    assert row["cache_hit"] == "False"
    assert row["timestamp"].startswith("20")


def test_an_unpriced_model_is_refused_before_the_first_call(monkeypatch, tmp_path):
    # Costing an unknown model 0.0 would zero the estimate, every logged cost and the
    # ledger, so rule 8's ceiling could never fire — a typo in .env would buy a
    # 25x-priced model at a logged $0.00. Absent is refused; free is explicit.
    fake = install(monkeypatch, tmp_path, model="gpt-4o")  # real, paid, not in PRICES
    with pytest.raises(llm.BudgetExceeded, match="no price for gpt-4o"):
        llm.complete("p", STOP)
    assert fake.calls == [] and rows() == [] and records() == []


def test_a_model_priced_zero_on_purpose_runs_and_costs_zero(monkeypatch, tmp_path, caplog):
    # notes.md section 5: the local backend's 0.0/0.0 is deliberate (D27), so phase 3
    # still writes calls.csv rows and a total_cost of 0.0.
    install(monkeypatch, tmp_path, model=LOCAL)
    llm.complete("p", STOP)
    (row,) = rows()
    assert float(row["cost_usd"]) == 0.0 and row["model"] == LOCAL
    assert "priced 0.0/0.0" in caplog.text


def test_spend_is_the_sum_of_every_row_not_the_largest(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, FakeClient(usage=BIG))
    for prompt in ("p1", "p2", "p3"):
        llm.complete(prompt, STOP)
    assert [float(r["cost_usd"]) for r in rows()] == [0.15] * 3
    assert llm._spend_so_far() == pytest.approx(0.45)  # not 0.15, the largest row


def test_the_ceiling_fires_on_the_call_that_would_cross_it(monkeypatch, tmp_path):
    fake = install(monkeypatch, tmp_path, FakeClient(usage=BIG), max_spend=0.25)
    llm.complete("p1", STOP)  # $0.15
    llm.complete("p2", STOP)  # $0.30 total — the ceiling is now behind us
    with pytest.raises(llm.BudgetExceeded, match=r"spent \$0.3000"):
        llm.complete("p3", STOP)
    assert len(fake.calls) == 2 and len(rows()) == 2


def test_the_budget_is_rechecked_before_every_sample(monkeypatch, tmp_path):
    # CoT-SC is the largest-cost condition (D4, ~21x CoT). One check sized for one
    # call, hoisted out of the loop, would overshoot the ceiling by up to 21x.
    fake = install(monkeypatch, tmp_path, FakeClient(usage=BIG), max_spend=0.30)
    with pytest.raises(llm.BudgetExceeded):
        llm.complete("p", STOP, temperature=0.7, n=5)
    assert len(fake.calls) == 2 and len(rows()) == 2


@pytest.mark.parametrize(
    "prompt, max_tokens, max_spend",
    [
        ("p", 100, 5e-5),         # output half alone: 100/1e6*0.60      = 6.0e-5
        ("p" * 100_000, 0, 3e-3),  # input half alone: 25,000/1e6*0.15    = 3.75e-3
        ("", 0, 3.9e-6),          # system message alone: 26.5/1e6*0.15  = 3.975e-6
    ],
)
def test_every_term_of_the_pre_call_estimate_is_counted(
    monkeypatch, tmp_path, prompt, max_tokens, max_spend
):
    # Each ceiling sits just under the term named in the comment and above the sum of
    # the others, so dropping that term stops the refusal from happening at all.
    fake = install(monkeypatch, tmp_path, max_spend=max_spend)
    with pytest.raises(llm.BudgetExceeded):
        llm.complete(prompt, STOP, max_tokens=max_tokens)
    assert fake.calls == [] and rows() == []


def test_budget_raises_before_the_call(monkeypatch, tmp_path):
    fake = install(monkeypatch, tmp_path, max_spend=0.0)
    with pytest.raises(llm.BudgetExceeded) as excinfo:
        llm.complete("p" * 400, STOP)
    assert fake.calls == [] and rows() == [] and records() == []
    message = str(excinfo.value)
    assert "spent $0.0000" in message  # spend so far
    assert "MAX_SPEND_USD=$0.00" in message  # the ceiling
    assert "max_tokens=100" in message and "'pppp" in message  # what was attempted


def test_budget_total_survives_a_process_restart(monkeypatch, tmp_path):
    # One call spends exactly $3.00 under a $5.00 ceiling.
    install(monkeypatch, tmp_path, FakeClient(usage={"prompt_tokens": 20_000_000,
                                                     "completion_tokens": 0}))
    llm.complete("p", STOP)
    assert llm._spend_so_far() == 3.0

    importlib.reload(llm)  # a fresh process: module globals start over
    fake = install(monkeypatch, tmp_path, max_spend=3.0)
    with pytest.raises(llm.BudgetExceeded) as excinfo:
        llm.complete("a different prompt", STOP)
    assert "spent $3.0000" in str(excinfo.value)  # read back off disk, not from RAM
    assert fake.calls == [] and len(rows()) == 1


# -- retries ---------------------------------------------------------------

def transient():
    return openai.APIConnectionError(request=httpx.Request("POST", "http://local"))


def test_a_retried_call_logs_one_row_and_charges_once(monkeypatch, tmp_path):
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    fake = install(monkeypatch, tmp_path, FakeClient(errors=[transient(), transient()]))
    assert llm.complete("p", STOP) == [CUT]
    assert len(fake.calls) == 3  # two failures then a success
    assert len(rows()) == 1 and llm._spend_so_far() == COST
    assert len(records()) == 1


def test_retries_are_capped_and_rate_limits_are_retryable(monkeypatch, tmp_path):
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    fake = install(monkeypatch, tmp_path, FakeClient(errors=[transient()] * 9))
    with pytest.raises(openai.APIConnectionError):
        llm.complete("p", STOP)
    assert len(fake.calls) == llm.MAX_ATTEMPTS == 4
    assert rows() == [] and records() == []  # a failed call logs and caches nothing
    assert openai.RateLimitError in llm.RETRYABLE
    assert openai.InternalServerError in llm.RETRYABLE
    assert issubclass(openai.APITimeoutError, llm.RETRYABLE)


def bad_request():
    return openai.BadRequestError(
        "context length exceeded",
        response=httpx.Response(400, request=httpx.Request("POST", "http://local")),
        body=None,
    )


def test_a_non_retryable_error_is_raised_on_the_first_attempt(monkeypatch, tmp_path):
    # A bad key, an unknown model id or a prompt over the context limit will not fix
    # itself: retrying it four times with backoff hides a loud failure behind ~10s of
    # fake flakiness, on every step of a 500-question run.
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    fake = install(monkeypatch, tmp_path, FakeClient(errors=[bad_request()] * 9))
    with pytest.raises(openai.BadRequestError):
        llm.complete("p", STOP)
    assert len(fake.calls) == 1
    assert rows() == [] and records() == []


def test_backoff_sleeps_between_retries_and_grows(monkeypatch, tmp_path):
    slept = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    monkeypatch.setattr(llm.random, "random", lambda: 0.5)  # pin the jitter
    install(monkeypatch, tmp_path, FakeClient(errors=[transient(), transient()]))
    assert llm.complete("p", STOP) == [CUT]
    assert slept == [1.0, 2.0]  # 2**attempt * (0.5 + jitter): a 429 gets ridden out


# -- what ships (the defaults and the header nothing else asserts) ---------

def test_the_calls_csv_header_is_the_one_recorded_in_notes():
    # Rule 10: step 07's cost gate and results.csv aggregation are written against
    # notes.md's header block, so the two must not drift apart.
    notes = (Path(__file__).resolve().parents[1] / "paper" / "notes.md").read_text()
    assert ",".join(llm.CALLS_HEADER) in notes


def test_the_shipped_defaults_are_the_ones_rule_8_relies_on(monkeypatch):
    # Every other test overrides MODEL and MAX_SPEND_USD, so the values actually in
    # force are asserted only here. load_dotenv is stubbed out because .env would
    # otherwise supply the value and hide a wrong code default.
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("MAX_SPEND_USD", raising=False)
    try:
        importlib.reload(llm)
        assert llm.MAX_SPEND_USD == 5.00  # CLAUDE.md rule 8's ceiling
        assert llm.MODEL == "gpt-4o-mini"  # notes.md section 5: ours, one for all
        assert llm.MODEL in llm.PRICES  # or the guard refuses every call
        assert llm.CACHE_DIR.parts[-2:] == ("cache", "llm")  # not the wiki store
    finally:
        importlib.reload(llm)


def test_the_real_client_carries_the_reference_decoding_params(monkeypatch):
    # OFFLINE: constructs the client and inspects the payload it WOULD send. Nothing
    # is transmitted, so the dummy key is never used.
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-nothing-is-sent")
    monkeypatch.setattr(llm, "MODEL", "gpt-4o-mini")
    payload = llm._client()._get_request_payload(
        [llm.SystemMessage("s"), llm.HumanMessage("p")],
        stop=STOP, temperature=0.0, max_tokens=100,
    )
    assert payload["model"] == "gpt-4o-mini"  # read at call time, never memoised
    assert payload["top_p"] == 1.0  # hotpotqa.ipynb:27-29
    assert payload["frequency_penalty"] == 0.0
    assert payload["presence_penalty"] == 0.0
    assert payload["temperature"] == 0.0
    assert payload["max_completion_tokens"] == 100  # langchain-openai 1.6.2's name
    assert payload["stop"] == STOP

    # Not memoised: a client cached under the old MODEL would keep answering as it
    # while _key() and _log_call() had already moved on to the new one.
    monkeypatch.setattr(llm, "MODEL", LOCAL)
    assert llm._client()._get_request_payload([llm.HumanMessage("p")])["model"] == LOCAL
