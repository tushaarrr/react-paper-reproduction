"""Standard, CoT and CoT-SC — the three no-tool conditions (C6 of paper/notes.md §6).

Specs: D2 (Standard), D3 (CoT), D4 + D13 + D39 (CoT-SC), **D37** (the answer parse
rule, which retracts D3's "LAST occurrence"), D38 (prompt construction, measured
against the exemplars), D1 (no instruction header on any of these three).

Three things that look like style and are not:

* **`parse_answer` splits on the literal `"Answer:"`, with no space, and keeps what
  follows the FIRST occurrence** (D37). Both FEVER baseline keys contain
  `Answer:REFUTES` with no space, so a `"Answer: "` split silently loses that answer
  shape; and if the stop list ever fails, the completion runs on into the *next*
  question's `Answer:`, which is what a LAST-occurrence split would return.
* **Standard does not go through `parse_answer`.** Its prompt already ends in
  `Answer:`, so the completion *is* the answer and D2 says `.strip()`. Parsing it for
  a second `Answer:` would return `""` for every well-formed completion.
* **CoT-SC issues one `complete(..., n=21)` call, never 21 `n=1` calls.** `src/llm.py`
  derives the cache key's `sample_index` from `range(n)`, so 21 separate `n=1` calls
  would all key on `sample_index=0`: one cache entry, one request, and 21 copies of
  the same sample voting unanimously. The distinctness is asserted in
  `tests/test_baselines.py`.
"""

from collections import Counter

from src import data, graph_react, llm

N_SAMPLES = 21  # paper §3.2. The D6 threshold is n/2 = 10.5, never n_valid/2 (D13).
SC_TEMPERATURE = 0.7  # CLAUDE.md rule 4: the only non-greedy condition.
MAX_TOKENS = 100

# Exact keys, never a prefix (notes.md §5 "Prompt keys, unused legacy").
KEYS = {
    ("hotpotqa", "standard"): ("prompts_naive.json", "webqa_simple6"),
    ("hotpotqa", "cot"): ("prompts_naive.json", "cotqa_simple6"),
    ("fever", "standard"): ("fever.json", "webqa_simple3"),
    ("fever", "cot"): ("fever.json", "cotqa_simple3"),
}


def _tail(question, task, condition):
    """The live-question tail: what D25 stores as the `standard`/`cot` trajectory."""
    seed = "Question" if task == "hotpotqa" else "Claim"
    lead = "Answer:" if condition == "standard" else "Thought:"
    return f"{seed}: {question}\n{lead}"


def build_prompt(question, task, condition):
    """D38. FEVER gets an extra leading `\\n` because `*_simple3` ends with a single
    `\\n` while its exemplars are `\\n\\n`-separated; HotpotQA must not get one."""
    file, key = KEYS[(task, "cot" if condition == "cotsc" else condition)]
    lead = "" if task == "hotpotqa" else "\n"
    # graph_react._exemplars is the repo's one verbatim exemplar loader (rule 2).
    return graph_react._exemplars(file, key) + lead + _tail(question, task, condition)


def _stop(task, condition):
    if condition == "standard":
        return ["\n"]
    return ["\nQuestion:"] if task == "hotpotqa" else ["\nClaim:"]


def parse_answer(text):
    """D37: the text after the FIRST literal `"Answer:"`, stripped; `""` if absent.

    No trailing-period strip: no exemplar answer ends in one, `normalize_answer`
    removes punctuation for scoring anyway, and the JSONL `prediction` must stay raw.
    """
    _, sep, tail = text.partition("Answer:")  # partition == first occurrence
    return tail.strip() if sep else ""


def _result(prediction, trajectory, n_calls, n_badcalls,
            winner_votes=None, empty_samples=None):
    return {
        "prediction": prediction, "trajectory": trajectory,
        "n_calls": n_calls, "n_badcalls": n_badcalls,
        "winner_votes": winner_votes, "empty_samples": empty_samples,
    }


def standard(question, task):
    """D2: one greedy call, prediction = the completion, stripped."""
    prompt = build_prompt(question, task, "standard")
    completion = llm.complete(
        prompt, stop=_stop(task, "standard"), temperature=0.0, max_tokens=MAX_TOKENS
    )[0]
    prediction = completion.strip()
    return _result(
        prediction, _tail(question, task, "standard") + completion,
        n_calls=1, n_badcalls=int(not prediction),
    )


def cot(question, task):
    """D3: ONE greedy call — the exemplars put `Thought:` and `Answer:` on
    consecutive lines, so a single continuation emits both."""
    prompt = build_prompt(question, task, "cot")
    completion = llm.complete(
        prompt, stop=_stop(task, "cot"), temperature=0.0, max_tokens=MAX_TOKENS
    )[0]
    prediction = parse_answer(completion)
    return _result(
        prediction, _tail(question, task, "cot") + completion,
        n_calls=1, n_badcalls=int(not prediction),
    )


def _tally(preds):
    """Votes over `normalize_answer`; empty (parse-failed) predictions never vote
    and `""` can therefore never win (D13)."""
    return Counter(data.normalize_answer(p) for p in preds if p)


def majority_vote(preds):
    """-> (winning RAW prediction, its vote count, number of empty samples).

    One scan settles both of D4's ordering rules at once: the first prediction whose
    normalized key holds the top count is (a) from the tied key whose first occurrence
    has the lowest sample index and (b) the lowest-index raw spelling of that key.
    The denominator is never returned because it is not measured — it is the paper's
    `n`, fixed at `N_SAMPLES` (D13); `empty_samples` is a diagnostic, not a divisor.
    """
    tally = _tally(preds)
    if not tally:
        return "", 0, len(preds)
    top = max(tally.values())
    winner = next(p for p in preds if p and tally[data.normalize_answer(p)] == top)
    return winner, top, sum(1 for p in preds if not p)


def cot_sc(question, task, n=N_SAMPLES):
    """D4/D39: the CoT prompt, n independent samples at temperature 0.7, majority vote.

    `llm.complete(n=n)` is n separate requests under `sample_index` 0..n-1 (see the
    module docstring), so the cache holds n entries and `calls.csv` n rows.
    """
    prompt = build_prompt(question, task, "cotsc")
    samples = llm.complete(
        prompt, stop=_stop(task, "cotsc"), temperature=SC_TEMPERATURE,
        max_tokens=MAX_TOKENS, n=n,
    )
    preds = [parse_answer(s) for s in samples]
    winner, votes, empty = majority_vote(preds)
    trajectory = {  # D25: rule 6 says *full*, and for CoT-SC that is all n samples.
        "winner": samples[preds.index(winner)] if winner else "",
        "samples": samples,
        "votes": dict(_tally(preds)),
    }
    # A sample with no `Answer:` is a bad call (D3), so n_badcalls == empty_samples.
    return _result(winner, trajectory, n_calls=n, n_badcalls=empty,
                   winner_votes=votes, empty_samples=empty)
