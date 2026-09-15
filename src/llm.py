"""The LLM client: chat adaptation, disk cache, budget guard, cost log.

Spec: paper/notes.md section 5 (the per-condition decoding table, the price table
D16), the C3 row of section 6, D14 (the cache stores raw *and* cut text), D28 (the
system message is part of the cache key), prompts/claude-code/04-llm-client.md, and
CLAUDE.md rules 4, 5, 6 and 8.

The reference calls a raw completion endpoint with one prompt string
(hotpotqa.ipynb:22-32). We send that same completion-style prompt as a single user
message alongside the fixed SYSTEM_MESSAGE below — the "Chat-vs-completion
adaptation" and "System message" rows of notes.md's deviations table. The system
message is short, fixed and identical for every condition and both tasks.

Two decisions worth stating where they are made:

* **The pre-call cost estimate.** Rule 8 has to refuse a call *before* it is issued,
  when its real token counts do not exist yet. We estimate the input side at
  len(text) / 4 characters-per-token — no tokenizer dependency, and wrong only by a
  few percent on English prose — and the output side at the full `max_tokens`, which
  is its hard ceiling, so the estimate never runs low on the half we control. The
  estimate only ever gates; it is never logged. Every number in results/calls.csv
  comes from the response's own `usage` field (notes.md section 5, D16). A model
  absent from PRICES is **refused**, not costed at 0.0 — a guard that cannot price
  a call cannot enforce a ceiling on it.
* **n > 1 issues n independent requests**, not one request with the provider's native
  `n`. notes.md D4 and the section-5 decoding table both say "21 independent
  requests", C3 requires 21 calls.csv rows on a cold cache, and a row's
  `sample_index` column only means anything when one row is one sample. Cost is then
  ~21x the prompt tokens for CoT-SC, which is what D4's effect estimate already
  assumes.
"""

import csv
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache" / "llm"
CALLS_CSV = ROOT / "results" / "calls.csv"
CALLS_HEADER = [
    "timestamp", "model", "sample_index",
    "prompt_tokens", "completion_tokens", "cost_usd", "cache_hit",
]

MODEL = os.getenv("MODEL", "gpt-4o-mini")
MAX_SPEND_USD = float(os.getenv("MAX_SPEND_USD", "5.00"))

# notes.md section 5, D16: USD per 1M tokens, (input, output), rates of 2026-09-15.
# A model ABSENT from this table is refused before its first call (_check_budget):
# costing it 0.0 would zero the estimate, zero every logged cost and zero
# _spend_so_far(), i.e. switch CLAUDE.md rule 8 off for the one case — an
# unrecognised model string — where it is needed most. Free is explicit: phase 3's
# locally served student has no per-token price (notes.md section 5, D27).
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "qwen2.5-3b-prompted": (0.0, 0.0),
    "qwen2.5-3b-react-lora": (0.0, 0.0),
    "qwen2.5-3b-react-lora-q4": (0.0, 0.0),
}

# notes.md, "System message" row of the deviations table. Byte-identical there.
SYSTEM_MESSAGE = (
    "Continue the text exactly in the format of the examples. "
    "Do not write an Observation line; stop before it."
)

MAX_ATTEMPTS = 4
RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,  # openai.APITimeoutError subclasses this
    openai.InternalServerError,  # transient 5xx
)


class BudgetExceeded(RuntimeError):
    """CLAUDE.md rule 8: raised before the call that would cross MAX_SPEND_USD."""


def complete(prompt, stop, temperature=0.0, max_tokens=100, n=1):
    """-> list[str] of n completions, each cut at the first stop string.

    Decoding parameters are the section-5 table's: temperature 0 everywhere except
    CoT-SC's 0.7, max_tokens 100, top_p 1, frequency/presence penalty 0.
    """
    if isinstance(stop, str):
        stop = [stop]  # else it iterates into one stop string per character
    keys = [_key(prompt, stop, temperature, max_tokens, i) for i in range(n)]
    out = [_cached(k) for k in keys]
    for i, key in enumerate(keys):
        if out[i] is not None:
            continue  # cache hit: no request, no row, no spend
        _check_budget(prompt, max_tokens)
        raw, usage = _request(prompt, stop, temperature, max_tokens)
        _log_call(usage, i)
        out[i] = _cut(raw, stop)
        _write_cache(key, raw, out[i])
    return out


# -- cache (D14, D28) ------------------------------------------------------

def _key(prompt, stop, temperature, max_tokens, sample_index):
    """Rule 5's "exact input": the six fields of 04-llm-client.md:4 plus the system
    message (D28), which every request carries and which we expect to be tuned."""
    return {
        "model": MODEL,
        "system_message": SYSTEM_MESSAGE,
        "prompt": prompt,
        "stop": list(stop),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "sample_index": sample_index,
    }


def _cache_path(key):
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return CACHE_DIR / (digest + ".json")


def _cached(key):
    path = _cache_path(key)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["text"]


def _write_cache(key, raw, text):
    """Both fields, D14: `raw` is 08-verify.md:10's evidence that cutting happened,
    and storing only `text` makes that evidence cost a fresh call."""
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"key": key, "raw": raw, "text": text}), encoding="utf-8"
    )
    os.replace(tmp, path)  # atomic: an interrupted run leaves no truncated entry


# -- the call --------------------------------------------------------------

def _client():
    """Built fresh per request, never memoised: _key() and _log_call() read MODEL at
    call time, so a client cached under the old MODEL would label every cache entry
    and calls.csv row with a model that did not answer. Construction is cheap."""
    return ChatOpenAI(  # hotpotqa.ipynb:27-29
        model=MODEL, top_p=1, frequency_penalty=0.0, presence_penalty=0.0
    )


def _request(prompt, stop, temperature, max_tokens):
    """One request, retried on rate limits and transient 5xx. The caller logs and
    caches only what this returns, so a retry can neither double-count spend nor
    write a second calls.csv row."""
    messages = [SystemMessage(SYSTEM_MESSAGE), HumanMessage(prompt)]
    for attempt in range(MAX_ATTEMPTS):
        try:
            result = _client().generate(
                [messages],
                stop=list(stop),  # enforced at the API too, then again client-side
                temperature=temperature,
                max_tokens=max_tokens,
            )
            break
        except RETRYABLE:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt * (0.5 + random.random()))
    usage = (result.llm_output or {}).get("token_usage") or {}
    return result.generations[0][0].text, usage


def _cut(text, stop):
    """First occurrence of any stop string, the stop text itself excluded
    (04-llm-client.md:3). Providers differ on honouring stop lists and on whether
    they include the stop text, so we cut regardless of what came back."""
    hits = [text.find(s) for s in stop if s and s in text]
    return text[:min(hits)] if hits else text


# -- cost, budget, log (rules 5, 6, 8) -------------------------------------

def _cost(prompt_tokens, completion_tokens):
    in_rate, out_rate = PRICES[MODEL]  # unpriced models are refused in _check_budget
    if (in_rate, out_rate) == (0.0, 0.0):
        logging.warning("llm: model %r is priced 0.0/0.0; logging cost 0.0", MODEL)
    return round(prompt_tokens / 1e6 * in_rate + completion_tokens / 1e6 * out_rate, 10)


def _spend_so_far():
    """Cumulative spend, read back from results/calls.csv — the ledger is the log, so
    the total survives a process restart and a resumed run, and cache hits (which
    write no row) cost nothing."""
    # ponytail: re-reads the whole CSV per real call; ~ms at 10k rows. Memoise
    # against the file's mtime if a run ever notices.
    if not CALLS_CSV.exists():
        return 0.0
    with open(CALLS_CSV, newline="") as f:
        return sum(float(row["cost_usd"]) for row in csv.DictReader(f))


def _check_budget(prompt, max_tokens):
    if MODEL not in PRICES:
        raise BudgetExceeded(
            f"no price for {MODEL}; add it to PRICES in src/llm.py "
            f"((0.0, 0.0) for a locally served model)"
        )
    spent = _spend_so_far()
    in_rate, out_rate = PRICES[MODEL]
    estimate = (len(SYSTEM_MESSAGE) + len(prompt)) / 4 / 1e6 * in_rate
    estimate += max_tokens / 1e6 * out_rate
    if spent + estimate > MAX_SPEND_USD:
        raise BudgetExceeded(
            f"spent ${spent:.4f} so far, ceiling MAX_SPEND_USD=${MAX_SPEND_USD:.2f} "
            f"(CLAUDE.md rule 8); refused a call estimated at ${estimate:.4f} "
            f"(model={MODEL}, max_tokens={max_tokens}, prompt {len(prompt)} chars "
            f"starting {prompt[:60]!r})"
        )


def _log_call(usage, sample_index):
    """One row per real request. Token counts are the response's own usage numbers,
    never a local estimate — a mis-estimate would silently mis-gate rule 8.
    `cache_hit` is always False: a hit issues no request and writes no row (C3)."""
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    CALLS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CALLS_CSV.exists()
    with open(CALLS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CALLS_HEADER)
        writer.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%S"), MODEL, sample_index,
            prompt_tokens, completion_tokens,
            _cost(prompt_tokens, completion_tokens), False,
        ])
