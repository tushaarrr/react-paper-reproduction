"""ReAct and Act as an explicit LangGraph StateGraph (C4 and C5 of notes.md section 6).

The loop is hotpotqa.ipynb:85-115 (FEVER.ipynb:78-108 is the same function without the
instruction header); the state is the 11 fields of 05-react-graph.md:3; the step limit is
a `force_finish` NODE with a pure router, not a state-writing conditional edge (D22).

Three things that look like style choices and are not:

* **Whitespace parity.** `.strip()` runs once, on the WHOLE completion, and the thought
  and the action are never stripped individually anywhere in this file. See notes.md
  section 1, "Whitespace parity in the scratchpad". The scratchpad IS the next step's
  prompt, so tidying it changes what the model sees.
* **The action travels in the state**, as the plain string field `action` (a `str`, so
  checkpointing is unaffected — unlike the env of D21). The reference never recovers it:
  it is a local for the rest of the loop iteration (hotpotqa.ipynb:95-102), and our
  two-node split is the only reason it has to travel at all. NEVER re-derive it from the
  scratchpad: an action whose own text contains `Action {i}: ` is truncated by `rsplit`
  and a thought that contains it is truncated by `split`, while the scratchpad stays
  byte-correct either way — so `runs/` looks perfect and a different Wikipedia action was
  executed. The thought travels in the scratchpad alone; `f"Thought {i}: {thought}\nAction
  {i}: {action}"` plus `execute`'s `f"\nObservation {i}: {obs}\n"` is byte-identical to
  the reference's single `step_str` (hotpotqa.ipynb:104).
* **The env is a closure argument** (D21), never an 11th state field: a live object in
  the state breaks checkpointing and contradicts the declared schema. `run_one` owns it
  and calls `reset()` once per question; no node here ever resets.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from src import llm

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
MAX_STEPS = 7  # hotpotqa.ipynb:91 `for i in range(1, 8)`, FEVER included (notes.md s2)

# hotpotqa.ipynb:77-82, character for character: 521 chars, line 1 ends "three types: "
# WITH the trailing space, the block ends "Here are some examples.\n". HotpotQA
# react/act only (D1) — FEVER's four prompt values carry their own header inline.
INSTRUCTION = (
    "Solve a question answering task with interleaving Thought, Action, Observation "
    "steps. Thought can reason about the current situation, and Action can be three "
    "types: \n"
    "(1) Search[entity], which searches the exact entity on Wikipedia and returns the "
    "first paragraph if it exists. If not, it will return some similar entities to "
    "search.\n"
    "(2) Lookup[keyword], which returns the next sentence containing keyword in the "
    "current passage.\n"
    "(3) Finish[answer], which returns the answer and finishes the task.\n"
    "Here are some examples.\n"
)

# Exact keys, never a prefix or a glob: `webthink_simple_3` (underscore, HotpotQA's
# unused legacy key) is one character from FEVER's `webthink_simple3` (notes.md s5).
KEYS = {
    ("hotpotqa", "react"): ("prompts_naive.json", "webthink_simple6"),
    ("hotpotqa", "act"): ("prompts_naive.json", "webact_simple6"),
    ("fever", "react"): ("fever.json", "webthink_simple3"),
    ("fever", "act"): ("fever.json", "webact_simple3"),
}


class ReActState(TypedDict):
    question: str
    task: str
    scratchpad: str
    action: str  # the 11th field: what execute hands the env, never re-parsed
    step: int
    done: bool
    answer: str
    n_calls: int
    n_badcalls: int
    hit_step_limit: bool
    condition: str


def initial_state(question, task, condition):
    """`step` starts at 1: the first continuation must read `Thought 1:`, and starting at
    0 silently buys an 8th model call and a `Thought 0:` prefix (notes.md section 1)."""
    return ReActState(
        question=question, task=task, scratchpad="", action="", step=1, done=False,
        answer="", n_calls=0, n_badcalls=0, hit_step_limit=False, condition=condition,
    )


@lru_cache(maxsize=None)
def _exemplars(file, key):
    """The authors' exemplars, verbatim (CLAUDE.md rule 2). Never rewritten, never
    re-whitespaced: their leading/trailing newlines differ per key by design (notes.md
    section 5, "Exemplar whitespace")."""
    return json.loads((PROMPTS / file).read_text(encoding="utf-8"))[key]


def prompt_prefix(task, condition, question):
    """instruction (HotpotQA only, D1) + exemplars + the live question + "\\n"
    (hotpotqa.ipynb:83, :89). Constant for a whole episode; the scratchpad is appended
    to it at every step, and the context is never truncated."""
    file, key = KEYS[(task, condition)]
    head = INSTRUCTION if task == "hotpotqa" else ""
    seed = "Question" if task == "hotpotqa" else "Claim"  # wrappers.py:97 / :166
    return f"{head}{_exemplars(file, key)}{seed}: {question}\n"


def _call(prompt, stop):
    """Rule 4 decoding, the section-5 table's react/act rows: temperature 0, 100 new
    tokens. `complete` returns a list; one request, so one element."""
    return llm.complete(prompt, stop=stop, temperature=0, max_tokens=100)[0]


# -- nodes -----------------------------------------------------------------

def think_act(state, condition):
    # `condition` comes from build_graph; the state field exists for the JSONL and must
    # agree, or a run is labelled as the condition it did not run.
    assert state["condition"] == condition, (state["condition"], condition)
    i = state["step"]
    prompt = prompt_prefix(state["task"], condition, state["question"])
    prompt += state["scratchpad"]
    stop = [f"\nObservation {i}:"]
    n_calls, n_badcalls = state["n_calls"] + 1, state["n_badcalls"]

    if condition == "act":
        # D5/D17: the action is the FIRST LINE of the stripped completion. A chat model
        # emits "Search[A]\nAction 2: Search[B]" past the stop string, and the env's
        # prefix/suffix tests would happily search the whole blob.
        action = _call(prompt + f"Action {i}:", stop).strip().split("\n")[0]
        if "[" not in action:  # empty or malformed; Act has no retry path, but it counts
            n_badcalls += 1
        block = f"Action {i}: {action}"
    else:
        thought_action = _call(prompt + f"Thought {i}:", stop)
        try:
            # hotpotqa.ipynb:95. STRICT two-way unpack on the WHOLE-string strip, and the
            # separator INCLUDES the trailing space. `split(sep, 1)` would not raise on a
            # two-separator completion and would execute "Search[A]\nAction 2: Search[B]".
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except ValueError:  # bare `except` at :96; both unpack errors are ValueError
            n_badcalls += 1  # :98 — only here, never on a good call
            n_calls += 1  # :99 — the retry is a second billed call
            thought = thought_action.strip().split("\n")[0]
            action = _call(f"{prompt}Thought {i}: {thought}\nAction {i}:", ["\n"]).strip()
        block = f"Thought {i}: {thought}\nAction {i}: {action}"

    # No .strip() on `thought` or `action` here, ever (notes.md section 1, parity).
    return {
        "scratchpad": state["scratchpad"] + block,
        "action": action,
        "n_calls": n_calls,
        "n_badcalls": n_badcalls,
    }


def execute(state, env):
    i = state["step"]
    action = state["action"]  # carried from think_act; re-parsing it is the R1 bug
    # hotpotqa.ipynb:102 lowercases the first character only, and raises IndexError on
    # "" — guarded, D9: an empty action reaches the env as "" and answers "Invalid action: ".
    obs, done, info = env.step(action[0].lower() + action[1:] if action else action)
    obs = obs.replace("\\n", "")  # :103 — the two-character sequence, not a newline
    update = {
        "scratchpad": state["scratchpad"] + f"\nObservation {i}: {obs}\n",  # :104
        "step": i + 1,
        "done": done,
    }
    if done:
        update["answer"] = info.get("answer") or ""
    return update


def force_finish(state, env):
    """hotpotqa.ipynb:110-111, as a node because a path function's writes are discarded
    (D22). It runs AFTER the last scratchpad append and appends nothing itself, so the
    trajectory ends at `Observation 7: <obs>\\n` (D24)."""
    env.step("finish[]")
    return {"hit_step_limit": True, "answer": "", "done": True}


def route_after_execute(state):
    """PURE. Anything a LangGraph path function writes to the state is discarded
    (verified on langgraph 1.2.11), which is why force_finish is a real node."""
    if state["done"]:
        return "end"
    return "force_finish" if state["step"] > MAX_STEPS else "think_act"


def build_graph(condition, env):
    assert condition in ("react", "act"), condition
    graph = StateGraph(ReActState)
    graph.add_node("think_act", lambda s: think_act(s, condition))
    graph.add_node("execute", lambda s: execute(s, env))
    graph.add_node("force_finish", lambda s: force_finish(s, env))
    graph.set_entry_point("think_act")
    graph.add_edge("think_act", "execute")
    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {"think_act": "think_act", "force_finish": "force_finish", "end": END},
    )
    graph.add_edge("force_finish", END)
    return graph.compile()
