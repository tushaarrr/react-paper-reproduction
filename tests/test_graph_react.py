"""C4 and C5 of paper/notes.md section 6 — the ReAct and Act graphs.

Every test here is OFFLINE: `llm.complete` is replaced by a fake that RECORDS the
outbound request (CLAUDE.md rule 11) and the env is a fake that records the actions it
was handed. No test builds a client, and no test touches the network, the LLM cache or
results/calls.csv.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, graph_react

Q = "Which movie did Irene Jacob complete before the film directed by Stuart Bird?"
FIELDS = {
    "question", "task", "scratchpad", "action", "step", "done", "answer",
    "n_calls", "n_badcalls", "hit_step_limit", "condition",
}
OBS = "Richard Milhous Nixon was the 37th president."
FINISH_OBS = "Episode finished, reward = 0\n"

# THE PARITY LITERALS. `RAW` is the first live completion, byte for byte, keyboard-
# unmodified: note the space after "Bird." and the space after "Search[Irene Jacob]".
# `BLOCK` is what the reference driver (hotpotqa.ipynb:93-104) turns it into — one
# .strip() on the WHOLE completion, then a split on "\nAction 1: " INCLUDING the
# trailing space, and no individual strip of either half. So the thought KEEPS its
# trailing space and the action LOSES its own (the whole-string strip took it), and
# that surviving space goes into the scratchpad, i.e. into the next step's prompt.
RAW = (
    "I need to search for Irene Jacob and find out what movie she completed before "
    "the American action crime thriller film directed by Stuart Bird. \n"
    "Action 1: Search[Irene Jacob] "
)
BLOCK = (
    "Thought 1: I need to search for Irene Jacob and find out what movie she completed "
    "before the American action crime thriller film directed by Stuart Bird. \n"
    "Action 1: Search[Irene Jacob]\n"
    "Observation 1: <obs>\n"
)


class FakeLLM:
    """Records every outbound request (rule 11) and serves scripted completions."""

    def __init__(self, *completions):
        self.completions = list(completions)
        self.calls = []

    def complete(self, prompt, stop, temperature=None, max_tokens=None, n=1):
        self.calls.append({
            "prompt": prompt, "stop": stop,
            "temperature": temperature, "max_tokens": max_tokens,
        })
        return [self.completions.pop(0)]  # IndexError if a test under-scripts it


class FakeEnv:
    """src.wiki_env.WikiEnv.step's contract: the 3-tuple of D7 and its info dict."""

    def __init__(self, *observations):
        self.observations = list(observations)
        self.actions = []
        self.answer = None

    def step(self, action):
        self.actions.append(action)
        if action.startswith("finish[") and action.endswith("]"):
            self.answer = action[len("finish["):-1]
            return FINISH_OBS, True, {"steps": len(self.actions), "answer": self.answer}
        obs = self.observations.pop(0) if self.observations else OBS
        return obs, False, {"steps": len(self.actions), "answer": None}


def run(monkeypatch, completions, observations=(), condition="react",
        task="hotpotqa", question=Q):
    llm, env = FakeLLM(*completions), FakeEnv(*observations)
    monkeypatch.setattr(graph_react.llm, "complete", llm.complete)
    graph = graph_react.build_graph(condition, env)
    state = graph.invoke(graph_react.initial_state(question, task, condition))
    return state, llm, env


def prefix(condition="react", task="hotpotqa", question=Q):
    return graph_react.prompt_prefix(task, condition, question)


def assert_decoding(call, step):
    """Rule 11 + rule 4: every step call carries the section-5 table's parameters."""
    assert call["stop"] == [f"\nObservation {step}:"]
    assert call["temperature"] == 0
    assert call["max_tokens"] == 100


# -- C4(0): state and prompt shape -----------------------------------------

def test_initial_state_has_the_eleven_fields_and_starts_at_step_one():
    state = graph_react.initial_state(Q, "hotpotqa", "react")
    assert set(state) == FIELDS  # including `action` — corrections-log entry 8
    assert state["step"] == 1  # 0 buys an 8th call and a "Thought 0:" prefix
    assert (state["scratchpad"], state["answer"], state["action"]) == ("", "", "")
    assert (state["n_calls"], state["n_badcalls"]) == (0, 0)
    assert state["done"] is False and state["hit_step_limit"] is False


def test_prompt_prefix_whitespace_per_task_and_condition():
    """notes.md section 5: the instruction header is HotpotQA react/act only (D1), and
    the exemplar keys' leading/trailing newlines differ on purpose."""
    react = prefix("react")
    assert react.startswith(graph_react.INSTRUCTION)
    assert len(graph_react.INSTRUCTION) == 521  # hotpotqa.ipynb:77-82, byte for byte
    assert "three types: \n" in graph_react.INSTRUCTION  # the trailing space on line 1
    # webthink_simple6 starts with a bare \n and ends with a single \n: one blank line
    # before the FIRST exemplar, none before the live question.
    assert "Here are some examples.\n\nQuestion: What is the elevation range" in react
    assert react.endswith(f"\nAction 3: Finish[yes]\nQuestion: {Q}\n")

    # webact_simple6 is the mirror image: no leading \n, a \n\n tail.
    act = prefix("act")
    assert act.startswith(graph_react.INSTRUCTION + "Question: What is the elevation")
    assert act.endswith(f"\nAction 3: Finish[yes]\n\nQuestion: {Q}\n")

    for condition in ("react", "act"):  # FEVER carries its own header inline (D1)
        fever = prefix(condition, "fever", "Some claim.")
        assert graph_react.INSTRUCTION not in fever
        assert fever.startswith("\nDetermine if there is Observation that SUPPORTS")
        assert fever.endswith("\n\nClaim: Some claim.\n")
    # ...and the two FEVER values are NOT interchangeable: both heads and both tails are
    # identical, so only the Thought lines tell webthink_simple3 from webact_simple3. A
    # swapped pair would prompt "Thought i:" against thought-free exemplars for 500
    # questions with no other symptom (mirrors the C5 scratchpad assertion).
    assert "Thought" in prefix("react", "fever", "Some claim.")
    assert "Thought" not in prefix("act", "fever", "Some claim.")


# -- C4(a): the happy path -------------------------------------------------

def test_two_step_episode_answer_and_exact_scratchpad(monkeypatch):
    state, llm, env = run(
        monkeypatch,
        completions=[
            "I need to search Richard Nixon.\nAction 1: Search[Richard Nixon]",
            "The answer is Richard Nixon.\nAction 2: Finish[Richard Nixon]",
        ],
        observations=[OBS],
    )
    assert state["answer"] == "Richard Nixon"
    assert state["done"] is True and state["hit_step_limit"] is False
    assert (state["n_calls"], state["n_badcalls"]) == (2, 0)
    assert state["step"] == 3
    assert state["scratchpad"] == (
        "Thought 1: I need to search Richard Nixon.\n"
        "Action 1: Search[Richard Nixon]\n"
        f"Observation 1: {OBS}\n"
        "Thought 2: The answer is Richard Nixon.\n"
        "Action 2: Finish[Richard Nixon]\n"
        # The env's own trailing newline AND the step separator, as in the reference
        # (hotpotqa.ipynb:104 appends "\n" to an obs that already ends with one).
        f"Observation 2: {FINISH_OBS}\n"
    )
    # rule 11: what was SENT, not just what came back.
    assert len(llm.calls) == 2
    assert llm.calls[0]["prompt"] == prefix() + "Thought 1:"
    assert llm.calls[1]["prompt"] == prefix() + (
        "Thought 1: I need to search Richard Nixon.\n"
        "Action 1: Search[Richard Nixon]\n"
        f"Observation 1: {OBS}\n"
        "Thought 2:"
    )
    assert_decoding(llm.calls[0], 1)
    assert_decoding(llm.calls[1], 2)
    # The FIRST character only is lowercased (hotpotqa.ipynb:102).
    assert env.actions == ["search[Richard Nixon]", "finish[Richard Nixon]"]


# -- THE PARITY TEST -------------------------------------------------------

def test_parity_the_scratchpad_keeps_the_models_whitespace(monkeypatch):
    """The live raw completion in, the byte-exact scratchpad block out.

    Kills: stripping the thought individually (the space after "Bird." disappears) and
    splitting on "\\nAction 1:" without the trailing space (the action gains a leading
    space, the block reads "Action 1:  Search[...]" and the env is handed " Search[...]").
    """
    state, llm, env = run(
        monkeypatch,
        completions=[RAW, "So the answer is Beyond Silence.\nAction 2: Finish[Beyond Silence]"],
        observations=["<obs>"],
    )
    assert state["scratchpad"].startswith(BLOCK)
    assert state["scratchpad"][:len(BLOCK)] == BLOCK  # byte for byte
    assert "Stuart Bird. \nAction 1: Search[Irene Jacob]\n" in state["scratchpad"]
    # That surviving space is in the next step's prompt, which is the whole point.
    assert llm.calls[1]["prompt"] == prefix() + BLOCK + "Thought 2:"
    assert env.actions[0] == "search[Irene Jacob]"  # no leading space, no trailing space
    assert state["answer"] == "Beyond Silence"


# -- C4(b)(c)(f): the parse-failure path -----------------------------------

def test_parse_failure_retries_with_stop_newline(monkeypatch):
    """rule 12: THREE calls, n_calls 3, n_badcalls 1 — three distinct numbers, so
    'incremented on parse failure' cannot be confused with 'incremented every call'."""
    state, llm, env = run(
        monkeypatch,
        completions=[
            # Genuinely multi-line: the branch is reached precisely when the model
            # rambled, and `[0]` vs `[-1]` vs "no split at all" are three different
            # thoughts — in the scratchpad AND in the retry prompt (rule 11).
            "First line.\nSecond line.\nThird line.",  # zero "\nAction 1: " separators
            "  Search[Richard Nixon]  ",  # the retry answers with the action alone
            "The answer is Richard Nixon.\nAction 2: Finish[Richard Nixon]",
        ],
        observations=[OBS],
    )
    assert (state["n_calls"], state["n_badcalls"]) == (3, 1)
    assert len(llm.calls) == 3
    assert llm.calls[1]["prompt"] == prefix() + "Thought 1: First line.\nAction 1:"
    assert llm.calls[1]["stop"] == ["\n"]  # NOT ["\nObservation 1:"]
    assert llm.calls[1]["temperature"] == 0 and llm.calls[1]["max_tokens"] == 100
    assert_decoding(llm.calls[2], 2)  # the next step's stop is rebuilt, not reused
    assert state["scratchpad"].startswith(
        "Thought 1: First line.\n"
        "Action 1: Search[Richard Nixon]\n"  # hotpotqa.ipynb:101 strips the RETRY only
    )
    assert env.actions == ["search[Richard Nixon]", "finish[Richard Nixon]"]
    assert state["answer"] == "Richard Nixon"


def test_two_separators_also_count_as_a_bad_call(monkeypatch):
    """C4(c). The strict two-way unpack raises 'too many values'; split(sep, 1) or
    str.partition would not, and would search "A]\\nAction 1: Search[B"."""
    state, llm, env = run(
        monkeypatch,
        completions=[
            "I need X.\nAction 1: Search[A]\nAction 1: Search[B]",
            "Search[A]",
            "Found it.\nAction 2: Finish[A]",
        ],
        observations=[OBS],
    )
    assert (state["n_calls"], state["n_badcalls"]) == (3, 1)
    assert env.actions[0] == "search[A]"
    assert llm.calls[1]["stop"] == ["\n"]
    assert llm.calls[1]["prompt"] == prefix() + "Thought 1: I need X.\nAction 1:"
    assert state["scratchpad"].startswith("Thought 1: I need X.\nAction 1: Search[A]\n")


def test_empty_completion_reaches_the_env_and_the_episode_continues(monkeypatch):
    """C4(f)/D9: "" would raise IndexError on action[0] in the reference."""
    state, llm, env = run(
        monkeypatch,
        completions=["", "", "Now I know.\nAction 2: Finish[Richard Nixon]"],
        observations=["Invalid action: "],
    )
    assert env.actions == ["", "finish[Richard Nixon]"]
    assert (state["n_calls"], state["n_badcalls"]) == (3, 1)
    assert state["scratchpad"].startswith("Thought 1: \nAction 1: \nObservation 1: Invalid action: \n")
    assert state["done"] is True


# -- C4(d): the step limit -------------------------------------------------

def seven_steps(monkeypatch):
    return run(
        monkeypatch,
        completions=[
            f"Still looking {i}.\nAction {i}: Search[Entity {i}]" for i in range(1, 8)
        ],
    )


def test_step_limit_routes_through_force_finish(monkeypatch):
    state, llm, env = seven_steps(monkeypatch)
    # The FINAL state, not an intermediate one: a router that writes this instead of a
    # node leaves it False on langgraph 1.2.11 (D22).
    assert state["hit_step_limit"] is True
    assert state["answer"] == ""
    assert state["done"] is True
    assert state["step"] == 8  # execute incremented past MAX_STEPS, then the router ran
    assert (state["n_calls"], state["n_badcalls"]) == (7, 0)
    assert len(llm.calls) == 7  # exactly 7 model steps, never 8
    assert llm.calls[6]["prompt"].endswith(f"Observation 6: {OBS}\nThought 7:")
    assert env.actions == [f"search[Entity {i}]" for i in range(1, 8)] + ["finish[]"]
    assert data.exact_match(state["answer"], "Richard Nixon") == 0  # the episode scores 0


def test_step_limit_appends_nothing_after_observation_seven(monkeypatch):
    """D24: the forced finish[] runs AFTER the last scratchpad append, so its
    observation never enters the trajectory."""
    state, _, _ = seven_steps(monkeypatch)
    assert state["scratchpad"].endswith(f"\nObservation 7: {OBS}\n")
    assert not state["scratchpad"].endswith(FINISH_OBS)
    for absent in ("Thought 8:", "Action 8:", "Observation 8:", "Episode finished"):
        assert absent not in state["scratchpad"]


def test_route_after_execute_is_pure():
    """It returns a route and writes nothing — the reason force_finish is a node."""
    for state, expected in [
        ({"done": True, "step": 2}, "end"),
        ({"done": True, "step": 99}, "end"),  # done wins over the limit
        ({"done": False, "step": 7}, "think_act"),  # the 7th step still gets to run
        ({"done": False, "step": 8}, "force_finish"),
    ]:
        before = dict(state)
        assert graph_react.route_after_execute(state) == expected
        assert state == before


# -- C5: Act ---------------------------------------------------------------

def test_act_takes_only_the_first_line(monkeypatch):
    """D5/D17. The whole blob would reach the env as search[A]\\nAction 2: Search[B —
    a garbage entity that still passes the env's prefix and suffix tests.

    The scripted completions carry the LEADING SPACE a chat model puts after the
    "Action 1:" it is continuing, which is the overwhelmingly common real shape: without
    the `.strip()` the env is handed " Search[A]", wikienv strips it back to "Search[A]",
    the capital S fails the lowercase prefix test and EVERY Act step scores
    "Invalid action:" while n_badcalls stays 0.
    """
    state, llm, env = run(
        monkeypatch,
        completions=[" Search[A]\nAction 2: Search[B]", " Finish[Richard Nixon]"],
        observations=[OBS],
        condition="act",
    )
    assert env.actions == ["search[A]", "finish[Richard Nixon]"]
    assert state["scratchpad"] == (
        f"Action 1: Search[A]\nObservation 1: {OBS}\n"
        f"Action 2: Finish[Richard Nixon]\nObservation 2: {FINISH_OBS}\n"
    )
    assert "Thought" not in state["scratchpad"]
    assert llm.calls[0]["prompt"] == prefix("act") + "Action 1:"
    assert llm.calls[1]["prompt"] == prefix("act") + (
        f"Action 1: Search[A]\nObservation 1: {OBS}\nAction 2:"
    )
    assert_decoding(llm.calls[0], 1)
    assert_decoding(llm.calls[1], 2)
    assert (state["n_calls"], state["n_badcalls"]) == (2, 0)


def test_act_counts_bad_calls_without_retrying(monkeypatch):
    """rule 12 again: 5 calls, 2 bad, and the counted test ("[" is absent) exercised in
    three distinct shapes. Act has no retry path (D5), so a malformed first line goes to
    the env as-is — but it is still counted.

    The shapes matter because `"[" not in action` and `"]" not in action` are
    indistinguishable on a completion carrying neither: an action truncated by
    max_tokens=100 has "[" and no "]" and is a GOOD call per D17, while a stray thought
    ending in "]" is not an action at all. Two of the first and one of the second, so the
    two counts cannot coincide.
    """
    state, llm, env = run(
        monkeypatch,
        completions=[
            "Search[Truncated entity",   # "[" only — max_tokens cut it: a GOOD call
            "Lookup[second truncation",  # ditto
            "Nice thought]",             # "]" only — BAD
            "I should look this up",     # neither bracket — BAD
            "Finish[Richard Nixon]",
        ],
        condition="act",
    )
    assert (state["n_calls"], state["n_badcalls"]) == (5, 2)
    assert len(llm.calls) == 5  # one call per step: no retry was issued
    assert env.actions == [
        "search[Truncated entity", "lookup[second truncation", "nice thought]",
        "i should look this up", "finish[Richard Nixon]",
    ]
    assert all(call["stop"] == [f"\nObservation {i}:"] for i, call in enumerate(llm.calls, 1))


# -- R1 and the remaining whitespace/scrub literals -------------------------

def test_the_action_is_carried_in_the_state_and_never_re_parsed(monkeypatch):
    """Corrections-log entry 8. The scratchpad stays byte-correct under a re-parse, so
    only the action the ENV was handed can show this; every case below is a shape a chat
    model actually emits."""
    # (a) the action's own text contains the label. rsplit sends "the Movie]".
    _, _, env = run(
        monkeypatch,
        completions=["Hmm.\nAction 1: Search[Action 1: The Movie]", "Got it.\nAction 2: Finish[A]"],
        observations=[OBS],
    )
    assert env.actions[0] == "search[Action 1: The Movie]"

    # (b) the parse-failure retry echoes the label it was prompted with. The reference
    # executes it verbatim and scores "Invalid action:"; a re-parse silently repairs it.
    state, _, env = run(
        monkeypatch,
        completions=["I rambled.", " Action 1: Search[X]", "Got it.\nAction 2: Finish[A]"],
        observations=[OBS],
    )
    assert env.actions[0] == "action 1: Search[X]"
    assert state["scratchpad"].startswith("Thought 1: I rambled.\nAction 1: Action 1: Search[X]\n")

    # (c) a THOUGHT containing the label, which is what kills the mirror-image `split`.
    _, _, env = run(
        monkeypatch,
        completions=["I recall Action 1: Search[old] failed.\nAction 1: Search[A]",
                     "Got it.\nAction 2: Finish[A]"],
        observations=[OBS],
    )
    assert env.actions[0] == "search[A]"

    # (d) the same shape in Act (D5/D17), which has no retry guard and no counter for it.
    _, _, env = run(
        monkeypatch,
        completions=["Action 1: Search[A]", "Finish[A]"],
        observations=[OBS],
        condition="act",
    )
    assert env.actions[0] == "action 1: Search[A]"


def test_parity_two_spaces_after_the_label_survive_into_the_env(monkeypatch):
    """The second parity literal (rule 12). In RAW the action's own strip is a no-op —
    the whole-completion strip had already taken its trailing space — so only a model
    that writes TWO spaces after "Action 1:" can pin that the action half is never
    stripped: the separator eats one space, the other belongs to the action."""
    state, _, env = run(
        monkeypatch,
        completions=["I need X.\nAction 1:  Search[A]", "Got it.\nAction 2: Finish[A]"],
        observations=[OBS],
    )
    assert state["scratchpad"].startswith(f"Thought 1: I need X.\nAction 1:  Search[A]\nObservation 1: {OBS}\n")
    # wikienv strips it back to "Search[A]", whose capital S then scores "Invalid
    # action:" — that is the reference's behaviour and it must stay visible.
    assert env.actions[0] == " Search[A]"


def test_observation_scrub_is_the_two_character_backslash_n(monkeypatch):
    """hotpotqa.ipynb:103 runs unconditionally, and it removes the literal backslash-n
    that clean_str's unicode-escape round trip leaves behind — NOT real newlines, which
    are part of the page text and of every later prompt."""
    state, _, _ = run(
        monkeypatch,
        completions=["I need X.\nAction 1: Search[A]", "Got it.\nAction 2: Finish[A]"],
        observations=["Paris.\\nIt is big.\nA real newline survives."],
    )
    assert "Observation 1: Paris.It is big.\nA real newline survives.\n" in state["scratchpad"]
    assert "\\n" not in state["scratchpad"]


def test_build_graph_rejects_an_unknown_condition():
    with pytest.raises(AssertionError):
        graph_react.build_graph("cot", FakeEnv())


def test_the_graph_condition_and_the_state_condition_must_agree(monkeypatch):
    """A graph built for act, run on a state labelled react, would write react rows."""
    monkeypatch.setattr(graph_react.llm, "complete", FakeLLM("Finish[x]").complete)
    graph = graph_react.build_graph("act", FakeEnv())
    with pytest.raises(AssertionError):
        graph.invoke(graph_react.initial_state(Q, "hotpotqa", "react"))
