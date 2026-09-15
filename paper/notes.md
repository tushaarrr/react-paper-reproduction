# ReAct — implementation notes

Paper: Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, ICLR 2023 (arXiv 2210.03629), `paper/paper.pdf`.
Authors' code: `reference/` (read-only). Citations below are either **[paper §x]** or **file:line**. Nothing is cited from memory;
every code line number was produced with `grep -n` / `sed -n` against the file named.

The reference implements **ReAct only**. Act, Standard, CoT, CoT-SC and both combination rules exist nowhere in `reference/`;
their specs live in ["Our inventions"](#our-inventions-not-in-paper-not-in-reference-code) and are binding.

---

## 1. The ReAct loop

ReAct augments the environment action space `A` with the unbounded language space `L`, so the agent acts in `Â = A ∪ L`
[paper §2]. An element of `L` is a *thought*: it "does not affect the external environment, thus leading to no observation
feedback. Instead, a thought `â_t` aims to compose useful information by reasoning over the current context `c_t`, and update
the context `c_{t+1} = (c_t, â_t)`" [paper §2]. For knowledge-intensive tasks "we alternate the generation of thoughts and
actions so that the task-solving trajectory consists of multiple thought-action-observation steps" [paper §2] — dense thought,
strict `Thought i → Action i → Observation i`. The model generates `Thought i` and `Action i`; the environment generates
`Observation i` [paper Fig. 1 caption]. The context is never truncated: the prompt at step `i` is the fixed instruction +
exemplars + the live question + every step emitted so far, and the model continues from the literal prefix `Thought {i}:`.
The episode ends when the model emits `finish[answer]` [paper §3.1, lowercase there] — spelled `Finish[answer]` by every
**ReAct/Act** exemplar and by the instruction header (`hotpotqa.ipynb:80`); the Standard and CoT keys contain `Finish[`
**0** times (counted: `webact_simple6` 6, `webthink_simple6` 6, `webact_simple3` 3, `webthink_simple3` 3,
`cotqa_simple6`/`webqa_simple6`/`cotqa_simple3`/`webqa_simple3` 0) —, and lowercased on its first character only by
`hotpotqa.ipynb:102` before dispatch — or when the step limit is reached (7 for HotpotQA, 5 for FEVER per [paper §3.2];
**7 for both in the code** — see §2 and the code log).

### Pseudocode (what we implement — code-faithful, paper-annotated)

```python
# HotpotQA / FEVER, dense thought. Prompt assembly = hotpotqa.ipynb:83; webthink() = hotpotqa.ipynb:85-115
# (body :86-115; `for i in range(1, 8)` at :91, loop body :92-109; :110-114 run after the loop).
# FEVER.ipynb:78-108 is the same function with no instruction header (def at :78, loop at :84, loop body :85-102).
prompt = instruction + exemplars          # instruction: HotpotQA code-only (§5); FEVER header is inside the JSON value
prompt += question + "\n"                 # :89 / :82; question already carries "Question: " / "Claim: "
                                          # wrappers.py:97 / wrappers.py:166
n_calls = n_badcalls = 0
for i in range(1, 8):                     # 7 steps; hotpotqa.ipynb:91, FEVER.ipynb:84
    n_calls += 1
    thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])   # :93 / :86
    try:
        thought, action = thought_action.strip().split(f"\nAction {i}: ")          # :95 / :88  STRICT 2-WAY UNPACK
    except Exception:                                                              # bare except in ref; :96 / :89
        n_badcalls += 1                                                            # :98 / :91
        n_calls += 1                                                               # :99 / :92
        thought = thought_action.strip().split('\n')[0]                            # :100 / :93
        action  = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip() # :101 / :94 — see note
    obs, done, info = step(env, action[0].lower() + action[1:])                    # :102 / :95 — see arity note below
    obs = obs.replace('\\n', '')          # strips the 2-char sequence \n, NOT newlines; :103 / :96
    prompt += f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"      # :104-105 / :97-98
    if done:
        break
if not done:
    obs, done, info = step(env, "finish[]")   # forced empty finish, scores 0; :111 / :104
```

Two spelling notes against the source, neither behavioural: the retry call's stop list is written `stop=[f"\n"]` at
`hotpotqa.ipynb:101` / `FEVER.ipynb:94` — an f-string with no placeholders, value identical to `["\n"]` (`llm`'s default at
`hotpotqa.ipynb:21` really is `stop=["\n"]`); and the scratchpad append is two statements in the reference,
`step_str = f"..."` (`hotpotqa.ipynb:104`, `FEVER.ipynb:97`) then `prompt += step_str` (`:105` / `:98`).

### Whitespace parity in the scratchpad (verified against the first live call)

**`.strip()` runs once, on the WHOLE completion, and neither the thought nor the action is ever stripped individually.**
`hotpotqa.ipynb:95` is `thought, action = thought_action.strip().split(f"\nAction {i}: ")` — one strip, and a literal
separator that **includes the trailing space**. The only other strip in the loop is `:101`, on the *retry* call's return.
Nothing tidies the halves afterwards, and `:104` interpolates them as they are.

This is not cosmetic: the scratchpad **is** the next step's prompt (`:105`), so any whitespace we tidy changes what the model
sees at every later step of the episode, and diverges from the reference on the byte level.

Verified against this repo's first live call, whose raw completion was, byte for byte (Python repr, so the trailing spaces
are visible — a fenced block cannot show them):

```python
raw = ('I need to search for Irene Jacob and find out what movie she completed before the American action crime '
       'thriller film directed by Stuart Bird. \nAction 1: Search[Irene Jacob] ')
```

Replaying `:93-104` on it gives, all three verified:

* `thought == 'I need to search ... directed by Stuart Bird. '` — **keeps** its trailing space (the whole-string strip only
  touched the two ends of the completion, and this space is interior to it).
* `action == 'Search[Irene Jacob]'` — **loses** its trailing space, because that one *was* at the end of the completion.
* the step's scratchpad block is:

```python
block = ('Thought 1: I need to search for Irene Jacob and find out what movie she completed before the American action '
         'crime thriller film directed by Stuart Bird. \nAction 1: Search[Irene Jacob]\nObservation 1: <obs>\n')
```

The space after `Bird.` travels into step 2's prompt. **Rule: do not tidy whitespace anywhere in the scratchpad, and never
strip `thought` or `action` individually anywhere in the pipeline.** Pinned by the parity test in
`tests/test_graph_react.py`, which feeds that exact raw string and asserts that exact block; two of the step-05 mutations
(strip the thought individually; split on `"\nAction {i}:"` without the trailing space) are killed by it alone — the second
one also hands the env `' Search[Irene Jacob]'`, which `wikienv.py:127` strips back to `Search[...]`, whose **capital** `S`
then fails the lowercase prefix test at `:132` and scores `Invalid action:`.

**Two consequences of the same literalism, worth stating because both look like bugs:**

* The observation's own trailing newline is **not** absorbed. `:104` appends `f"...Observation {i}: {obs}\n"` to an `obs`
  that, for `finish[...]`, already ends with `\n` — so a normally finishing trajectory ends with
  `Episode finished, reward = 0\n\n`, i.e. the §2 literal **plus** the step separator. Byte-equality assertions (C1(f),
  C4(a)) must expect both newlines.
* **The thought is not a state field; the action is.** `think_act` appends
  `f"Thought {i}: {thought}\nAction {i}: {action}"` to the scratchpad, and its concatenation with `execute`'s
  `f"\nObservation {i}: {obs}\n"` is byte-identical to the reference's single `step_str`, which is what the parity test
  checks. The action *also* travels in the state, as the 11th field `action` of `05-react-graph.md:3` (a plain `str`, so
  unlike the env of D21 it costs checkpointing nothing). **`execute` must never recover it from the scratchpad.** An
  earlier version of this section justified `rsplit(f"Action {i}: ", 1)[-1]` with "the real action is always the last";
  that is false, and the correction is logged as CLAUDE.md corrections-log entry 8. `rsplit` truncates an *action* whose
  own text contains the label — `Action 1: Search[Action 1: The Movie]` executes `the Movie]`, where the reference
  executes `search[Action 1: The Movie]` — and `split` truncates on a *thought* that contains it, so neither direction is
  safe. Two more reachable cases, both chat-model shapes: a retry that echoes the label it was prompted with
  (`" Action 1: Search[X]"` → the reference executes `action 1: Search[X]` and scores `Invalid action:`, a re-parse
  silently "repairs" it to `search[X]`), and the same shape in Act (D5/D17), which has no retry guard and no counter for
  it. In every case the scratchpad stays byte-correct, so the parity test, `runs/` and `results/calls.csv` all look
  normal while a different Wikipedia action was executed.

**`step(env, action)` is not `env.step`.** `hotpotqa.ipynb:102` and `:111` (FEVER.ipynb:95, :104) call the notebook's retry
helper defined at `hotpotqa.ipynb:46-52` / `FEVER.ipynb:46-52`:
`while attempts < 10: try: return env.step(action) except requests.exceptions.Timeout: attempts += 1`.
It never actually retries: `requests` is referenced at `:51` but imported in neither notebook (`grep -n "import requests"` →
0 hits in both), so the `except` clause raises `NameError` the moment any exception reaches it. It also falls through and
returns `None` after 10 failures, which would then unpack-crash. We replace it (see "Deliberate deviations we add").

**Return arity — decided (D7).** The reference returns a 4-tuple `(obs, reward, done, info)` (`wikienv.py:160`); we use the
3-tuple `(observation, done, info)` from `prompts/claude-code/02-environment.md:1`. Safe because `reward` is set to `0` at
`wikienv.py:125` and never reassigned, so the dropped element is always `0`; EM is computed by our runner at episode end,
exactly as the reference wrappers do (`wrappers.py:109-124`). `info = {"steps": int, "answer": str | None}`
(`wikienv.py:41-42`). Logged in "Taken from code, not paper".

**`n_steps` is the loop index, not `env.steps`.** `wikienv.py:158` increments `steps` on *every* branch including the forced
`finish[]`, so a step-limit episode leaves `info["steps"] == 8`. What we log as `n_steps` per question is `i`, the index of
the last model step (max 7), so `mean_steps` and `pct_hit_step_limit` in `results/results.csv` are comparable across
conditions. Record `info["steps"]` separately if wanted; never use it as `n_steps`.

**Step accounting for every condition — decided (D15).** Only `act` and `react` have a loop, so the other five rows need an
explicit rule, or five of the seven `results/results.csv` condition rows are invented at implementation time:

| condition class | JSONL `n_steps` | JSONL `hit_step_limit` | `results.csv` `mean_steps` / `pct_hit_step_limit` | `results.csv` `total_cost` |
|---|---|---|---|---|
| `standard`, `cot`, `cotsc` | `0` | `false` | **empty string, not `0`** — `0` would read as a measurement rather than "not applicable" | sum of the run's `cost_usd` |
| `act`, `react` | the loop index `i` at which the episode ended (max 7) — **never** `env.steps`, which reaches 8 after the forced `finish[]` | `true` iff the loop exhausted | computed over the run's lines | sum of the run's `cost_usd` |
| `react_to_cotsc`, `cotsc_to_react` | inherited from whichever source line supplied the chosen prediction | inherited likewise | computed over the inherited values | **`0.0`** — D6 issues no LLM calls |

Combination lines additionally carry `source` ∈ {`"react"`, `"cotsc"`} per question, so the mix is auditable from `runs/`
alone. This is what C9's "every number in `README.md` also appears in `results/results.csv`" guard checks against.

**Graph state — the 11 fields of `05-react-graph.md:3`** (`question, task, scratchpad, action, step, done, answer,
n_calls, n_badcalls, hit_step_limit, condition`), a `TypedDict`. `action` carries the parsed action from `think_act` to
`execute` (corrections-log entry 8 — re-deriving it from the scratchpad executes the wrong action while leaving the
trajectory byte-correct). **`step` initialises to 1**, and `execute` increments it *after*
appending the step to the scratchpad, so the router (`route_after_execute`, whose `step > 7` test replaces the state-writing
conditional edge of `05-react-graph.md:9` — D22 below) sees `step == 8` after the 7th model step. Starting at 0 gives 8 model calls and a `Thought 0:` continuation prefix — silently,
with no test failure unless C4(d) counts calls.

**The env is not in the state — ownership and lifecycle (D21).** The state is exactly those 11 fields; a live `WikiEnv` is
**not** a twelfth one — a live object in the state breaks checkpointing and contradicts the declared schema. Instead:

* `build_graph(condition, env)` returns a compiled graph whose `think_act`, `execute` and `force_finish` nodes **close over**
  `env`. The env reaches `execute` through that closure, never through the state.
* `src/run.py:run_one(idx, question, task, condition, env)` **owns** the env: it calls `env.reset()` **once per question**,
  before invoking the graph, then invokes the graph and computes EM from the final state. No node ever calls `reset()`.
* **One `WikiEnv` instance is reused across every question in a run**, matching the reference, which constructs the env once
  (`hotpotqa.ipynb:42-43`, `env = wikienv.WikiEnv()` then the wrapper) and resets per question (`hotpotqa.ipynb:86`,
  `question = env.reset(idx=idx)`). The disk cache (D18) makes reuse free; `reset()` is the thing that prevents page and
  lookup bleed between questions — the leak described under `reset()` in §2. D7 deletes the wrapper layer that used to carry
  that `reset`, so it is reassigned here, and it is **tested in C8** (a two-question run), not in C1, where no runner exists
  and the assertion would pass trivially.

**The step limit needs a NODE, not a conditional edge (D22) — prompt-spec correction to `05-react-graph.md:9`.**
That line instructs: "Conditional edge after execute: END if done; if step > 7 and not done, **set hit_step_limit=True, call
finish[]** and END; else back to think_act." **That is not implementable in LangGraph as written.** A conditional edge's path
function receives the state and returns a route only; anything it writes to the state is **discarded**. Verified empirically
against the installed **langgraph 1.2.11**: a router that does `state["hit_step_limit"] = True` and returns `END`, sitting
after a node that returns `{"step": state["step"] + 1}`, yields the final state `{'step': 8, 'hit_step_limit': False}` — the
node's write survives, the router's does not. The forced `finish[]` would likewise fire from a routing function as an
untracked side effect. This is a defect in the step prompt, not in the plan; it is logged as a prompt-spec correction in the
"Deliberate deviations we add" table.

Decided shape, behaviourally identical to the reference's post-loop `if not done: step(env, "finish[]")`
(`hotpotqa.ipynb:110-111`):

* `route_after_execute(state) -> str` is **pure**: `"end"` if `done`, `"force_finish"` if `step > 7`, else `"think_act"`.
  It writes nothing to the state.
* `force_finish` is a **real node**: sets `hit_step_limit = True`, calls `env.step("finish[]")`, sets `answer = ""` and
  `done = True`, returns that state update, and routes to `END`.
* Graph: `think_act → execute → { END | force_finish → END | think_act }`.

`force_finish` is therefore a third node an implementer must write, a third entry point C4 must name and step 08 must cite.

**What a step-limit episode's scratchpad ends with (D24).** Nothing is appended after the 7th step. `hotpotqa.ipynb:110-111`
runs the forced `finish[]` **after** the last `prompt += step_str` (`:105`), so its observation never enters the trajectory:
the scratchpad ends at `Observation 7: <obs>\n`, and `force_finish` appends no `Thought 8:` / `Action 8:` / `Observation 8:`
lines. The line records `hit_step_limit = true`, `answer = ""`, `em = 0`. Consequently the §2 claim that every trajectory ends
with the `Episode finished, reward = 0\n` literal is narrowed to episodes that finish **normally**, via a model-emitted
`finish[...]`.

Exact serialization of one step **within an episode** (no blank line between steps, one trailing `\n`):

```
Thought {i}: {thought}
Action {i}: {action}
Observation {i}: {obs}
```

This says nothing about the exemplar → live-question boundary, which differs per task and per key — see the whitespace
table in §5. Getting that boundary wrong is the single most common silent prompt defect.

Stop condition, precisely:
1. `finish[...]` parses → `wikienv.py:148-151` sets `self.answer`, `done = True`.
2. Loop exhausted after `i == 7` → driver sends `finish[]`, which sets `answer = ""` (`wikienv.py:149-150`, `""` is not
   `None` so the episode terminates) and scores EM 0 against any non-empty gold. In our graph that is the `force_finish`
   node, which also sets `hit_step_limit = True` (D22); nothing is appended to the scratchpad after it (D24).
3. Any later `step()` call once `answer is not None` returns immediately with `done = True` and **without incrementing
   `steps`** (`wikienv.py:128-130` vs `:158`).

---

## 2. The Wikipedia environment

Paper spec, in full [paper §3.1 "Action Space"]:

> "(1) `search[entity]`, which returns the first 5 sentences from the corresponding entity wiki page if it exists, or else
> suggests top-5 similar entities from the Wikipedia search engine, (2) `lookup[string]`, which would return the next
> sentence in the page containing string, simulating Ctrl+F functionality on the browser. (3) `finish[answer]`, which would
> finish the current task with `answer`."

That is the whole of the paper's environment specification. Everything below is `reference/wikienv.py`.

### `clean_str` — every observation string passes through this

`wikienv.py:10-11`, verbatim:

```python
def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
```

It round-trips the scraped text through `unicode-escape`/`latin1`. Two consequences that an implementor who substitutes
`html.unescape` or a no-op will silently lose: (a) real escape sequences inside the HTML are *decoded*; (b) literal
two-character `\` + `n` sequences appear in observations, which is exactly what the driver's `obs.replace('\\n', '')`
(`hotpotqa.ipynb:103`) patches over. Copy the line as-is; every C1 fixture assertion depends on it.

**On some inputs it does not degrade — it raises (MIN1).** Reproduced in this repo's venv against the `wikienv.py:10-11`
body: `clean_str(r'literal \u4e2d escape')` → `UnicodeEncodeError: 'latin-1' codec can't encode character '\u4e2d'`, and
`clean_str(r'path C:\xyz here')` → `UnicodeDecodeError: 'unicodeescape' codec can't decode bytes ... truncated \xXX escape`.
A page whose text carries a literal `\x` or `\u` sequence would kill a 500-question run mid-flight — deterministically, and
identically on the cached rerun, since rule 5 replays the same bytes. **Policy: wrap the call; on `UnicodeDecodeError` or
`UnicodeEncodeError` return the input block unchanged and log a warning** (the same shape of guard as D9's empty action).
Asserted in C1(l).

### Dispatch

`wikienv.py:124-160`. `action = action.strip()` (`:127`), then prefix tests that are **lowercase and case-sensitive** and
require a trailing `]`: `search[` (`:132`), `lookup[` (`:137`), `finish[` (`:148`), `think[` (`:153`), else invalid
(`:155-156`). `steps += 1` on every branch (`:158`). `reward = 0` at `:125`, never reassigned.

### `reset()` — full state list

`wikienv.py:44-57` resets **six** fields plus the observation. All six must be reset, or question *k*'s page leaks into
question *k+1*'s `lookup` and every downstream observation drifts:

```python
self.obs = ("Interact with Wikipedia using search[], lookup[], and finish[].\n")   # :47-48
self.page = None            # :49
self.lookup_keyword = None  # :50
self.lookup_list = None     # :51
self.lookup_cnt = None      # :52
self.steps = 0              # :53
self.answer = None          # :54
```

Both task wrappers discard that observation and substitute `f"Question: {q}"` / `f"Claim: {c}"` (`wrappers.py:97`, `:166`).
Because `construct_lookup_list` returns `[]` when `page is None` (`:61-62`), a correct reset makes `lookup[anything]` before
any `search` return exactly `"No more results.\n"` — that is the C1 assertion that proves the reset is complete.

### `search[entity]`

URL (`wikienv.py:99-100`), spaces → `+` and nothing else escaped:

```python
entity_ = entity.replace(" ", "+")
search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
```

Article vs. results page is decided purely by `soup.find_all("div", {"class": "mw-search-result-heading"})` (`:105-107`).

* **Failure (results page)** — `:108-109`:
  ```python
  self.result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
  self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
  ```
  The list renders as Python `repr(list[str])` — single quotes, `', '` separators — and a `.` **after** the `]`, no trailing
  newline. All titles stay in `result_titles`; only 5 reach the observation.

  **The five titles are NOT a stable live constant (D19).** Four fetches of `?search=Colorado+orogenyy` on 2026-09-15
  returned 20 `mw-search-result-heading` divs every time but **two different orderings** within the top 5 — one fetch put
  `Colorado Mineral Belt` at position 3 and `Sevier orogeny` at 4, the other three the reverse. Wikipedia's search backend
  reorders near-ties between requests, so a five-title literal taken from the network is not reproducible.
  **Rule: record the fixture first, then paste the literal that *that recorded file* produces.** Done —
  `tests/fixtures/colorado_orogenyy_similar.html` is recorded (20 result divs) and replaying `wikienv.py:106-109` against it
  yields, byte for byte:
  ```text
  Could not find Colorado orogenyy. Similar: ['Colorado orogeny', 'Laramide orogeny', 'Sevier orogeny', 'Colorado Mineral Belt', 'Wyoming Craton'].
  ```
  C1(c) asserts that against the fixture, never against the network. The `slow` live smoke test asserts only the stable
  parts: `len(result_divs) == 20` and `titles[0] == 'Colorado orogeny'`.
* **Success (article)** — `:111-122`. The scrape is **not** a join; quoting `:111` and `:115-120` verbatim:
  ```python
  page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]   # :111
  ...
  self.page = ""                        # :115
  for p in page:                        # :116
    if len(p.split(" ")) > 2:           # :117  (> 2, i.e. blocks of <= 2 space-separated tokens are dropped)
      self.page += clean_str(p)         # :118
      if not p.endswith("\n"):          # :119  tested on the PRE-clean_str block
        self.page += "\n"               # :120
  ```
  The **separator between blocks is load-bearing**: drop it and adjacent blocks fuse into one `'. '` split, shifting the
  5-sentence observation. The trailing newline after the last block is **not**, and neither is `:119`'s `endswith("\n")`
  guard — blocks come from `p.get_text().strip()` (`:111`), so the guard is never true, and `get_page_obs` drops empty
  paragraphs anyway. `"\n".join(kept)` is therefore **equivalent** here; we write the loop for line-parity with `:115-120`
  but omit the unreachable guard (`src/wiki_env.py:115-120`).
  Then `get_page_obs` (`:76-87`):
  ```python
  paragraphs = page.split("\n"); paragraphs = [p.strip() for p in paragraphs if p.strip()]
  sentences = []
  for p in paragraphs: sentences += p.split('. ')
  sentences = [s.strip() + '.' for s in sentences if s.strip()]
  return ' '.join(sentences[:5])
  ```
  So "first 5 sentences" = **first 5 `'. '`-delimited fragments**, each with `'.'` re-appended — which doubles the period at
  every paragraph boundary and merges real sentences whose period is followed by a citation marker (`2018.[1] Renzi has...`).
  No trailing newline. Verified against `tests/fixtures/colorado_orogeny_hit.html`: the doubled period appears as
  `...part of the larger Yavapai orogeny.. The Colorado orogen, formerly called...`.
* **Document order is lost.** `find_all("p") + find_all("ul")` places **every** `<p>` block before **every** `<ul>` block, so
  navigation chrome (which lives in `<ul>`) lands at the *end* of `self.page`. It can pollute `lookup` results; it never
  reaches the first-5-sentence observation, whose fragments come from the article lead. (This replaces an earlier claim in
  this file that the lead sentence is `Pages for logged out editors learn more.` — that claim was false; see the drift note
  below.)
* **Disambiguation retry (D8) — load-bearing.** `:112-113`: if any scraped block contains `"may refer to:"`, the env
  re-searches `"[" + entity + "]"`. This is why the exemplars and [paper App. C.1] contain
  `Could not find [Adam Clayton Powell]. Similar: [...]` **with square brackets**, while App. E.1 shows the unbracketed
  `Could not find goddess frigg. Similar: [...]`. Verified live 2026-09-15: `Adam Clayton Powell` still hits a disambiguation
  page and still resolves to the bracketed miss. Implement it or our observations stop matching the format the model is
  few-shot primed on. The reference recursion is unbounded; **we guard it to depth 1** so a pathological page cannot loop
  (logged as a row of the "Deliberate deviations we add" table under "Taken from code, not paper"; D8 is indexed
  under "Our inventions" as a stub pointing here). **Depth-1 outcome (MIN2):** if the bracketed retry page is *itself* a
  disambiguation page, do **not** re-search again — return that retry page's own result as-is, whichever branch it lands in
  (miss literal or 5-sentence article observation). Asserted in C1(m).
* **Zero-result pages (D10, 2026 drift).** A query with no near-matches renders a page with **no**
  `mw-search-result-heading` divs at all, so the reference logic falls through to the article branch. Verified against
  `tests/fixtures/nonexistent_miss.html` (`grep -c mw-search-result-heading` → 0): the observation becomes
  `There were no results matching the query.. The page "Qwertzuiop Zzyzxian Orogeny" does not exist. You can create a draft and submit it for review or request that a redirect be created.. Main pageContentsCurrent eventsRandom articleAbout WikipediaContact us. HelpLearn to editCommunity portalRecent changesUpload fileSpecial pages.`
  **Detection (D23):** zero `mw-search-result-heading` divs **and** extracted page text beginning with
  `There were no results matching the query`. **The emitted observation is the ordinary miss literal of `wikienv.py:108-109`
  with an empty `result_titles`** — one literal, byte for byte, and *not* the raw "There were no results…" text:
  ```text
  Could not find Qwertzuiop Zzyzxian Orogeny. Similar: [].
  ```
  C1(d) asserts exactly that against the fixture. **Miss semantics:** like the reference's miss branch, this path leaves
  `page`, `lookup_keyword`, `lookup_list` and `lookup_cnt` **untouched** — a failed search never clears the current page,
  because the reset at `wikienv.py:122` is inside the success branch only. Logged twice: as **D10/D23** under
  "Our inventions" and as a row of the "Deliberate deviations we add" table (which is where `08-verify.md:19`'s per-deviation effect estimate lives) — it cannot affect parity on real
  questions, only on pathological searches.
* A **failed** search does not clear `page` / `lookup_*` — the reset at `:122` is inside the success branch only.

**Observation drift (D12).** `Pages for logged out editors learn more.` appears **479 times** in the authors' own stored 2022
outputs (counted with `str.count` over `reference/FEVER.ipynb`; `grep -c` reports 480 because its `.` is a wildcard and one
further occurrence is followed by an escaped newline rather than a period — **0** in `hotpotqa.ipynb` either way), because Wikipedia served that banner inside a `<p>` at
the time. A live fetch on 2026-09-15 of `Colorado orogeny`, `Milhouse` and `High Plains (United States)` contains the string
**zero** times, and the first fragment is the real lead paragraph. Treat exact observation strings from the stored notebook
outputs as historical, not as fixtures.

### `lookup[string]`

`wikienv.py:137-147`. `construct_lookup_list` (`:59-74`) runs the same split pipeline over `self.page` and keeps sentences
where `keyword.lower() in p.lower()` (case-insensitive substring, no `[:5]` cap; returns `[]` if `page is None`, `:61-62`).
The list is rebuilt only when the keyword string differs (exact, case-sensitive, `:139`).

* Hit (`:146`): `f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) " + sentence` — 1-based, spaces around `/`, one
  space after `)`, **no trailing newline**.
* Exhausted (`:144`): `"No more results.\n"` — **trailing newline included**.

### `finish[answer]`

`wikienv.py:148-152`: `answer = action[len("finish["):-1]`, `done = True`, obs `f"Episode finished, reward = {reward}\n"`
which from `WikiEnv` is always `reward = 0` (`wikienv.py:125`, never reassigned). The reference's *task wrapper* overwrites
it with the real EM (`wrappers.py:130-131`, `wrappers.py:190-191`), so in the reference the agent sees e.g.
`Episode finished, reward = 1\n`.

**Our env emits exactly `"Episode finished, reward = 0\n"` — always, one literal.** D7 drops `reward` from the
return tuple and moves EM to the runner, so there is no real-EM value to substitute at env level, and **the runner never
rewrites the observation**. EM lives only in the JSONL `em` field. This is the literal that ends every **normally finishing** ReAct/Act
trajectory written to `runs/` (CLAUDE.md rule 6) and every `text` in `data/sft/{react,act}_train.jsonl` (**not** `cot_train.jsonl` — a CoT trajectory never touches the env) (C11 keeps only `em == 1`
records, which by construction finished normally), which is what lets C1(f) and C4(a) assert byte-equality instead of using
`...` placeholders. **A step-limit episode does not end with it (D24):** the forced `finish[]` runs *after* the last
scratchpad append (`hotpotqa.ipynb:110-111` vs `:105`), so that trajectory ends at `Observation 7: <obs>\n` and this literal
never appears in it. The one behavioural consequence: unlike the reference, our fine-tuning
trajectories never leak the gold EM into the trajectory text — a leak the reference would otherwise train on.

### The fetch layer — `WikiEnv._fetch(url)` (D18)

Everything above assumes an HTTP GET. That GET is **ours**, not the reference's: `wikienv.py:99-102` calls
`requests.get(search_url)` with no timeout, no `User-Agent` and no cache. CLAUDE.md rule 5 and `02-environment.md:6` require
the **cache** and the **retries** — that line reads in full "A disk cache (sqlite or a JSON directory) keyed by the exact URL,
so reruns never hit the network. Retry timeouts up to 10 times." and says nothing about a header. **The `User-Agent` is ours
(D18)**, because bare-UA requests to `en.wikipedia.org` are 403'd today; it belongs in the deviations table, and is not
quoted here as a prompt requirement. All four live in one function, `WikiEnv._fetch(url)`, owned and tested by C1 (the fourth, `response.raise_for_status()`, is ours — see the deviations table):

* **Disk cache keyed by the exact URL.** Format: a **JSON directory**, `data/cache/wiki/<sha256(url)>.json`, each file
  `{"url": ..., "fetched_at": ..., "html": ...}`. (`02-environment.md:6` allows sqlite or a JSON directory; pinning one
  stops C1 and C3 each inventing a different store. The LLM cache is the separate store described in C3.) A second call for
  the same URL makes **zero** network calls and returns byte-identical text.
* **Up to 10 attempts on timeout** — 1 initial try + 9 retries; 10 consecutive timeouts raise (`02-environment.md:6`,
  which says "retry timeouts up to 10 times"; the reference's dead helper it replaces, `hotpotqa.ipynb:46-52`, is
  `attempts = 0; while attempts < 10`, i.e. 10 attempts total). `MAX_ATTEMPTS = 10`; C1(k) pins it. Do **not** "fix" this
  to 11 attempts to match a literal reading of "10 retries" — that breaks C1(k).
* **A `User-Agent` header.** Bare-UA requests to `en.wikipedia.org` are 403'd today. The exact string, used by the env and
  by every recorded fixture:
  ```text
  react-langgraph-repro/0.1 (research reproduction; contact via repo)
  ```
  Recorded once here so fixtures are reproducible. To re-record a fixture:
  `curl -sL -A 'react-langgraph-repro/0.1 (research reproduction; contact via repo)' 'https://en.wikipedia.org/w/index.php?search=Colorado+orogenyy' > tests/fixtures/colorado_orogenyy_similar.html`

### Other observations `WikiEnv` can emit

| Trigger | Exact literal | Line |
|---|---|---|
| `reset()` | `Interact with Wikipedia using search[], lookup[], and finish[].\n` (discarded by both task wrappers) | `wikienv.py:47-48` |
| `think[...]` | `Nice thought.` (code-only 4th action; not in the paper's 3-action space; appears in 0 of the 12 prompt keys) | `wikienv.py:154` |
| anything else | `"Invalid action: {}".format(action)` — the argument is the bare `action`, already stripped at `:127`; no penalty, `steps` still increments, episode continues | `wikienv.py:156` |

### Step limit per task

| Task | Paper | Code | What we use |
|---|---|---|---|
| HotpotQA | 7 [paper §3.2, fn 3] | `for i in range(1, 8)` — `hotpotqa.ipynb:91` | **7** |
| FEVER | **5** [paper §3.2] | `for i in range(1, 8)` — `FEVER.ipynb:84` | **7** (code + `tests/EXPECTED.md:24`) |

Paper §3.2: "We set 7 and 5 steps for HotpotQA and FEVER respectively as we find more steps will not improve ReAct
performance"; fn 3: "Of all trajectories with correct final answers, those with 7 steps on HotpotQA and 5 steps on FEVER
only take up 0.84% and 1.33% respectively." **The released FEVER notebook uses 7. Follow the code.**

---

## 3. Datasets

| | HotpotQA | FEVER |
|---|---|---|
| Dev file | `data/hotpot_dev_v1_simplified.json` (`wrappers.py:13`) | `data/paper_dev.jsonl` (`wrappers.py:19`) |
| Train file | `data/hotpot_train_v1.1_simplified.json` (`wrappers.py:12`) — **90,447** entries, keys `question, answer, type`; the bootstrap source for the 3,000 SFT trajectories (`tests/EXPECTED.md:15`, paper **§3.2** "Finetuning" — the *results* in §3.3) | `train.jsonl` (`wrappers.py:18`) — not present in `data/`, not used |
| Test file | `data/hotpot_test_v1_simplified.json` (`wrappers.py:14`) — 7,405 entries, key `question` **only, no `answer`** → unusable for EM. Never load it by mistake | — |
| Dev size (verified) | 7,405 entries, keys `question, answer, type` | 9,999 lines, keys `id, verifiable, label, claim, evidence`; labels 3333/3333/3333 |
| `type` values (verified) | `bridge` 5,918 / `comparison` 1,487 over the 7,405 dev entries | n/a |
| Reference loader | `wrappers.py:84-85` → `(question, answer)`, **drops `type`** | `wrappers.py:143-154` → `(claim, label)` |
| Our loader | `load_hotpotqa() -> (question, answer, type)` — 3-tuple, a **deliberate deviation** required by `prompts/claude-code/03-data-metrics.md:3` and by step 07's bridge/comparison breakdown (`07-full-runs.md:6`). EM-neutral: the extra field is never fed to a prompt | `load_fever() -> (claim, label)` |
| Prompt seed | `f"Question: {question}"` (`wrappers.py:97`) | `f"Claim: {claim}"` (`wrappers.py:166`) |
| Answer format | free-form short string inside `Finish[...]` | exactly one of `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO` |
| Metric | EM [paper Table 1 header `HotpotQA (EM)`] | accuracy [paper Table 1 header `Fever (Acc)`] |

Split/size for Table 1 is **not stated in the paper**; only App. A.1 Table 5 says "On HotpotQA, we randomly sample a subset
of 500 validation questions." Everything about *which* 500 comes from the code.

### The 500 evaluation items

`hotpotqa.ipynb:126-132` and `FEVER.ipynb:9362-9368` are byte-identical:

```python
idxs = list(range(7405))
random.Random(233).shuffle(idxs)
for i in idxs[:500]:
```

Replayed locally: first five = `3687, 6238, 5388, 3522, 3824` (matches `tests/EXPECTED.md:9`), `max(idxs[:500]) == 7390`.
**FEVER uses `range(7405)` against a 9,999-line file**, so FEVER indices 7405–9998 are never sampled. Reproduce, do not fix
(`tests/EXPECTED.md:10-11`). The bound is a **constant 7405 for both tasks** — the sampler must never see
`len(load_fever())`, which is the obvious "correct" thing to do and the failure mode `tests/EXPECTED.md:10-11` singles out.
C2 pins this with `eval_indices_for("fever") == eval_indices(7405)` and `max(eval_indices_for("fever")) == 7390`.

### Metric normalization

`wrappers.py:42-56` — SQuAD normalization, applied in the order lower → remove punctuation → remove articles →
collapse whitespace:

```python
def normalize_answer(s):
  def remove_articles(text):   return re.sub(r"\b(a|an|the)\b", " ", text)
  def white_space_fix(text):   return " ".join(text.split())
  def remove_punc(text):       exclude = set(string.punctuation); return "".join(ch for ch in text if ch not in exclude)
  def lower(text):             return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))
```

`string.punctuation` is ASCII-only (curly quotes and en/em dashes survive); articles are removed *after* punctuation, so
`the-film` → `thefilm` and the article is not stripped. EM = equality of the two normalized strings (`wrappers.py:109-115`
HotpotQA, `:178-184` FEVER). FEVER therefore scores only if the prediction normalizes to `supports` / `refutes` /
`not enough info`. The paper reports EM only and never defines the normalization.

**Do not copy the reference's local names.** `HotPotQAWrapper.get_reward`/`get_metrics` bind `pred = normalize_answer(gold)`
and `gt = normalize_answer(prediction)` — swapped (`wrappers.py:111-112`, `:119-120`). Harmless (EM is symmetric and
`f1_score` guards both sides) but confusing.

### `f1_score` — `wrappers.py:58-78`, verbatim

```python
def f1_score(prediction, ground_truth):
  normalized_prediction = normalize_answer(prediction)
  normalized_ground_truth = normalize_answer(ground_truth)

  ZERO_METRIC = (0, 0, 0)

  if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC
  if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC

  prediction_tokens = normalized_prediction.split()
  ground_truth_tokens = normalized_ground_truth.split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return ZERO_METRIC
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1, precision, recall
```

**Return shape: a 3-tuple `(f1, precision, recall)`.** The only caller takes `[0]` (`wrappers.py:122`:
`f1 = f1_score(pred, gt)[0]`). Our `src/data.py:f1(pred, gold)` returns the **float** — i.e. `f1_score(...)[0]` — so callers
never index. Keep the two yes/no/noanswer short-circuits: they are why `f1("yes", "no") == 0.0` instead of 0 by token
overlap. FEVER does not use F1 at all: `wrappers.py:193` sets `f1 = em = reward`.

---

## 4. The seven conditions of Table 1

Definitions from [paper §3.2 "Baselines" / "Combining Internal and External Knowledge"]. Quotes are **verbatim but some are
clipped to the defining clause** — items 1 and 3 start mid-sentence (the paper's sentence opens "We also build a self-consistency
baseline (CoT-SC) (Wang et al., 2022a;b) by …") and item 6 drops the trailing "as we find more steps will not improve ReAct
performance". Items 2, 4 and 7 are complete sentences.

1. **Standard** — "(a) Standard prompting (Standard), which removes all thoughts, actions, observations in ReAct
   trajectories." Prompt = `Question:` / `Answer:` pairs.
2. **CoT** (Wei et al. 2022) — "(b) Chain-of-thought prompting (CoT) (Wei et al., 2022), which removes actions and
   observations and serve as a reasoning-only baseline."
   **HotpotQA only:** every CoT exemplar thought opens with the literal `Let's think step by step. ` (trailing space; the
   string `Thought: Let's think step by step. ` occurs exactly **6/6** times in `prompts_naive.json['cotqa_simple6']`).
   **FEVER CoT exemplars carry no such prefix** — `"Let's think step by step"` occurs **0** times in
   `fever.json['cotqa_simple3']`; its thoughts open directly, e.g.
   `Thought: Nikolaj William Coster-Waldau appeared in the 2009 Fox television film Virtuality, so he has worked with the Fox Broadcasting Company.`
   **Do not add the prefix to FEVER.** The FEVER CoT continuation is bare `Thought:` (see D3).
3. **CoT-SC** (Wang et al. 2022a;b) — "sampling **21** CoT trajectories with decoding **temperature 0.7** during inference and
   adopting the **majority answer**, which is found to consistently boost performance over CoT."
4. **Act** — "(c) Acting-only prompt (Act), which removes thoughts in ReAct trajectories, loosely resembling how WebGPT
   (Nakano et al., 2021) interacts with the Internet to answer questions, though it operates on a different task and action
   space, and uses imitation and reinforcement learning instead of prompting." Prompt = `Question:` / `Action i:` /
   `Observation i:`. Full loop spec: **D5**.
5. **ReAct** — the full dense-thought method of §1.
6. **ReAct → CoT-SC** — "when ReAct fails to return an answer within given steps, back off to CoT-SC. We set 7 and 5 steps
   for HotpotQA and FEVER respectively." Operationally: back off iff the ReAct run hit the step limit (`hit_step_limit`).
7. **CoT-SC → ReAct** — "when the majority answer among n CoT-SC samples occurs **less than n/2 times** (i.e. internal
   knowledge might not support the task confidently), back off to ReAct." For n = 21: n/2 = 10.5, so back off iff the
   winning vote count ≤ 10; keep CoT-SC at 11.

The four baselines are **ablations of the same exemplars**, not independently written prompts — the same 6 (HotpotQA) /
3 (FEVER) questions in the same order across all four keys.

### Target numbers (`paper/targets.csv`, from Table 1, PaLM-540B)

| Condition | HotpotQA EM | Fever Acc | Source |
|---|---|---|---|
| Standard | 28.7 | 57.1 | Table 1 |
| CoT | 29.4 | 56.3 | Table 1 |
| CoT-SC | 33.4 | 60.4 | Table 1 |
| Act | 25.7 | 58.9 | Table 1 |
| ReAct | 27.4 | 60.9 | Table 1 |
| CoT-SC → ReAct | 34.2 | 64.6 | Table 1 |
| ReAct → CoT-SC | 35.1 | 62.0 | Table 1 |
| Supervised SoTA | 67.5 | 89.5 | Table 1, context only |
| GPT-3 ReAct (davinci-002) | 30.8 | — | App. A.1 Table 5, 500 random val questions |

Table 1 fn a: "HotpotQA EM is 27.1, 28.9, 33.8 for Standard, CoT, CoT-SC in Wang et al. (2022b)."
Table 5's PaLM-540B ReAct HotpotQA is **29.4**, not Table 1's 27.4; the paper does not reconcile them (Table 5 is a
500-question subset). `reference/README.md` gives yet another set for the same setup: GPT-3 davinci-002 HotpotQA **30.4**
and FEVER **54**. The FEVER notebook's own stored output ends `270 500 0.54 4.1496...` = **54.0% EM on 500 questions**,
matching the README. Use `targets.csv` (= Table 1) as the reproduction target and treat 30.4 / 54.0 as the realistic
davinci-002 reference points.

Claims to test [paper §3.3]: ReAct > Act on both tasks; ReAct > CoT on FEVER (60.9 vs 56.3) and slightly below on HotpotQA
(27.4 vs 29.4); CoT-SC > CoT on both; each combination beats every single method on both; and the combinations, quoting §3.3 exactly,
"**reaching** CoT-SC performance with 21 samples using merely 3-5 samples" (Fig. 2).

---

## 5. Prompting details

### The HotpotQA instruction header — verbatim

`hotpotqa.ipynb:77-82`, character for character. **This is the single most important code-only literal in the repo**; it
appears nowhere in the paper (`grep -c "Solve a question answering task"` over the pdfminer-extracted paper text = **0**;
App. C.1 starts directly at `Question`).

```text
Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
```

Whitespace, stated explicitly because a fenced block cannot show it:

* Line 1 ends with `three types:` **followed by exactly one space**, then `\n`. Do not strip it.
* The block ends with `Here are some examples.` + **exactly one** `\n` and nothing else (the notebook's `"""` closes on
  `:82`, immediately after that newline).
* Total length **521** characters. Its Python repr begins
  `'Solve a question answering task ... three types: \n(1) Search[entity], ...'` and ends `...Here are some examples.\n'`.
* Because `webthink_simple6` **starts** with a bare `\n`, `instruction + webthink_simple6` (`hotpotqa.ipynb:83`) yields
  exactly one blank line between `Here are some examples.` and the first `Question:`.
* `webact_simple6` has **no** leading `\n`, so `instruction + webact_simple6` yields **no** blank line there. Reproduce both.

Two facts carried only by the three numbered lines: the action names are **capitalised** (`Search` / `Lookup` / `Finish`)
where [paper §3.1] writes them lowercase; and the header claims search "returns the first paragraph", which contradicts both
[paper §3.1] ("first 5 sentences") and `wikienv.py:87` (`' '.join(sentences[:5])`). **Reproduce it wrong, verbatim.**

### Table

| Detail | Value | Status |
|---|---|---|
| Exemplars, HotpotQA | 6 | **stated in paper** §3.2; fn 2 "We find more examples do not improve performance." Verified: 6 `Question:` blocks in each `*_simple6` key |
| Exemplars, FEVER | 3 | **stated in paper** §3.2. Verified: 3 `Claim:` blocks in each `*_simple3` key |
| Exemplar selection | random from the train split, hand-written trajectories | **stated in paper** §3.2 (the chosen items exist only in the JSON) |
| Prompt keys, used | `webthink_simple6` / `webact_simple6` / `cotqa_simple6` / `webqa_simple6`; `webthink_simple3` / `webact_simple3` / `cotqa_simple3` / `webqa_simple3` | **must come from reference code** — `hotpotqa.ipynb:76`, `FEVER.ipynb:76` (ReAct only); the other three per file are matched against paper App. C.1/C.2 blocks (the paper calls Standard "Original") |
| Prompt keys, **unused legacy** | `prompts_naive.json` has **8** top-level keys: the 4 above plus `webthink_simple`, `webthink_simple_3`, `cotqa_simple`, `webqa_simple`. `fever.json` has exactly 4. **Never load a legacy key.** `webthink_simple_3` (underscore, HotpotQA file) is one character from FEVER's `webthink_simple3` and loads silently. Select by exact string, never prefix or glob; assert *membership* of the 4 required keys, never key-set equality | **code-only** — key dumps of both JSONs |
| HotpotQA instruction header | the 5-line block above. **Scope: HotpotQA `react` and `act` only — never `standard`/`cot`/`cotsc`, never FEVER (D1)** | **must come from reference code** — `hotpotqa.ipynb:77-82`; the scope rule is ours (D1) |
| FEVER instruction header | `Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. ` (trailing space) | **stated in paper** App. C.2. Every `fever.json` value carries the header **inline** — as the literal first line in `webqa_simple3` and `cotqa_simple3`, and after a leading `\n` in `webact_simple3` and `webthink_simple3` (see the whitespace table: leading `\n` = 1 for both). So FEVER never prepends anything (`FEVER.ipynb:76` uses `prompt_dict['webthink_simple3']` bare) |
| Model | `text-davinci-002` for the released code; PaLM-540B for Table 1. **Ours: `gpt-4o-mini` (`.env` `MODEL`) for every condition — see the price table below (D16)** | **both** — `hotpotqa.ipynb:23`; paper Table 1 / App. A.1; our model is ours |
| Greedy decoding | ReAct/Act/Standard/CoT are greedy | **both.** The paper states greedy decoding in three places: Table 3 caption (ALFWorld), Table 5 caption (GPT-3), **and §3.3 footnote 4, p.6** — "We suspect that this could be due to the sub-optimal greedy decoding procedure, and future work using better decoding (e.g. beam search) might help address this issue", hanging off observation **B** of the §3.3 "ReAct vs. CoT" error analysis, i.e. the PaLM-540B HotpotQA ReAct row of Table 1. What is **code-only** is the concrete API setting `temperature=0` (`hotpotqa.ipynb:25`, `FEVER.ipynb:25`) |
| temperature, CoT-SC | `0.7`, 21 samples, majority answer | **stated in paper** §3.2. **Not implemented anywhere in `reference/`** — `grep -n temperature reference/FEVER.ipynb` returns `:25` (`temperature=0`) and `:2784`, the latter inside a stored Wikipedia observation about global average temperature, not code |
| max_tokens | `100` per call | **must come from reference code** — `hotpotqa.ipynb:26`. No token budget appears in the paper |
| top_p / frequency_penalty / presence_penalty | `1` / `0.0` / `0.0` | **must come from reference code** — `hotpotqa.ipynb:27-29` |
| Stop string, main call | `[f"\nObservation {i}:"]`, rebuilt each step, no trailing space | **must come from reference code** — `hotpotqa.ipynb:93`, `FEVER.ipynb:86` |
| Stop string, retry call | `["\n"]` — **spelled `[f"\n"]` in the source** (f-string, no placeholders; identical value). `llm`'s default at `:21` really is `["\n"]` | **must come from reference code** — `hotpotqa.ipynb:101`, `:21` |
| Call structure | one call per step returning `Thought i` **and** `Action i` together; a second call only on parse failure | **must come from reference code** — `hotpotqa.ipynb:93-101` |
| Prompt continuation prefix | `f"Thought {i}:"` (no trailing space) on the main call; `f"Thought {i}: {thought}\nAction {i}:"` on the retry; `f"Action {i}:"` for Act (D5) | **must come from reference code** — `hotpotqa.ipynb:93, 101`; Act is ours |
| Action case fix | `action[0].lower() + action[1:]` — first character only | **must come from reference code** — `hotpotqa.ipynb:102` |
| Step limit | 7 for both tasks | **must come from reference code** — `hotpotqa.ipynb:91`, `FEVER.ipynb:84`; paper §3.2 says 5 for FEVER |
| Timeout behaviour | forced `finish[]`, scored as a normal wrong answer | **must come from reference code** — `hotpotqa.ipynb:111` |
| Answer parsing, Standard / CoT / CoT-SC | **decided by us — see D2, D3, D4 in "Our inventions".** Not in the paper, not in `reference/` | **our invention**, logged under "Our inventions" |
| Act loop | **decided by us — see D5 in "Our inventions".** Neither notebook implements Act | **our invention**, logged under "Our inventions" |
| Vote comparison, CoT-SC | vote over `normalize_answer(pred)`; deterministic lowest-first-sample-index tie-break (D4) | **our invention**, logged under "Our inventions" |
| Eval subset | 500 | **stated in paper** App. A.1 (for the GPT-3 table only); the index construction is code-only |

### Decoding parameters per condition (the artifact `08-verify.md:15` asks for)

`08-verify.md:15` requires evidence that "Temperature and max_tokens are identical across conditions except CoT-SC". This is
the table to check the code against; CLAUDE.md rule 4 is the policy, this is its per-call expansion. **One model for all six
phase-1/2 rows** (`.env` `MODEL`); the seventh row is phase 3's deliberate second model (D27).

| call site | temperature | max_tokens | `stop` | `n` (requests) |
|---|---|---|---|---|
| `standard` (D2) | 0 | 100 | `["\n"]` | 1 |
| `cot` (D3) | 0 | 100 | `["\nQuestion:"]` (HotpotQA) / `["\nClaim:"]` (FEVER) | 1 |
| `cotsc` (D4) | **0.7** | 100 | same as `cot` | **21 independent requests**, `sample_index` 0–20 in the cache key |
| `act` (D5) | 0 | 100 | `[f"\nObservation {i}:"]`, rebuilt each step | 1 per step |
| `react`, main call | 0 | 100 | `[f"\nObservation {i}:"]`, rebuilt each step | 1 per step |
| `react`, **parse-failure retry call** | 0 | 100 | `["\n"]` (source spells it `[f"\n"]`) | 1, only on a bad call |
| **phase 3, local backend** (`12-evaluate-serve.md:3`): prompted-3B and fine-tuned 3B (D27) | 0 | 100 (`max_new_tokens`) | identical to the phase-1 condition it reproduces (`react` unless stated) | 1 per step |

The phase-3 row is reached through the **same `complete()` in `src/llm.py`** — `--model` points at a merged adapter served by
vLLM (or Ollama) and the call goes through the same disk cache and the same `results/calls.csv` append, so CLAUDE.md rules 4
and 5 hold for the student exactly as for `gpt-4o-mini`. Its price-table entry is `0.0 / 0.0` by design (self-hosted
inference has no per-token price) and is an **explicit** table row, not a fallback: an unpriced model is
refused, not costed at zero (see the price table below).

`top_p = 1`, `frequency_penalty = 0.0`, `presence_penalty = 0.0` everywhere (`hotpotqa.ipynb:27-29`). CoT-SC's `temperature`
is the **only** cell that differs across conditions; every `max_tokens` is 100.

**The `n` column is n independent requests, never the provider's native `n=21` (D4, confirmed D36).** One request asking for
21 completions would be cheaper — it bills the ~6 KB prompt once instead of 21 times — and it was considered. It is still
wrong here, for three reasons that outrank the saving: the cache key's `sample_index` (D28) and `calls.csv`'s `sample_index`
column both assume **one row and one cache entry per sample**, so a single 21-sample row makes both meaningless and breaks
C3's "21 `calls.csv` rows on a cold cache"; a CoT-SC question interrupted part-way could not resume free, because a native-`n`
response caches as one indivisible blob; and D4's effect estimate already budgets ≈21× CoT, so the extra prompt tokens are
the cost that was planned for, not an overrun. Settled — not to be re-derived at the next cost review.

### Model and price table (D16) — what makes CLAUDE.md rule 8 computable

The paper's models are `text-davinci-002` (released code) and PaLM-540B (Table 1). **Ours is `gpt-4o-mini`** — set by
`MODEL` in `.env`, one model for every condition per CLAUDE.md rule 4.

The price table lives in **`src/llm.py`** as a module-level dict keyed by model name (`04-llm-client.md:5`: "estimated cost
from a small price table in the file"). Rates as of **2026-09-15**, USD per 1M tokens:

| model | input $/1M | output $/1M |
|---|---|---|
| `gpt-4o-mini` | 0.15 | 0.60 |
| `qwen2.5-3b-prompted`, `qwen2.5-3b-react-lora`, `qwen2.5-3b-react-lora-q4` (phase 3, served locally) | 0.0 | 0.0 |

* `prompt_tokens` / `completion_tokens` come from the **provider response's `usage` field**, never from a local tokenizer
  estimate — a mis-estimate would silently mis-gate rule 8.
* A model **missing from the table is refused**: `_check_budget` raises `BudgetExceeded` before the request, naming the
  model. Costing an unpriced model `0.0` — what this bullet used to specify — zeroes the pre-call estimate, every logged
  `cost_usd` and therefore `_spend_so_far()`, so CLAUDE.md rule 8's ceiling can never fire: one typo in `.env`'s `MODEL`
  buys a 25x-priced model (14,000 calls ≈ $66.50 on `gpt-4o` against $3.99 on `gpt-4o-mini`) while the ledger reads $0.00.
  A guard that cannot price a call must not authorise it. **Free is explicit, not the default** — see the next bullet.
* **These rates are a local constant and can go stale.** Re-check them against the provider's pricing page before any cost
  total is quoted in `README.md`.
* The local-backend entry is `0.0 / 0.0` **deliberately**, so a phase-3 run still writes `calls.csv` rows and a
  `results.csv` `total_cost` of `0.0`. Because absence is now a refusal, each locally served model needs its **own explicit
  row** in `PRICES`: `qwen2.5-3b-prompted`, `qwen2.5-3b-react-lora` and `qwen2.5-3b-react-lora-q4` (the three `{model}`
  slots of section 6's naming table) all ship at `0.0 / 0.0`. Serving a fourth name means adding a fourth row — one line,
  and the refusal message says so. A `0.0 / 0.0` model still logs one warning per call, which is the only remaining
  warn-and-charge-zero path.
* C3 pins the arithmetic: a call with known `usage` counts produces a known `cost_usd`.
* **The model string is pinned in `.env` (`MODEL`) and must be written into every `results/` row** — the `model` column of
  `results/calls.csv` (one row per real call, rule 6) and the `model` column of `results/results.csv` (one row per
  `(task, condition, model, n)`, rule 6), and it is also the first field of the LLM cache key (D28). `src/llm.py` reads it
  once at import (`MODEL = os.getenv("MODEL", "gpt-4o-mini")`) and `complete()` takes no model argument, so a mid-project
  model change can never silently blend into one table: the new model's rows carry the new string, and its calls **miss** the
  cache rather than being served from the old model's completions. A `results.csv` row without a `model` column is not a
  result.
* **Budget guard (CLAUDE.md rule 8), in code.** `MAX_SPEND_USD` is read from `.env` (default `5.00`, recorded in
  `.env.example`). Cumulative spend is read back from `results/calls.csv` itself — the log **is** the ledger, so the total
  survives a process restart and a resumed run, which a per-process counter does not. `src/llm.py` raises `BudgetExceeded`
  **before** issuing the call that would cross the ceiling, naming spend so far, the ceiling and the attempted call. Cache
  hits issue no request, write no row, and therefore cost nothing. The pre-call estimate — the one number that cannot come
  from `usage`, because the call has not happened — is `len(system + prompt) / 4` characters-per-token on the input side and
  the full `max_tokens` on the output side (its hard ceiling, so the estimate never runs low on the half we control). It
  gates only; it is never logged.

### Temperature 0 is not determinism — the cache is (README Limitations)

`temperature=0` on a modern chat API is **not** a guarantee of identical output: the provider may re-route a request to a
different serving stack or fleet revision, batching is nondeterministic on GPU, and the model itself can be updated behind a
floating alias. The reference's `text-davinci-002` had the same property; the paper does not claim determinism either, only
"greedy decoding" (§3.3 fn 4).

**Our reproducibility across reruns therefore comes from the disk cache, not from the provider.** Every rerun of a populated
cache replays the same bytes for the same `(model, system_message, prompt, stop, temperature, max_tokens, sample_index)` key
(D28) and so reproduces EM exactly; a rerun that *misses* the cache — a new model string, an edited system message, a deleted
`data/cache/llm/` — is a fresh sample and may differ, at temperature 0 as well as at 0.7. Two consequences to state rather
than discover:

* A number in `README.md` is reproducible **given `data/cache/llm/`**, which CLAUDE.md rule 9 keeps out of git. Anyone
  re-running from a clean clone re-samples the model and can land on a different EM.
* This belongs in the README's Limitations section (`12-evaluate-serve.md:14`), next to D20's fine-tuning caveats.

**Scope of CLAUDE.md rule 4's "one model for all conditions" (D27).** It binds **phases 1–2**: the seven Table-1 conditions
all run on `gpt-4o-mini`, which is what makes the seven-condition comparison fair. **Phase 3 deliberately introduces a second
model** — the `Qwen2.5-3B-Instruct` student, evaluated both prompted with 6 exemplars and fine-tuned
(`12-evaluate-serve.md:8`). That is the §3.3 experiment, not a violation of the rule; a phase-3 model is never mixed into a
phase-1 comparison row, and its decoding settings, cache and cost log are the same ones (row 7 above).

### Exemplar whitespace — measured, not remembered

Counts are of the raw JSON string values (`prompts/*.json`, byte-identical to `reference/prompts/*.json`).

| key | file | leading `\n` | trailing `\n` | `\n\n` occurrences | own header inline? |
|---|---|---|---|---|---|
| `webqa_simple6` | prompts_naive | 0 | 1 | 0 | no |
| `cotqa_simple6` | prompts_naive | 0 | 1 | 0 | no |
| `webact_simple6` | prompts_naive | 0 | **2** | 1 | no |
| `webthink_simple6` | prompts_naive | **1** | 1 | 0 | no |
| `webqa_simple3` | fever | 0 | 1 | 2 | **yes** |
| `cotqa_simple3` | fever | 0 | 1 | 2 | **yes** |
| `webact_simple3` | fever | **1** | **2** | 3 | **yes** |
| `webthink_simple3` | fever | **1** | **2** | 3 | **yes** |

Consequences, all load-bearing:

* **HotpotQA exemplars are not blank-line separated; FEVER exemplars are.** The live question needs a different join per task.
* The driver does `prompt += question + "\n"` unchanged for both tasks (`hotpotqa.ipynb:89`, `FEVER.ipynb:82`). Because
  `webthink_simple6` ends with a **single** `\n` while `webthink_simple3` and `webact_simple3` end with `\n\n`, the ReAct
  prompt is `...Finish[yes]\nQuestion: <live>\n` on HotpotQA (no blank line) and `...Finish[NOT ENOUGH INFO]\n\nClaim: <live>\n`
  on FEVER (blank line present). Reproduce both.
* `webact_simple6` is the asymmetric one: **no** leading `\n`, but it **ends** with `\n\n`. With the instruction header
  prepended it therefore has no blank line before the first `Question:` and one blank line before the live `Question:`.
* For FEVER Standard/CoT the value ends with a single `\n` and exemplars are `\n\n`-separated, so D2/D3 prepend an extra
  `\n` to the live `Claim:` to keep the blank-line rhythm.

### Exemplar typos to preserve (CLAUDE.md rule 2)

* `cotqa_simple3` and `webqa_simple3` each contain the literal `Answer:REFUTES` — **no space** (1 occurrence each). A parser
  splitting on `"Answer: "` silently loses it. Split on `"Answer:"` (D3).
* FEVER exemplar 3's claim ends `...in 2003.?`.
* HotpotQA exemplar 6 answers `Yes` in Standard/CoT but `Finish[yes]` in Act/ReAct.

---

## 6. Component checklist

Dependency order. Every row names the module, the callable(s) step 08 must cite (`prompts/claude-code/08-verify.md:5`), the
test file required before the component is used in any run (CLAUDE.md rule 7), and a test with a **concrete input and a
concrete expected output**.

| # | Component | Our file | Entry point | Test file | Paper section | Test (input → expected output) |
|---|---|---|---|---|---|---|
| **C1** | Wikipedia environment **+ the fetch layer** (disk cache, retry, User-Agent — D18) | `src/wiki_env.py` | `WikiEnv.reset`, `WikiEnv.step`, `WikiEnv.search_step`, **`WikiEnv._fetch`** | `tests/test_wiki_env.py` | §3.1 Action Space; behaviour from `wikienv.py:44-160` | **(n) URL construction (owns EXPECTED.md L31):** `search[Colorado orogeny]` builds exactly `https://en.wikipedia.org/w/index.php?search=Colorado+orogeny` (spaces → `+`, asserted on the URL handed to `_fetch`, not on the response). (a) `search[Colorado orogeny]` against `tests/fixtures/colorado_orogeny_hit.html` (HTTP mocked) → observation starting `The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.` and containing the doubled period `...larger Yavapai orogeny.. The Colorado orogen...`, 5 `'. '`-fragments, no trailing newline. (b) `lookup[New Mexico]` on that same page → `(Result 1 / 2) It is recorded in the Colorado orogen, a >500-km-wide belt of oceanic arc rock that extends southward into New Mexico.`, then `(Result 2 / 2) ...`, then exactly `No more results.\n`. Also on that page: `lookup[High Plains]` → `(Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.` then `No more results.\n`; and `lookup[elevation]` → `No more results.\n` **immediately** (0 matches, distinguishing the empty-list path from exhaustion). (c) `tests/fixtures/colorado_orogenyy_similar.html` — **recorded** with the D18 User-Agent against `?search=Colorado+orogenyy`, 20 `mw-search-result-heading` divs; the expected literal is the one that *this file* produces, not a live value (D19): `Could not find Colorado orogenyy. Similar: ['Colorado orogeny', 'Laramide orogeny', 'Sevier orogeny', 'Colorado Mineral Belt', 'Wyoming Craton'].` (d) `tests/fixtures/nonexistent_miss.html` (entity `Qwertzuiop Zzyzxian Orogeny`) has **0** result divs → D10/D23 zero-result path: the observation is byte-equal to `Could not find Qwertzuiop Zzyzxian Orogeny. Similar: [].` — the ordinary miss literal with an empty list, **not** the raw `There were no results matching the query` text — and `page` / `lookup_keyword` / `lookup_list` / `lookup_cnt` are unchanged by the call (miss semantics: a `lookup` issued straight after still sees the page from the previous successful search). (e) after `reset()`, `lookup[anything]` → `No more results.\n` (all six fields cleared, `construct_lookup_list` returns `[]` when `page is None`). **The cross-question half of this property — a page fetched for question *k* being invisible to question *k+1* — is asserted in C8, not here (D21):** C1 has no runner, so a test that calls `reset()` itself would pass trivially while the real 500-question loop never reset. (f) `finish[Richard Nixon]` → `done is True`, `info["answer"] == "Richard Nixon"`, observation byte-equal to `"Episode finished, reward = 0\n"` (always `0`; the runner never rewrites it — see §2). (g) `blah` → `Invalid action: blah`, `done is False`. **(i) cache:** `_fetch(u)` twice for the same `u` → the second call makes **0** network calls and returns byte-identical text, and one file appears at `data/cache/wiki/<sha256(u)>.json`. **(j) User-Agent:** the request `_fetch` issues carries header `User-Agent: react-langgraph-repro/0.1 (research reproduction; contact via repo)`; a stubbed transport that 403s a bare UA and 200s that UA yields an observation, not an exception. **(k) retry:** a stubbed transport raising `Timeout` 3× then returning the `colorado_orogeny_hit.html` body → one observation, exactly 4 attempts; 10 consecutive timeouts → one raised error, not `None` (unlike `hotpotqa.ipynb:46-52`). **(l) `clean_str` guard (MIN1):** a one-block stub page whose text is `path C:\xyz here` (a stray backslash escape) → the observation contains that block **unchanged** and no exception escapes; the same page through the bare `wikienv.py:10-11` body raises `UnicodeDecodeError`, which is the behaviour being guarded. **(m) disambiguation depth 1 (D8/MIN2):** a stub whose first fetch returns a `may refer to:` block and whose second fetch (for `[entity]`) returns **another** `may refer to:` block → exactly **2** fetches, and the observation is that second page's own result, not a third search. (h) one `slow`-marked live `search[Colorado orogeny]`, asserting only the stable parts (`len(result_divs) == 20` and `titles[0] == 'Colorado orogeny'` for the `orogenyy` miss) |
| **C2** | Data loaders, eval-index sampler, SQuAD normalization, EM/F1, FEVER label check | `src/data.py` | `load_hotpotqa`, `load_fever`, `eval_indices`, `eval_indices_for`, `normalize_answer`, `exact_match`, `f1` | `tests/test_data.py` | §3.1 Domains; §3.3 (EM); normalization from `wrappers.py:42-78` | `eval_indices(7405)[:5] == [3687, 6238, 5388, 3522, 3824]`; `eval_indices_for("fever") == eval_indices(7405)` and `max(eval_indices_for("fever")) == 7390` (the FEVER sampler must never see `len(load_fever())`); `len(load_hotpotqa()) == 7405` and each item is a 3-tuple with `type in {"bridge","comparison"}`, `Counter(types) == {"bridge":5918,"comparison":1487}`; `len(load_fever()) == 9999`; `exact_match("The Chief of Protocol","chief of protocol") == 1`; `f1("Richard Milhous Nixon","Richard Nixon")` is a **float** strictly in (0,1) (`== 0.8`); `f1("yes","no") == 0.0` (yes/no short-circuit); FEVER `"supports."` vs gold `SUPPORTS` → 1; FEVER `"probably true"` → 0 |
| **C3** | LLM client + cache + cost log + price table | `src/llm.py` | `complete(prompt, stop, temperature=0, max_tokens=100, n=1) -> list[str]`, `PRICES` | `tests/test_llm.py` | decoding params code-only (`hotpotqa.ipynb:23-30`); CoT-SC temp 0.7 / n=21 from §3.2 | a canned response `"I need to search X.\nAction 1: Search[X]\nObservation 1: leak"` with `stop=["\nObservation 1:"]` → returns exactly `"I need to search X.\nAction 1: Search[X]"` (stop excluded, cut at first occurrence); calling `complete` twice with identical args → second call makes **0** network calls and appends **0** rows to `results/calls.csv`; `complete(p, n=21, temperature=0.7)` → 21 cache entries whose keys differ only in `sample_index`, and 21 `calls.csv` rows on a cold cache, 0 on a warm one; **cache record shape (D14):** after that same canned call, the stored entry has `raw == "I need to search X.\nAction 1: Search[X]\nObservation 1: leak"` and `text == "I need to search X.\nAction 1: Search[X]"` — the `raw` field is exactly what `08-verify.md:10` prints as "a cached call where the raw output was cut", with no fresh network call; **cost (D16):** a stubbed response with `usage = {prompt_tokens: 1000, completion_tokens: 100}` on `gpt-4o-mini` → the appended `calls.csv` row has `cost_usd == 0.00021` (1000/1e6×0.15 + 100/1e6×0.60), and an unknown model name → `cost_usd == 0.0` plus one warning, no exception; **cache key (D28):** two calls identical in every argument but issued under a different `system_message` **miss** each other — 2 network calls, 2 `calls.csv` rows, 2 cache files — while a repeat under the same system message is served from cache |
| **C3b** | Exemplar/prompt loader (CLAUDE.md rule 2 owner; `08-verify.md:9` byte-identity check) | `src/prompts.py` | `load_exemplars(task, condition)`, `build_prompt(task, condition, question, scratchpad, step)` | `tests/test_prompts.py` | §3.2 (exemplar counts); keys from `hotpotqa.ipynb:76`, `FEVER.ipynb:76` | all 8 required keys resolve and each returned string is **byte-equal** to the same key in `reference/prompts/`; loading `webthink_simple_3` or any other legacy key raises; `webthink_simple6` starts with `"\n"` and contains **0** `"\n\n"`; `webact_simple6` ends with `"\n\n"`; `webthink_simple3` ends with `"\n\n"`; `cotqa_simple3` contains the literal `Answer:REFUTES`; `cotqa_simple6` contains `"Thought: Let's think step by step. "` exactly 6 times and `cotqa_simple3` contains `"Let's think step by step"` exactly 0 times; `build_prompt("hotpotqa","react",...)` starts with the 521-char instruction header and `build_prompt("fever",*,...)` never does |
| **C4** | ReAct graph: explicit LangGraph `StateGraph`, `think_act` → `execute` → pure router → `force_finish` / `END` / `think_act` (D22) | `src/graph_react.py` | `build_graph(condition, env)` (D21 — the env is a closure argument, never a state field), `think_act`, `execute`, **`force_finish`**, `route_after_execute` | `tests/test_graph_react.py` | §2 (alternation), §3.2 (ReAct prompting); loop from `hotpotqa.ipynb:85-115`; state from `05-react-graph.md:3` | fake-LLM + fake-env: **(0)** the initial state has the 11 fields of `05-react-graph.md:3` (including `action`, corrections-log entry 8: the
action is carried from `think_act` to `execute` in the state and **never** recovered by re-splitting the scratchpad —
`Action 1: Search[Action 1: The Movie]` must reach the env as `search[Action 1: The Movie]`, a label-echoing retry
`" Action 1: Search[X]"` as `action 1: Search[X]`) and `step == 1`, the first prompt ends `"Thought 1:"` (never `"Thought 0:"`), and a 7-step episode makes exactly **7** model steps with `step == 8` when `route_after_execute` runs (the pure router that replaces `05-react-graph.md:9`'s state-writing edge, D22). (a) scripted 2-step episode ends with `answer == "Richard Nixon"` and a scratchpad byte-equal to `"Thought 1: ...\nAction 1: Search[x]\nObservation 1: ...\nThought 2: ...\nAction 2: Finish[Richard Nixon]\nObservation 2: ...\n"`. (b) completion with **zero** `"\nAction 1: "` separators → `n_badcalls == 1`, `n_calls == 3` after two steps, second call used `stop=["\n"]`. (c) completion with **two** `"\nAction 1: "` separators (`"I need X.\nAction 1: Search[A]\nAction 1: Search[B]"`) → also `n_badcalls == 1` and a retry: the strict two-way unpack raises `ValueError: too many values to unpack`, and `split(sep, 1)` / `str.partition` would **not** retry and would execute `Search[A]\nAction 1: Search[B]`. (d) 7 non-finishing steps → the `force_finish` **node** ran, and the **final state** (not an intermediate one) has `hit_step_limit is True`, `answer == ""`, `done is True`; env received `finish[]`; `n_steps == 7`; `route_after_execute` returned `"force_finish"` and wrote nothing itself. **Trajectory end (D24):** the scratchpad ends exactly at `Observation 7: <obs>\n` — it contains **no** `Thought 8:`, `Action 8:` or `Observation 8:` line and does **not** end with `Episode finished, reward = 0\n`. A router that sets `hit_step_limit` instead of a node fails this row on langgraph 1.2.11 (the write is discarded). (e) LLM returns action `Search[Milhouse]` → env receives `search[Milhouse]` (first char only). (f) LLM returns `""` → env receives `""` → `Invalid action: `, episode continues (D9). Then 3 real HotpotQA trajectories eyeballed |
| **C5** | Act variant | `src/graph_react.py` | `build_graph(condition="act", env=...)` (same D21/D22 shape, including `force_finish`) | `tests/test_graph_act.py` | §3.2 Baselines (c); spec **D5** | fake-LLM: the continuation is exactly `f"Action {i}:"`, `stop == [f"\nObservation {i}:"]`; the scratchpad contains **zero** occurrences of `"Thought"`; each step serializes as `f"Action {i}: {action}\nObservation {i}: {obs}\n"`; a malformed completion triggers **no** retry — Act has no parse-retry path — but it **does** count bad calls, so `n_badcalls == 0` is **not** asserted unconditionally (D17); **multi-action completion:** `"Search[A]\nAction 2: Search[B]"` → the env receives exactly `search[A]` (first line only) and `n_badcalls == 0`; a completion whose first line is empty or carries no `[` → the env receives that first line and `n_badcalls == 1`; the same scripted episode resolves to the same answer as C4(a); HotpotQA Act prompt starts with the instruction header and has **no** blank line before the first `Question:`, FEVER Act prompt starts with `\nDetermine if there is Observation that SUPPORTS` |
| **C6** | Standard, CoT, CoT-SC | `src/baselines.py` | `standard`, `cot`, `cot_sc`, `parse_answer`, `majority_vote` | `tests/test_baselines.py` | §3.2 Baselines (a)(b) + CoT-SC; specs **D2/D3/D4** | `parse_answer` replays all 6 HotpotQA and all 3 FEVER exemplar tails as fake completions and recovers the gold answer each time, **including** `"Answer:REFUTES"` → `"REFUTES"` (splitting on `"Answer: "` fails this); `parse_answer("Thought: ...\nAnswer: yes\nQuestion: next")` → `"yes\nQuestion: next"` (text after the **FIRST** `Answer:`, stripped — **D37**, which retracts D3's "last"; the earlier expectation `"yes"` assumed a first-line cut that no split-and-strip rule performs, and that the stop list makes unreachable); a completion with no `Answer:` → `""` and a bad-call count of 1; `majority_vote` on a canned 21-sample list with 12× `"Richard Nixon"` → `("Richard Nixon", 12, 0)` (winner, `winner_votes`, `empty_samples` — the as-built signature, see "Step 06 as built"); on a 10/10/1 tie the winner is the answer whose first occurrence has the lowest sample index, identical across 100 repeated calls; **empty samples (D13):** 11× `""` + 10× `"Richard Nixon"` → `("Richard Nixon", 10, 11)` — `""` never wins — and `cotsc_to_react` on that row picks the **ReAct** prediction because `10 <= 10`; all 21 empty → `("", 0, 21)`, `em == 0`; **raw-string rule (D4):** `["Richard Nixon.", "richard nixon", "Richard Nixon."]` → the recorded `prediction` is `"Richard Nixon."`, the raw string of the winning key's **lowest** sample index; prompt-shape asserts: HotpotQA Standard prompt ends `"\nQuestion: <q>\nAnswer:"` **and carries no instruction header**, **HotpotQA CoT prompt likewise carries no instruction header and ends `"\nQuestion: <q>\nThought:"` (D1)**, FEVER CoT prompt ends `"\n\nClaim: <c>\nThought:"` |
| **C7** | The two combination rules | `src/combine.py` | `react_to_cotsc`, `cotsc_to_react`, `write_results_row` | `tests/test_combine.py` | §3.2 "Combining Internal and External Knowledge" A and B; spec **D6** | named fixture pair `tests/fixtures/combine_react.jsonl` / `tests/fixtures/combine_cotsc.jsonl`, 5 rows each, joined on `idx` — idx 0 (`hit_step_limit false`, `votes 15`), 1 (`true`, `15`), 2 (`false`, `10`), 3 (`true`, `10`), 4 (`false`, `11`), ReAct predictions `R0..R4`, CoT-SC predictions `S0..S4`. Expected picks, literally: `react_to_cotsc` → `[R0, S1, R2, S3, R4]` with `source` `[react, cotsc, react, cotsc, react]`; `cotsc_to_react` → `[S0, S1, R2, R3, S4]` with `source` `[cotsc, cotsc, react, react, cotsc]`. Each output line also inherits `n_steps` / `hit_step_limit` from the source it picked (D15). Neither function issues an LLM call (`calls.csv` row count unchanged), and `total_cost == 0.0` |
| **C8** | Runner — **and the owner of the `WikiEnv` (D21)** | `src/run.py` | `main`, `run_one(idx, question, task, condition, env)` | `tests/test_run.py` | §3.1/§3.2 setup; eval indices from `hotpotqa.ipynb:126-132` | a 5-question run writes 5 JSONL lines to `runs/hotpotqa_react_{model}.jsonl` whose `idx` list equals `eval_indices(7405)[:5]`, each line carrying every field of the schema below; appends exactly **one** row to `results/results.csv` with the header below; a rerun is fully cache-served (cost delta 0.0) and byte-identical. **Env lifecycle (D21):** one `WikiEnv` is constructed for the whole run and passed to `build_graph(condition, env)`; `run_one` calls `env.reset()` **once per question**, and a spy on `reset` records exactly one call per question, in order. **Cross-question isolation (moved here from C1(e)):** a scripted two-question run where question 1 does `search[Colorado orogeny]` and question 2 opens with `lookup[New Mexico]` → question 2 gets `No more results.\n`, proving no page leaked. **Trajectory (D25):** one line per condition asserted non-empty and of the right shape — `standard`/`cot` a string starting `Question: ` (or `Claim: `), `cotsc` an object with `winner` / `samples` (21) / `votes`, `act`/`react` a scratchpad ending `Observation {n}: ...\n`, combination lines byte-equal to the source line named by `source`. **Rule 8 (CLAUDE.md:32-33) — both triggers:** every step-07 run is 500 questions × 2 tasks × 5 conditions, so it trips the ">100 questions" trigger regardless of cost and **requires an explicit ask before launch**, independent of the $5 estimate |
| **C9** | Results table + claims | `src/report.py` → `README.md`, `results/results.csv` | `make_table` | `tests/test_results_table.py` | Table 1 / `paper/targets.csv`; claims from §3.3 and `07-full-runs.md:5` | given a 3-row `results.csv` fixture (`react` EM 0.31, `act` EM 0.22, `cot` EM 0.29), `make_table` emits a markdown table whose ReAct row reads `31.0` and whose claim line 1 (`ReAct > Act on both tasks`) reads `PASS`; flip the fixture to `act` EM 0.35 and the same line reads `FAIL`; and a guard test asserting every number in `README.md` also appears in `results/results.csv` (no hand-typed figures); per-type (bridge/comparison) breakdown present for CoT, Act, ReAct |
| **C10** | Verification pass | `src/verify.py` → `reports/verification.md` | `assert_expected_values()` | `tests/test_expected_md.py` (the caller; the function itself lives in `src/verify.py`, so the row is not circular) | **NOT IN PAPER** — `tests/EXPECTED.md` is a repo artifact, so step 08's verdict column reads NOT IN PAPER for this row | asserts the EXPECTED.md items **not already pinned by C1/C2/C4**, and cross-references those that are. Asserted here: `eval_indices(7405)[:5]` equals L9's five values; `len(load_hotpotqa()) == 7405` with keys `question, answer, type` (L14); `data/hotpot_train_v1.1_simplified.json` exists (L15); `len(load_fever()) == 9999` with the five keys (L16); the 8 prompt keys of L19-21 are present and byte-identical to `reference/prompts/` (L19-21); `STEP_LIMIT == 7` (L24); `stop == f"\nObservation {i}:"` (L26); the forced `finish[]` scores 0 (L24-25); `normalize_answer` order and FEVER label restriction (L38-40). **Cross-referenced, not re-asserted here** (each named with its owner, so the coverage claim is exact): L10-11 FEVER sampler bound → C2 (`eval_indices_for("fever") == eval_indices(7405)`, `max(...) == 7390`); L27 bad-call counter → C4(b)(c); L28 first-character lowercasing → C4(e); L31 search URL → C1(n); L32 5-sentence rule, L33 `Could not find` literal, L34 lookup literals, L35 reward 0 → C1(a)(c)(b)(f). `reports/verification.md` then carries one row per component with verdict MATCH / DIFFERS / NOT IN PAPER, and no unexplained DIFFERS vs `tests/EXPECTED.md` or `paper/targets.csv` |
| **C11** | Fine-tuning trajectory collection | `src/collect.py` (writes `data/sft/{react,act,cot}_train.jsonl`) | `collect_trajectories(condition, n_target=3000)`, `keep_trajectory(record) -> bool` (the validator) | `tests/test_sft_data.py` | §3.2 "Finetuning" (method); §3.3 (results); `10-collect-trajectories.md` | **runs before the collection run (CLAUDE.md rule 7) against a hand-written 3-line `tests/fixtures/sft_mini.jsonl`** — two well-formed `em == 1` records (one `react`, one `cot`) and one `em == 0` record → the validator rejects exactly 1 and keeps 2; then, once `src/collect.py` has run, the same assertions over the real files: every record in all three files has `em == 1`; the three files' **question sets are identical** (`10-collect-trajectories.md:5`) and questions are unique within each file; each record has `question, text, answer, n_steps, source_idx`; a `react` record's `text` re-parsed by `src/graph_react.py`'s parser yields the same `answer` field; a `cot` record's `text` contains no `Observation`; keep-rate, mean steps and total cost printed. **Rule 8 (CLAUDE.md:32-33) — both triggers:** the collection sweep targets 3,000 kept trajectories over the 90,447-entry train split, so it is far over 100 questions and **requires an explicit ask before launch** as well as at the $5 estimate; `10-collect-trajectories.md:5`'s $15 is a *second, later* checkpoint, not a replacement (see "Disagreements" → Run gates) |
| **C12** | LoRA training + fine-tuned eval | `finetune/train.py`, `finetune/eval.py` → `runs/`, `results/results.csv`, `results/finetuning.csv` | `build_dataset`, `mask_labels`, `train.main`, `eval.main` (two distinct entry points, one per file) | `tests/test_masking.py` — **moved under `tests/` for CLAUDE.md rule 7**; it needs only a tokenizer and one JSONL record, so it runs before any training | §3.2 "Finetuning" (the 3,000-trajectory bootstrap) and §3.3 (the results). App. B.1 gives **the paper's** setting — batch 64, 4,000 steps ReAct/Act, 2,000/1,000 Standard/CoT — of which **only batch 64 is in force here** (`11-train-lora.md:5`); our schedule is 3 epochs ≈ 141 optimizer steps on Qwen2.5-3B-Instruct, so 4,000 steps is context, not our setting (D20) | decoding the **unmasked** label positions of **the single record in `tests/fixtures/sft_one_react.jsonl`** (one `react` trajectory, `em == 1`, 3 steps) yields a string containing no `"Observation"` substring and none of the prompt tokens (`11-train-lora.md:3`); masked positions are all `-100`; **`results/finetuning.csv` carries the 8 columns of `12-evaluate-serve.md:5`** (model, prompted-or-finetuned, exemplars used, EM, mean steps, % hit step limit, tokens/sec, prompt tokens/question) — asserted against `tests/fixtures/finetuning_two_rows.csv` (row 1 `Qwen2.5-3B-Instruct`/prompted/6 exemplars, row 2 the same model/fine-tuned/0 exemplars), and the quantized-GGUF row of `12-evaluate-serve.md:12` is one more row in the same file with the same columns; **rule 6 (D27):** each phase-3 evaluation is a real run and writes its own `runs/hotpotqa_react_{model}.jsonl` (the `{model}` slot of the naming table carries `qwen2.5-3b-prompted`, `qwen2.5-3b-react-lora`, `qwen2.5-3b-react-lora-q4`) **and** appends a normal `results/results.csv` row — `results/finetuning.csv` is the extra summary, not the only record; **rules 4 and 5 (D27):** the local backend is reached through the same `src/llm.py:complete()`, so the cache, `results/calls.csv` and temperature 0 / 100-token decoding apply, with the `0.0` price entry; **rule 8 (CLAUDE.md:32-33):** the 500-question eval (`12-evaluate-serve.md:1`) trips the ">100 questions" trigger and **requires an explicit ask before launch**; the post-quantization re-run (`:12`) is exactly 100, so rule 8's count trigger ("over 100") does not fire and only its $5 cost trigger applies; and the three step-12 claims are each printed PASS/FAIL with numbers — **(1)** fine-tuned ReAct-3B EM > same 3B model prompted with 6 exemplars, **(2)** fine-tuned ReAct EM > fine-tuned CoT EM and > fine-tuned Act EM, **(3)** fine-tuned ReAct-3B prompt tokens/question < prompted version, reported as a % reduction (`12-evaluate-serve.md:8-10`) |

### Canonical names — CLI token → results.csv → targets.csv → run file

Three naming schemes exist in the repo; this is the mapping, so `results/results.csv` can be joined to `paper/targets.csv`.

| `--condition` (`06-baselines-combine.md:7`) | `results.csv` `condition` | `targets.csv` row | run file |
|---|---|---|---|
| `standard` | `standard` | `Standard` | `runs/{task}_standard_{model}.jsonl` |
| `cot` | `cot` | `CoT` | `runs/{task}_cot_{model}.jsonl` |
| `cotsc` | `cotsc` | `CoT-SC` | `runs/{task}_cotsc_{model}.jsonl` |
| `act` | `act` | `Act` | `runs/{task}_act_{model}.jsonl` |
| `react` | `react` | `ReAct` | `runs/{task}_react_{model}.jsonl` |
| — (`src/combine.py` only) | `react_to_cotsc` | `ReAct->CoT-SC` | `runs/{task}_react_to_cotsc_{model}.jsonl` |
| — (`src/combine.py` only) | `cotsc_to_react` | `CoT-SC->ReAct` | `runs/{task}_cotsc_to_react_{model}.jsonl` |
| — (`finetune/eval.py`, phase 3) | `react` with `model = qwen2.5-3b-prompted` | — (no paper row; phase 3 is §3.3, not Table 1) | `runs/hotpotqa_react_qwen2.5-3b-prompted.jsonl` |
| — (`finetune/eval.py`, phase 3) | `react` with `model = qwen2.5-3b-react-lora` | — | `runs/hotpotqa_react_qwen2.5-3b-react-lora.jsonl` |
| — (`finetune/eval.py`, phase 3) | `react` with `model = qwen2.5-3b-react-lora-q4` | — | `runs/hotpotqa_react_qwen2.5-3b-react-lora-q4.jsonl` |

The three phase-3 rows exist because CLAUDE.md rule 6 applies to **every** run: the `{model}` slot already distinguishes
them, so the fine-tuned, prompted-3B and quantized evaluations each write per-question JSONL and a `results.csv` row, and
`results/finetuning.csv` is an additional summary carrying the eight columns of `12-evaluate-serve.md:5` (D27).

`task ∈ {hotpotqa, fever}`. Display names in §4 (`ReAct → CoT-SC`) are for prose only; `targets.csv` uses ASCII `->`.

**`results/results.csv` header**, verbatim (`06-baselines-combine.md:7`):
`task,condition,model,n,metric,mean_steps,pct_hit_step_limit,total_cost`

**`results/calls.csv` header** — **seven** columns, verbatim as written by `src/llm.py` (`CALLS_HEADER`, pinned by a test in
`tests/test_llm.py` against this line). Five are **our column names** for the fields `04-llm-client.md:5` lists in prose
("timestamp, model, prompt_tokens, completion_tokens, estimated cost from a small price table in the file") — the first four
match its words, and **`cost_usd` is our name for "estimated cost"**, a token that does not appear in that file. The other
two are ours:
`timestamp,model,sample_index,prompt_tokens,completion_tokens,cost_usd,cache_hit`

* `sample_index` — `0` outside CoT-SC, `0`–`20` within it. It is what makes one row one **sample** rather than one question
  (D4, D28); without it C3's "21 rows on a cold cache" cannot be checked against the ledger at all. This file is the only
  place the column appears as a CSV column; elsewhere it is described as a cache-key field.
* `cache_hit` — **constant `False` today, by construction**: a cache hit issues no request and writes no row (C3), so every
  row in the file is a real call. It is declared anyway so that a later decision to log hits adds rows rather than changing
  the header under readers. Two things bind any such change: `_spend_so_far()` sums `cost_usd` over **every** row, so a hit
  row must carry `cost_usd == 0.0` or it double-charges rule 8's ledger, and a hit rate must then be computed from
  `cache_hit`, never from the row count.
(The `results/results.csv` header above **is** verbatim: `06-baselines-combine.md:7` literally lists
`(task, condition, model, n, metric, mean_steps, pct_hit_step_limit, total_cost)`.)

**Per-question JSONL schema (C8).** CLAUDE.md rule 6 lists the **minimum**, not the maximum. Every line carries:

`idx, question, gold, prediction, em, n_steps, hit_step_limit, trajectory` (rule 6) **plus**
`f1, condition, winner_votes, empty_samples, n_calls, n_badcalls, cost` — and `source` on combination lines only.
**Field names are D39/D41's, as built** (`winner_votes` for the old `votes`, `empty_samples` for the old `n_valid`, `cost`
for the old `cost_usd`, and `f1` is new; `n_samples` is a module constant, not a field). The bullets below still use the
older spellings in places — the mapping is the "Step 06 as built" table at the end of "Our inventions".

* **`trajectory` — defined per condition (D25).** Rule 6 requires a full trajectory on **every** line, and five of the seven
  conditions have no loop, so each needs its own definition. In every case it is the **live** part only; the instruction +
  exemplar prefix is constant across a run and would add ~6 KB to every line (the reference stores the whole prompt,
  `hotpotqa.ipynb:114` `'traj': prompt`, which we deliberately do not copy).
  * `react` / `act`: the **scratchpad** — the live question line plus every `Thought i / Action i / Observation i` step, i.e.
    the exact string the graph accumulated. A normally finishing episode therefore ends with
    `Episode finished, reward = 0\n`; a step-limit episode ends at `Observation 7: <obs>\n` (D24).
  * `standard` / `cot`: the live-question prompt **tail** plus the completion — `f"Question: {q}\nAnswer:{completion}"` for
    `standard`, `f"Question: {q}\nThought:{completion}"` for `cot` (`Claim:` in place of `Question:` on FEVER).
  * `cotsc`: an object `{"winner": <raw completion of the winning sample>, "samples": [all 21 raw completions],
    "votes": {<normalized answer>: count}}`. Rule 6 says *full*, and for CoT-SC the full thing is all 21 samples; at ≈100
    tokens each the cost is acceptable and it makes the vote auditable after the fact.
  * `react_to_cotsc` / `cotsc_to_react`: the trajectory of whichever source run supplied the chosen prediction, copied
    verbatim, alongside the `source` field naming that run. These conditions write no new LLM calls, so they invent no
    trajectory of their own.
* `votes` = the winning answer's vote count, `null` outside CoT-SC. `cotsc_to_react` (D6) consumes it and nothing else
  carries it; without this field the combination rule cannot be computed from `runs/` at all.
* `n_valid` = the number of **non-empty** CoT-SC samples (D13), `null` outside CoT-SC. `votes <= n_valid <= n_samples`;
  recorded because a low `n_valid` is the diagnostic that distinguishes "the model disagreed with itself" from "the model
  failed to emit `Answer:`", and both back off to ReAct under D6.
* `n_samples` = 21 for CoT-SC, 1 otherwise. **The D6 threshold is always `n_samples / 2 = 10.5`, never `n_valid / 2`** — the
  paper's `n` is the sample count (§3.2).
* `n_steps` / `hit_step_limit` follow the D15 table in §1: `0` / `false` for `standard`/`cot`/`cotsc`, the loop index for
  `act`/`react`, inherited for the two combination conditions.
* `source` ∈ {`"react"`, `"cotsc"`} on `react_to_cotsc` / `cotsc_to_react` lines only — which run supplied this line's
  prediction (D15). Absent elsewhere.
* `n_calls` / `n_badcalls` mirror `info` in `hotpotqa.ipynb:114`; C4's tests assert on them.
* `cost_usd` lets step 07's cost gate (CLAUDE.md rule 8) be checked per condition without re-reading `calls.csv`.

---

## Our inventions (not in paper, not in reference code)

The reference implements **ReAct only**: `hotpotqa.ipynb:76` loads `webthink_simple6` and `FEVER.ipynb:76` loads
`webthink_simple3`; `grep -n "webact\|cotqa\|webqa"` over both notebooks returns **no hits**, and
`grep -c "majority\|vote\|temperature=0.7"` returns 0 for `hotpotqa.ipynb` and 5 for `FEVER.ipynb` — all five inside stored
Wikipedia observations (`:2829, :5162, :5712, :8116, :8292`), never code. So Act, Standard, CoT, CoT-SC, both combination
rules and the parsers are ours. These specs are **decided**, not open questions.

### Index of decisions D1–D28 and D37–D42

Every decision is recorded **where the implementing step looks for it**; this table is the index, not a second home for the
content. A row with a location outside this section has a stub heading below pointing at it, so `grep "^### D"` finds all
of them. (D29–D36 were locked in earlier steps' decision files and are not re-hosted here; only D36 is cited, in §5.)

| D | Decision | Where it actually lives |
|---|---|---|
| D1 | instruction-header scope (HotpotQA `react`/`act` only) | **§ "Our inventions" → D1** below; cross-referenced from the §5 table's "HotpotQA instruction header" row and asserted in C6 |
| D2 | Standard prompt + parse | D2 below |
| D3 | CoT prompt + parse (one call) — **its parse bullet is retracted by D37**: FIRST occurrence, not last | D3 below, and D37 |
| D4 | CoT-SC (21 requests, vote, tie-break, empty-sample rule) | D4 below (**D13 is folded into it**) |
| D5 | Act loop (**including D17's first-line rule**) | D5 below |
| D6 | the two combination rules | D6 below |
| D7 | `step()` returns a 3-tuple | **§1, "Return arity — decided (D7)"**, and the `step() arity (D7)` row of "Taken from code, not paper"; stub below |
| D8 | disambiguation recursion capped at depth 1 | **§2, "Disambiguation retry (D8)"**, effect estimate in the "Deliberate deviations we add" table; stub below |
| D9 | empty / malformed action | D9 below |
| D10 | zero-result search pages | D10 below **and** a row of the "Deliberate deviations we add" table |
| D11 | `tests/EXPECTED.md:45/46` fine-tuning claims — escalate, do not amend | **§ "Disagreements with tests/EXPECTED.md" → L45, L46**; stub below |
| D12 | observation drift (`Pages for logged out editors learn more.`) | **§2, "Observation drift (D12)"**; stub below |
| D13 | CoT-SC: empty samples do not vote | **folded into D4** below (and the C8 JSONL schema's `n_valid` bullet); stub below |
| D14 | LLM cache stores `raw` **and** `text` | **C3's row in §6** (`08-verify.md:10` evidence); stub below |
| D15 | step accounting for the no-tool and combination conditions | **§1, "Step accounting for every condition"**, and the C8 JSONL schema bullets; stub below |
| D16 | price table and our model name | **§5, "Model and price table (D16)"**; stub below |
| D17 | Act: the action is the first line of the completion | **folded into D5** below, asserted in C5; stub below |
| D18 | `WikiEnv._fetch` — disk cache, 10 attempts on timeout (1 try + 9 retries), User-Agent | **§2, "The fetch layer — `WikiEnv._fetch(url)` (D18)"**, tests in C1(i)(j)(k); stub below |
| D19 | record the fixture before pinning its literal | **§2, the "Failure (results page)" bullet**, asserted in C1(c); stub below |
| D20 | `EXPECTED.md:44` is partially untestable here | **§ "Disagreements with tests/EXPECTED.md" → L44**, plus a fine-tuning row in the deviations table; stub below |
| D21 | env ownership and lifecycle — `build_graph(condition, env)`, `run_one` resets once per question, env is not a state field | **§1, "The env is not in the state"**, tested in C8; stub below |
| D22 | the step limit is a `force_finish` **node**; the router is pure (prompt-spec correction to `05-react-graph.md:9`) | **§1, "The step limit needs a NODE"**, a deviations-table row, and C4's entry points and item (d); stub below |
| D23 | the one literal a zero-result search page emits | **folded into D10** below and into §2's "Zero-result pages" bullet; asserted in C1(d) |
| D24 | a step-limit episode appends nothing after `Observation 7` | **§1, "What a step-limit episode's scratchpad ends with"**, narrowing the §2 terminal-literal claim; asserted in C4(d); stub below |
| D25 | what the JSONL `trajectory` field holds, per condition | **the C8 JSONL schema's `trajectory` bullet in §6**; asserted in C8; stub below |
| D26 | CLAUDE.md rule 8 has **two** triggers, and the `$15` vs `$5` conflict | **the C8 / C11 / C12 rows of §6** and **"Disagreements with tests/EXPECTED.md" → Run gates**; stub below |
| D27 | rule 4's "one model" is scoped to phases 1–2; phase 3's student runs under the same cache, cost log and decoding | **§5, "Scope of CLAUDE.md rule 4's one model"** + decoding-table row 7 + the naming table's phase-3 rows + C12; stub below |
| D28 | the LLM cache key includes the system message | **the first row of the "Deliberate deviations we add" table**; asserted in C3; stub below |
| D37 | the answer parse rule — first occurrence of the literal `Answer:` (**retracts D3's "last"**) | **D37 below**, with the measured exemplar table and one verbatim exemplar per key; asserted in C6 |
| D38 | baseline prompt construction per task and condition | **D38 below** (D1–D3 restated with the measured whitespace); asserted in C6 |
| D39 | CoT-SC: 21 requests at 0.7 under distinct `sample_index`, `winner_votes`, `empty_samples` | **D39 below**; asserted in C6 |
| D40 | the two combination rules + `source` logging + the threshold note | **D40 below**; asserted in C7 |
| D41 | log EM **and** F1 on every condition (its example is corrected in place) | **D41 below**; asserted in C8 |
| D42 | the budget: `MAX_SPEND_USD` 5.00 → 20.00, measured from the step-05 live run | **D42 below**; the gate itself is `src/llm.py`, called from `src/run.py` |

### D1 — instruction-header scope

Prepend the `hotpotqa.ipynb:77-82` header **only** for HotpotQA `react` and `act`. Never for HotpotQA
`standard`/`cot`/`cotsc`: it describes `Search`/`Lookup`/`Finish` actions that do not exist in those conditions and would be
misleading context, and `webqa_simple6`/`cotqa_simple6` carry no header of their own. Never for **any** FEVER condition —
all four `fever.json` values already open with
`Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION. ` inline.

### D2 — Standard

* HotpotQA: `webqa_simple6 + f"Question: {q}\nAnswer:"`
* FEVER: `webqa_simple3 + f"\nClaim: {c}\nAnswer:"` — the extra leading `\n` restores the blank-line separation, because
  `webqa_simple3` ends with a single `\n` but its exemplars are `\n\n`-separated.
* **One** call, `temperature=0`, `max_tokens=100`, `stop=["\n"]`. Prediction = the completion, `.strip()`.

### D3 — CoT (greedy)

* HotpotQA: `cotqa_simple6 + f"Question: {q}\nThought:"`, `stop=["\nQuestion:"]`
* FEVER: `cotqa_simple3 + f"\nClaim: {c}\nThought:"`, `stop=["\nClaim:"]`
* **ONE** call, not two. The exemplars put `Thought:` and `Answer:` on consecutive lines, so a single continuation emits
  both. `temperature=0`, `max_tokens=100`. (Two calls would double CoT-SC cost for nothing.)
* **Parse:** take the text after the **LAST** occurrence of the literal `Answer:` — **not** `"Answer: "` — then `.strip()`.
  `cotqa_simple3` and `webqa_simple3` both contain `Answer:REFUTES` with no space; a `"Answer: "` split drops it silently.
  > **RETRACTED in part by D37** (2026-09-15): the literal `Answer:` is right, the **LAST** is not.
  > D37 takes the **FIRST** occurrence. Kept here rather than overwritten so the change is visible:
  > the two agree on every stop-cut completion and differ only when the stop list fails, where LAST
  > returns the *next* question's answer. Everything else in this bullet stands.
* Do **not** prepend `Let's think step by step. ` to the FEVER continuation: 0 occurrences in `cotqa_simple3` (§4).
* If no `Answer:` appears in the completion, the prediction is `""` (scores 0) and the call counts as a bad call.

### D4 — CoT-SC

* Identical prompt to D3. **n = 21 independent requests**, not one request with `n=21`. `temperature=0.7`.
* The cache key includes `sample_index`, so each of the 21 caches separately and a partially completed run resumes free.
* Exactly **one** `results/calls.csv` row per *uncached* sample. Cost per question ≈ 21 × (prompt_tokens × in-rate +
  completion_tokens × out-rate); with the CoT prompt ≈ 500–700 prompt tokens and ≤ 100 completion tokens, budget
  ≈ 21 × that per question and check it against CLAUDE.md rule 8's $5 gate **before** launching step 07's `cotsc` run.
* Vote over `normalize_answer(pred)`; record the winner's **raw** prediction and its vote count (`votes` in the C8 schema).
* **Which raw string:** when several distinct raw predictions share the winning normalized key
  (`"Richard Nixon"`, `"Richard Nixon."`, `"richard nixon"`), record the raw string of the **lowest sample index** — the same
  rule as the tie-break below, so one rule covers both and the JSONL `prediction` field is pinned.
* **Tie-break, deterministic:** among tied answers pick the one whose *first* occurrence has the lowest sample index. Never
  `random`, never `Counter.most_common()` ordering alone.
* **Empty samples do not vote (D13).** A sample whose completion contains no `Answer:` yields the empty prediction (D3).
  Empty predictions are **excluded from the tally** — `""` can never be the winner.
  * The denominator stays the paper's **`n = 21`**, not the count of valid samples: §3.2 says "when the majority answer among
    *n* CoT-SC samples occurs less than *n*/2 times", and *n* is the sample count. The D6 threshold is therefore **10.5
    regardless of how many samples parsed**.
  * That consequence is the desired one: a question where many samples fail to parse produces a low winner count and so backs
    off to ReAct under `cotsc_to_react` — exactly the paper's "internal knowledge might not support the task confidently".
  * Record `votes` (winner count), `n_valid` (non-empty samples) and `n_samples` (21) in the JSONL.
  * If **all 21** samples are empty: prediction `""`, `votes = 0`, `n_valid = 0`, EM 0 — and `0 <= 10`, so `cotsc_to_react`
    takes the ReAct prediction. Asserted in C6.

### D5 — Act

* prompt = `[instruction if HotpotQA]` + `webact_simple6` (HotpotQA) / `webact_simple3` (FEVER, header inline) +
  `f"{Question|Claim}: {x}" + "\n"`.
* Per step **one** call on `prompt + f"Action {i}:"` with `stop=[f"\nObservation {i}:"]`, then `.strip()`.
* **The action is the FIRST LINE of the stripped completion (D17):** `action = completion.strip().split("\n")[0]`.
  A chat model can emit `Search[A]\nAction 2: Search[B]` past the stop string. Without this rule the whole blob goes to the
  env, whose prefix test (`wikienv.py:132`) still matches `search[` and whose suffix test still sees a trailing `]`, so the
  entity silently becomes `A]\nAction 2: Search[B` — a garbage search with no counter and no retry. ReAct is protected from
  the identical case by its strict two-way unpack (C4(c)); Act has no such guard, so it needs this one.
* **No** thought/action split, therefore **no bad-call retry path** — there is no Thought to split off. But Act **does count
  bad calls (D17):** a first line that is empty, or that contains no `[`, increments `n_badcalls`. So `n_badcalls == 0` must
  **not** be asserted unconditionally for Act (C5).
* Scratchpad step = `f"Action {i}: {action}\nObservation {i}: {obs}\n"` (no Thought line).
* Same step limit 7, same `action[0].lower() + action[1:]`, same `obs.replace('\\n','')`, same forced `finish[]` as ReAct.
* **Whitespace hazard:** `webact_simple6` is the one key with **no** leading `\n` and a **`\n\n`** tail (see §5), the mirror
  image of `webthink_simple6`. So HotpotQA Act has no blank line after `Here are some examples.` and one blank line before
  the live `Question:`. Do not "normalize" either.

### D6 — combination rules (paper §3.2, verbatim)

* `react_to_cotsc`: the ReAct prediction, unless `hit_step_limit` is true → the CoT-SC prediction.
* `cotsc_to_react`: the CoT-SC prediction, unless the winner's votes < n/2 (21/2 = 10.5, i.e. `votes <= 10`) → the ReAct
  prediction. **`n` is `n_samples` (21), never `n_valid`** — empty samples are excluded from the tally but not from the
  denominator (D13), so a question whose samples mostly failed to parse backs off to ReAct by construction.
* Both are pure post-processing over two `runs/` JSONL files. They launch **no** LLM calls, so their `results.csv`
  `total_cost` is `0.0`, and each output line inherits `n_steps` / `hit_step_limit` from the source it picked and records
  `source` (D15).

### D7 — `step()` returns a 3-tuple

Stub. Defined in **§1, "Return arity — decided (D7)"** and logged in the `step() arity (D7)` row of "Taken from code, not
paper". Summary: `(observation, done, info)` per `02-environment.md:1`; safe because `wikienv.py:125` fixes `reward = 0`.

### D8 — disambiguation recursion capped at depth 1

Stub. Defined in **§2, "Disambiguation retry (D8) — load-bearing"**; the effect estimate is the `Disambiguation recursion capped at depth 1 (D8)` row of the "Deliberate
deviations we add" table under "Taken from code, not paper". Summary: reproduce `wikienv.py:112-113` (re-search
`"[" + entity + "]"` when a block contains `"may refer to:"`), but cap the recursion at one retry.

### D9 — empty / malformed action

`action[0].lower() + action[1:]` raises `IndexError` on `""` in the reference (`hotpotqa.ipynb:102`), which with a chat model
and client-side stop cutting is a realistic outcome and would kill a 500-question run mid-flight. We guard: an empty action is
passed through as `""`, the env answers `Invalid action: ` (matching `wikienv.py:155-156` semantics — `"".strip()` is `""`,
no prefix matches) and the episode continues. Asserted in C4(f).

### D10 — zero-result search pages (2026 drift)

A query with no near-matches renders a page with **no** `mw-search-result-heading` divs, so the reference logic falls through
to the article branch and returns `There were no results matching the query..` plus sidebar nav chrome. Verified against
`tests/fixtures/nonexistent_miss.html` (`Qwertzuiop Zzyzxian Orogeny`). We treat a page whose extracted text begins with
`There were no results matching the query` **and that has zero `mw-search-result-heading` divs** as a miss with an **empty**
Similar list. It cannot affect parity with the reference on real questions, only on pathological searches.

**The literal, pinned (D23).** The observation is the ordinary miss string of `wikienv.py:108-109` with an empty
`result_titles` — **not** the raw "There were no results…" text:

```text
Could not find Qwertzuiop Zzyzxian Orogeny. Similar: [].
```

C1(d) asserts exactly this, byte for byte, against the fixture. **Miss semantics:** the path leaves `page`,
`lookup_keyword`, `lookup_list` and `lookup_cnt` untouched, exactly like the reference's miss branch — a failed search does
not clear the current page.

### D11 — `tests/EXPECTED.md:45/46` fine-tuning claims: escalate, do not amend

Stub. Defined in **"Disagreements with tests/EXPECTED.md" → L45 and L46**. Summary: `EXPECTED.md:45` (Claim A) misquotes
paper §3.3 — the "beats all 540B prompting methods" result is **PaLM-62B** finetuned ReAct, not 8B. `EXPECTED.md:46`
(Claim B) agrees in substance but paraphrases "significantly" as "much" and drops the "for both PaLM-8/62B" scope.
`tests/EXPECTED.md` is authoritative (`EXPECTED.md:3-4`), is **not** edited by this document, and the conflict is escalated
to the repo owner. Both claims are untestable here (no PaLM-8B/62B/540B, and no `standard_train.jsonl` SFT run); C12 prints
the surrogate claims instead.

### D12 — observation drift

Stub. Defined in **§2, "Observation drift (D12)"** and in the "Page scrape" row of "Taken from code, not paper". Summary:
`Pages for logged out editors learn more.` is a 2022 artefact of the authors' stored outputs (479× in `reference/FEVER.ipynb`,
0× in a 2026-09-15 live fetch). Treat exact observation strings from those stored outputs as historical, not as fixtures.

### D13 — CoT-SC: empty samples do not vote

Stub. **Folded into D4** above (the "Empty samples do not vote" bullets) and into the C8 JSONL schema's `n_valid` bullet,
because that is where step 06 looks. Summary: `""` is excluded from the tally and can never win; the threshold denominator
stays `n = 21`.

### D14 — the LLM cache stores `raw` and `text`

Stub. Defined in **C3's row of §6**. Summary: each cache entry stores `raw` (the provider's completion text exactly as
returned, before any client-side cutting) **and** `text` (what `complete()` returns, after cutting at the first stop string).
Without `raw`, `08-verify.md:10` ("show a cached call where the raw output was cut") cannot be satisfied from the cache and
step 08 would have to re-run a populated 500-question cache to recover the evidence.

### D15 — step accounting for the no-tool and combination conditions

Stub. Defined in **§1, "Step accounting for every condition"** (a table) and repeated as bullets in the C8 JSONL schema.
Summary: `standard`/`cot`/`cotsc` log `n_steps = 0`, `hit_step_limit = false`, and write **empty** `mean_steps` /
`pct_hit_step_limit` in `results.csv`; `act`/`react` log the loop index, never `env.steps`; combination rows inherit both
from the chosen source line, record `source`, and have `total_cost = 0.0`.

### D16 — price table and our model name

Stub. Defined in **§5, "Model and price table (D16)"**. Summary: our model is `gpt-4o-mini` (`.env`); rates live in
`src/llm.py` as a module-level dict, 0.15 / 0.60 USD per 1M input / output tokens as of 2026-09-15; token counts come from
the response `usage`; an unknown model costs 0.0 with a warning. This is what makes CLAUDE.md rule 8's $5 gate computable —
and the rates are a local constant that must be re-checked before any cost total is quoted in `README.md`.

### D17 — Act: the action is the first line

Stub. **Folded into D5** above and asserted in C5, because that is where step 05/06 looks. Summary:
`action = completion.strip().split("\n")[0]`; an empty first line, or one with no `[`, counts as a bad call.

### D18 — `WikiEnv._fetch`: disk cache, 10 attempts on timeout, User-Agent

Stub. Defined in **§2, "The fetch layer — `WikiEnv._fetch(url)` (D18)"**, tested by C1(i)(j)(k). Summary: JSON-directory
cache at `data/cache/wiki/<sha256(url)>.json` keyed by the exact URL, up to 10 attempts on timeout (1 initial try + 9
retries; 10 consecutive timeouts raise), and the User-Agent
`react-langgraph-repro/0.1 (research reproduction; contact via repo)` — recorded once so fixtures are reproducible.

### D19 — record the fixture before pinning its literal

Stub. Defined in **§2, the "Failure (results page)" bullet**, asserted by C1(c). Summary: live Wikipedia search rankings are
not stable (four fetches of `Colorado+orogenyy` on 2026-09-15 gave 20 result divs every time but two different top-5
orderings), so no five-title literal may be pinned from a live fetch. Record the fixture first, paste the literal *that file*
produces, and let the live smoke test assert only the stable parts.

### D20 — `tests/EXPECTED.md:44` is partially untestable here

Stub. Defined in **"Disagreements with tests/EXPECTED.md" → L44**, with a matching row in the "Deliberate deviations we add"
table. Summary: L44 agrees with paper App. B.1, but of its three values only **batch size 64** is reproduced here;
PaLM-8B/62B and the 4,000-step schedule are not (Qwen2.5-3B-Instruct, LoRA, 3 epochs ≈ 141 optimizer steps).

### D21 — env ownership and lifecycle

Stub. Defined in **§1, "The env is not in the state — ownership and lifecycle (D21)"**, tested in C8. Summary: the `WikiEnv`
is **not** a state field; `build_graph(condition, env)` closes over it; `src/run.py:run_one(idx, question, task, condition,
env)` owns it and calls `env.reset()` once per question before invoking the graph; one env instance serves a whole run, as in
`hotpotqa.ipynb:42-43` + `:86`.

### D22 — the step limit is a node, not a conditional edge

Stub. Defined in **§1, "The step limit needs a NODE, not a conditional edge (D22)"**, with a row in the "Deliberate
deviations we add" table and entry points plus item (d) in C4. Summary: `05-react-graph.md:9` instructs the conditional edge
to set `hit_step_limit` and call `finish[]`; verified against langgraph 1.2.11, a path function's state writes are discarded,
so the router is made pure (`"end"` / `"force_finish"` / `"think_act"`) and `force_finish` becomes a real node. Behaviourally
identical to `hotpotqa.ipynb:110-111`.

### D23 — the zero-result observation literal

Stub. **Folded into D10** above and into §2's "Zero-result pages" bullet; asserted in C1(d). Summary: the emitted string is
`Could not find {entity}. Similar: [].`, and the path leaves `page` / `lookup_*` untouched.

### D24 — a step-limit episode appends nothing after `Observation 7`

Stub. Defined in **§1, "What a step-limit episode's scratchpad ends with (D24)"**; it narrows the §2 claim about the
`Episode finished, reward = 0\n` literal to normally finishing episodes, and is asserted in C4(d). Summary: the forced
`finish[]` runs after the last scratchpad append (`hotpotqa.ipynb:110-111` vs `:105`), so its observation never enters the
trajectory; the line carries `hit_step_limit = true`, `answer = ""`, `em = 0`.

### D25 — what the JSONL `trajectory` field holds

Stub. Defined in **the C8 JSONL schema's `trajectory` bullet (§6)**, because that is where step 06/07 looks, and asserted in
C8. Summary: `react`/`act` store the scratchpad; `standard`/`cot` the live-question prompt tail plus the completion; `cotsc`
an object with the winning completion, all 21 raw samples and the vote map; the two combination conditions copy the
trajectory of the source line named by `source`. Never the constant instruction + exemplar prefix.

### D26 — CLAUDE.md rule 8 has two triggers

Stub. Defined in **the C8 / C11 / C12 rows of §6** (where the runs are planned) and in **"Disagreements with
tests/EXPECTED.md" → Run gates**. Summary: rule 8 (`CLAUDE.md:32`) gates on question count **and** on cost, independently;
every step-07 run, the step-10 collection sweep and the step-12 500-question eval are all over 100 questions and need an
explicit ask before launch whatever the estimate says. `10-collect-trajectories.md:5`'s `$15` conflicts with rule 8's `$5`;
CLAUDE.md is the rules file and wins.

### D27 — rule 4's "one model" is scoped to phases 1–2

Stub. Defined in **§5, "Scope of CLAUDE.md rule 4's one model for all conditions (D27)"**, with the decoding table's seventh
row, the naming table's three phase-3 rows and C12's assertions. Summary: the seven Table-1 conditions all run on
`gpt-4o-mini`; phase 3's `Qwen2.5-3B-Instruct` student (prompted and fine-tuned) is the §3.3 experiment, not a violation, and
goes through the same `complete()`, the same disk cache, the same `results/calls.csv` and the same temperature 0 / 100-token
decoding, with a `0.0` price entry and its own `runs/` JSONL and `results.csv` row.

### D28 — the LLM cache key includes the system message

Stub. Defined in **the first row of the "Deliberate deviations we add" table**, asserted in C3. Summary: the key is
`(model, system_message, prompt, stop, temperature, max_tokens, sample_index)` — one field more than
`04-llm-client.md:4` lists, because every request carries the system message too and rule 5 says "keyed by the exact input".
Without it, editing the system message silently serves the whole run from stale completions. C3 asserts that changing the
system message **misses** the cache.

**Numeric fields are normalised before hashing**, added with corrections-log entry 9: `_key` stores
`float(temperature)` and `int(max_tokens)`, because `json.dumps` spells `0` as `"0"` and `0.0` as `"0.0"` and the two
therefore hashed to different sha256s. `src/graph_react.py` passes `temperature=0` while `complete`'s own default (and the
entry already sitting in `data/cache/llm/`) is `0.0`, so every ReAct and Act call of a rerun would have missed the cache,
re-issued and re-billed while returning identical text. C3 asserts that the two spellings share one cache path and cost one
request.

### Decisions D37–D42 (step 06 — baselines, combination rules, runner)

Locked before step 06, and reproduced here **with the measurements behind them** so that
`grep "^### D"` finds them and steps 07/08 never have to reopen the decision file they were locked
in. All six are implemented as written; the single disagreement is an evidence error inside D41's
example (recorded under D41 and in CLAUDE.md's corrections log), not a change to any rule.

#### The measured exemplar table — the basis for D37 and D38

Counts are `str.count` over the four baseline keys, not recollection:

| key | file | exemplars | terminator | `Answer: ` **with** a space | answers ending `.` | own header | blank-line separated | chars |
|---|---|---|---|---|---|---|---|---|
| `webqa_simple6` | `prompts_naive.json` | 6 | `Answer: X` | 6/6 | 0/6 | no | no | 738 |
| `cotqa_simple6` | `prompts_naive.json` | 6 | `Answer: X` | 6/6 | 0/6 | no | no | 1,945 |
| `webqa_simple3` | `fever.json` | 3 | `Answer: X` | **2/3** | 0/3 | yes | yes | 363 |
| `cotqa_simple3` | `fever.json` | 3 | `Answer: X` | **2/3** | 0/3 | yes | yes | 725 |

The 2/3 rows are the whole reason D37 exists: **the third FEVER exemplar in both baseline keys
reads `Answer:REFUTES`, with no space.** `tests/test_baselines.py::test_both_spellings_are_really_in_the_exemplars`
re-measures both cells (`"Answer:REFUTES" in value` and `value.count("Answer: ") == 2`) so the
table cannot drift away from the files.

One exemplar quoted **verbatim** per key — the second item of each file, which is the same
question/claim in all four, and on FEVER is the `REFUTES` one, i.e. D37's evidence:

`prompts_naive.json["webqa_simple6"]`, exemplar 2 of 6 (ends with one `\n`, no blank line):

```text
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Answer: Richard Nixon
```

`prompts_naive.json["cotqa_simple6"]`, exemplar 2 of 6 — note `Thought: Let's think step by step. `
(HotpotQA only, 6/6) and the closing `so the answer is X.`:

```text
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought: Let's think step by step. Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Answer: Richard Nixon
```

`fever.json["webqa_simple3"]`, exemplar 2 of 3 — **`Answer:REFUTES`, no space**, and the block ends
with `\n\n` (the blank-line separation D38 restores for the live claim):

```text
Claim: Stranger Things is set in Bloomington, Indiana.
Answer:REFUTES

```

`fever.json["cotqa_simple3"]`, exemplar 2 of 3 — same no-space `Answer:`, and the thought carries
**no** `Let's think step by step. ` prefix (0/3 on FEVER; do not add it):

```text
Claim: Stranger Things is set in Bloomington, Indiana.
Thought: Stranger Things is in the fictional town of Hawkins, Indiana, not in Bloomington, Indiana.
Answer:REFUTES

```

### D37 — the answer parse rule (**retracts the parse bullet of D3**)

`src/baselines.py::parse_answer`, one line:

```python
_, sep, tail = text.partition("Answer:")   # partition == FIRST occurrence
return tail.strip() if sep else ""
```

Split the (already stop-cut) completion on the literal `"Answer:"` — **not** `"Answer: "` — take
what follows the **FIRST** occurrence, and `.strip()` it.

* **`"Answer:"`, not `"Answer: "`.** `Answer:REFUTES` appears in *both* FEVER baseline keys (table
  above), so a parser splitting on the spaced form drops that answer shape silently — it returns
  `""`, which scores 0 and is indistinguishable in `runs/` from a model that never answered.
* **FIRST, not LAST. This CORRECTS D3**, whose parse bullet says "the text after the **LAST**
  occurrence". D3 is retracted **by name** on that point only; the rest of D3 (prompt, one call,
  greedy, no `Let's think step by step. ` on FEVER, empty prediction = bad call) stands. After the
  stop cut the two rules agree, because a well-formed completion contains exactly one `Answer:`;
  they differ exactly when the stop list fails and the completion runs on into
  `Question: <next>\nThought: ...\nAnswer: <other>`, where LAST returns **a different question's
  answer** and FIRST returns this one's. FIRST is therefore identical when cutting worked and
  strictly safer when it did not.
* **No trailing-period strip.** No exemplar answer ends in `.` (0/6, 0/3 above), `normalize_answer`
  removes punctuation for scoring anyway, and CLAUDE.md rule 6's `prediction` must stay raw.
* A completion with no `Answer:` at all yields `""`: EM 0, and one bad call (D3).
* Not part of the rule: any cut at the first newline. `parse_answer` returns everything after the
  first `Answer:`, trailing junk included. The §6 C6 row used to give
  `parse_answer("Thought: ...\nAnswer: yes\nQuestion: next") == "yes"`, which no split-and-strip
  rule produces — corrected there, and noted here because it is the one place the two spellings of
  the rule looked different.
* **Standard does not use `parse_answer` at all** (D2): its prompt already ends in `Answer:`, so
  the completion *is* the answer and is only stripped. Feeding it to `parse_answer` would return
  `""` for every well-formed Standard completion.

### D38 — baseline prompt construction (D1–D3 restated with the measured whitespace)

| condition | prompt | stop |
|---|---|---|
| HotpotQA `standard` | `webqa_simple6 + f"Question: {q}\nAnswer:"` | `["\n"]` |
| HotpotQA `cot` / `cotsc` | `cotqa_simple6 + f"Question: {q}\nThought:"` | `["\nQuestion:"]` |
| FEVER `standard` | `webqa_simple3 + f"\nClaim: {c}\nAnswer:"` | `["\n"]` |
| FEVER `cot` / `cotsc` | `cotqa_simple3 + f"\nClaim: {c}\nThought:"` | `["\nClaim:"]` |

The extra leading `\n` on FEVER reproduces its blank-line exemplar separation (each `*_simple3`
value ends with a single `\n` while its exemplars are `\n\n`-separated); HotpotQA must **not** get
one. **No instruction header is prepended for any baseline on either task (D1)** — the FEVER keys
carry theirs inline as their own first line. `temperature=0.0`, `max_tokens=100`, one call.

### D39 — CoT-SC (D4/D13 restated, plus the diagnostic this step adds)

Same prompt as D38's `cot` row. **n = 21 independent requests at temperature 0.7**, one
`llm.complete(..., n=21)` call, which is 21 requests under `sample_index` 0–20 (§5's `n` column).
Twenty-one separate `n=1` calls would be the same code path with every sample keyed on
`sample_index=0`: one request, one cache entry, and one answer voting twenty-one times —
`tests/test_baselines.py::test_cot_sc_issues_21_requests_under_21_distinct_sample_indices` asserts
the distinctness against the cache keys, not against the return value.

* Vote over `normalize_answer`; the recorded prediction is the winner's **raw** string.
* **Empty (parse-failed) samples do not vote, and the denominator stays 21** (D13), so the D40
  threshold is 10.5 whatever the parse rate. `empty_samples` is recorded per question precisely so
  that a low winner count can be read correctly: many empty samples means the *parser* (or the
  model's format compliance) failed, not that the model was internally uncertain, and the README
  must be able to tell those two apart.
* `winner_votes` is the **winner's** count, never the sample total.
* Deterministic tie-break: among tied normalized answers, the one whose **first** occurrence has
  the lowest sample index; its raw string is the prediction. One scan of the samples settles this
  and the several-raw-spellings case together.

### D40 — the combination rules, plus `source` logging

* `react_to_cotsc`: the ReAct prediction **unless `hit_step_limit`**, then CoT-SC's. It keys off
  the step limit, never off `em` — the paper backs off when ReAct "fails to return an answer within
  given steps", not when it answers wrongly, and an `em`-keyed rule would also be an oracle.
* `cotsc_to_react`: the CoT-SC prediction **unless `winner_votes < 21/2 = 10.5`** (i.e. `<= 10`),
  then ReAct's.
* Both are pure post-processing over two `runs/` JSONL files joined on `idx`; `src/combine.py`
  issues **no** LLM call and imports nothing that can.
* Every output line records `source ∈ {"react", "cotsc"}`, so the fallback fraction is readable
  from `runs/` alone. This is not bookkeeping: if ReAct's step-limit rate is high,
  `react_to_cotsc` collapses onto CoT-SC and stops being a combination at all, and that has to be
  visible in the README rather than inferred.
* Note on the threshold, for whoever reviews the mutation table: `votes < 10.5`, `votes <= 10.5`
  and `votes <= 21 // 2` are the **same predicate** for integer vote counts at odd n, so no test
  can separate them; `votes < 21 // 2` is a different one and is killed by the 10-vote fixture
  row. The boundary is therefore pinned exhaustively (fallback set == {0..10}) and from both sides
  (10 → ReAct, 11 → CoT-SC).

### D41 — log EM **and** F1 on every condition

EM is the Table 1 headline; F1 is the diagnostic that separates a wrong answer from a right answer
EM rejected. Every condition logs both, including the two combination conditions (which inherit
them from the line they picked).

> **Evidence correction.** D41 cites live question 5388 — prediction `torpedo boats and submarines`
> against gold `torpedoes` — as "EM 0, F1 > 0". Measured, that pair scores **EM 0 and F1 0.0**:
> SQuAD normalization lowercases and strips punctuation but does **not** stem, so `torpedo` and
> `torpedoes` are different tokens and the overlap is empty. The decision stands; the example does
> not. A pair on the same question that does show the gap is `torpedoes and submarines` → **EM 0,
> F1 0.5**, and that is what `tests/test_run.py` pins (both pairs, so the corrected claim cannot
> quietly revert). Logged as corrections-log entry 11.

### D42 — the budget, and what it is measured from

`MAX_SPEND_USD` was raised **5.00 → 20.00** in `.env`. **The original gate was $5.00** (CLAUDE.md
rule 8), and rule 8's ask-before-spending requirement is **unchanged** — only the ceiling moved.
Measured basis, from the 16 real calls of the step-05 live run:

| quantity | measured |
|---|---|
| calls per question (ReAct, HotpotQA) | 5.33 |
| mean prompt tokens | 1,980 (min 1,611, max 2,374) |
| cost per call | $0.000333 |
| ReAct on 500 HotpotQA questions | ≈ $0.89 |
| all five conditions on both tasks | ≈ $6.40 |

So **the $5.00 ceiling would have been hit partway through phase 1** — with the cheap conditions
already paid for and the expensive one half-finished, which is the worst place to stop. That, not
a change of policy, is why the number moved.

Two consequences that are code, not prose:

* `src/run.py` calls `llm.check_budget_for_batch` **before the first call of every run** and exits
  on a refusal, and it refuses to start a run of more than 100 questions without `--yes-over-100`
  — rule 8's *other* trigger (D26), which no cost estimate can satisfy.
* The runner's pre-flight deliberately sizes ReAct/Act **above** the measured 5.33 calls (it
  assumes 8: seven steps plus a parse-failure retry) and at the step-05 *maximum* prompt length
  (~9,500 chars ≈ 2,374 tokens), because `llm._estimate` wants the last step's prompt, not the
  mean. A gate that rounds down is not a gate.
* Raising the ceiling in `.env` broke `tests/test_llm.py`, which asserted the reloaded module's
  `MAX_SPEND_USD == 5.00`; the assertion now pins the *invariant* (finite and non-negative) rather
  than the operator's number. Corrections-log entry 12.

### Step 06 as built — signatures and JSONL field names

The names the code uses, because C6/C7/C8 above were written before D39/D41 fixed them:

| where | as built | earlier name in §6 |
|---|---|---|
| `baselines.majority_vote(preds)` | `-> (winner_raw, votes, empty_samples)` | `(winner, votes, n_valid)` |
| JSONL, CoT-SC winner count | `winner_votes` | `votes` |
| JSONL, unparsed sample count | `empty_samples` (`n_valid == 21 - empty_samples`) | `n_valid` |
| JSONL, per-question spend | `cost` | `cost_usd` |
| JSONL, new in D41 | `f1` on every line | — |
| `baselines`/`run` result rows | `winner_votes` / `empty_samples` are `null` outside `cotsc` | unchanged |

`n_samples` is not a JSONL field: it is the constant 21 in both `src/baselines.py` and
`src/combine.py` (pinned equal by a test), because the D40 threshold is the paper's n and must not
become a per-line value that a future writer can get wrong. `results/results.csv` keeps its
verbatim eight-column header — F1 lives in `runs/`, per question, where the diagnostic is useful.

### Disagreements with the locked decisions

One, and it is an **evidence** error rather than a rule change: D41's cited EM/F1 example does not
score what D41 says it scores (see the correction box under D41, and corrections-log entry 11). The
decision itself — log EM and F1 on every condition — is implemented as written.

Otherwise none. D1–D28 and D37–D42 are implemented as written, and every one of them is findable
from the index at the head of this section. D29–D36 were settled in earlier steps' locked-decision
files; the only one this document cites is D36 (§5, the `n` column).

---

## Taken from code, not paper

Standing log. Every detail we take from `reference/` lands here.

| Detail | Value | reference file:line | Why the paper is insufficient |
|---|---|---|---|
| HotpotQA instruction header | the full 5-line block, quoted character-for-character in §5 (521 chars; trailing space after `three types:`; exactly one trailing `\n`) | `hotpotqa.ipynb:77-82` | App. C.1 shows all four HotpotQA prompts starting at `Question`; the string appears nowhere in the paper (0 grep hits). FEVER's header, by contrast, *is* in App. C.2. The header's own text is wrong about its environment ("first paragraph" vs `wikienv.py:87`'s first 5 sentences) and capitalises the action names the paper writes lowercase — reproduce verbatim anyway |
| Instruction/exemplar join | `webthink_prompt = instruction + webthink_examples` | `hotpotqa.ipynb:83`; `prompts/prompts_naive.json` | App. C is a LaTeX two-column table; no whitespace is specified |
| Step limit 7 for **both** tasks | `for i in range(1, 8)` | `hotpotqa.ipynb:91`, `FEVER.ipynb:84` | Paper §3.2 says 5 steps for FEVER; `tests/EXPECTED.md:24` sides with the code |
| Stop-string construction | `stop=[f"\nObservation {i}:"]`, rebuilt with the current step index | `hotpotqa.ipynb:93`, `FEVER.ipynb:86` | The paper never describes how a step's generation is terminated |
| Thought/Action split | `thought_action.strip().split(f"\nAction {i}: ")` — leading newline, one trailing space | `hotpotqa.ipynb:95`, `FEVER.ipynb:88` | Paper shows the format only as a typeset table |
| Parse-failure retry | bare `except:` → thought = `thought_action.strip().split('\n')[0]`, second call `llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=["\n"]).strip()`, `n_badcalls += 1` (`:98`) **and** `n_calls += 1` (`:99`) — a bad step costs 2 calls. **The trigger is any unpack failure, not just a missing separator:** the reference unpacks into exactly two names, so `ValueError` fires on **0** occurrences (`not enough values to unpack`) **and on ≥2** (`too many values to unpack (expected 2, got 3)`). Reproduce with a strict two-way unpack `thought, action = s.split(sep)`; never `split(sep, 1)` or `str.partition`, which silently return `['I need X.', 'Search[A]\nAction 1: Search[B]']` and diverge in `n_badcalls`, `n_calls` and in the action executed | `hotpotqa.ipynb:96-101`, `FEVER.ipynb:89-94` | Paper is silent on malformed output; the branch fired 3× in 500 questions in the authors' stored FEVER run |
| Action first-char lowercasing | `action[0].lower() + action[1:]` — first character only, entity casing preserved; `IndexError` on `""` → see D9 | `hotpotqa.ipynb:102` | Paper §3.1 writes `search[...]` lowercase, App. C writes `Search[...]`; the env only matches lowercase (`wikienv.py:132,137,148`) |
| Observation scrub | `obs.replace('\\n', '')` — removes the 2-character sequence backslash-`n`, not real newlines | `hotpotqa.ipynb:103` | Not mentioned; it is a patch over `clean_str`'s unicode-escape round trip (`wikienv.py:10-11`) |
| Scratchpad step format | `f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"` — labels re-emitted by the driver, no blank line between steps | `hotpotqa.ipynb:104` | App. C specifies no separators |
| Retry helper `step(env, action)` | `while attempts < 10: try: return env.step(action) except requests.exceptions.Timeout: attempts += 1` — called at `:102` and `:111`, **not** `env.step` directly. Dead code: `requests` is never imported in either notebook, so the `except` raises `NameError`; it also returns `None` after 10 failures | `hotpotqa.ipynb:46-52`, `FEVER.ipynb:46-52` | Paper says nothing about transport failures |
| Timeout / step-limit behaviour | forced `step(env, "finish[]")` → `answer = ""` → EM 0, indistinguishable from a wrong answer | `hotpotqa.ipynb:111`; `wikienv.py:149-151` | Paper never says what a step-limit hit scores |
| `clean_str` | `p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")` | `wikienv.py:10-11` | Not in the paper; determines every observation byte |
| Search URL | `entity.replace(" ", "+")` into `https://en.wikipedia.org/w/index.php?search={entity_}`; nothing else escaped; no `timeout=`, no `User-Agent`, no retry inside the env | `wikienv.py:99-102` | Paper says only "a simple Wikipedia web API". Do **not** substitute `urllib.parse.quote` — it changes which page Wikipedia returns |
| Article-vs-results detection | presence of any `div.mw-search-result-heading` | `wikienv.py:105-107` | Not in the paper |
| Page scrape | `page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]` (all `<p>` then all `<ul>`, **out of document order** — `<ul>` nav chrome lands at the end of `self.page`, where it can pollute `lookup` but never the first-5-sentence observation); then per block, kept iff `len(p.split(" ")) > 2`, appended as `clean_str(p)` with `"\n"` appended only `if not p.endswith("\n")` (tested on the **pre**-`clean_str` block). **Not** a `"\n".join` | `wikienv.py:111`, `:115-120` | Paper gives no HTML→text pipeline. Verified 2026-09-15 against live fetches of `Colorado orogeny`, `Milhouse` and `High Plains (United States)`: the first surviving block is the real lead paragraph. (`Pages for logged out editors learn more.` is **not** the first sentence today — it occurs 479× in `reference/FEVER.ipynb`'s stored 2022 outputs — `grep -c` says 480, its `.` being a wildcard — and 0× in a live fetch: dataset drift, so exact observation strings from those outputs are not fixtures) |
| "First 5 sentences" rule | split page on `"\n"`, then each paragraph on the literal `'. '`, strip, re-append `'.'`, `' '.join(sentences[:5])` — doubles the period at paragraph ends, merges sentences before citation markers | `wikienv.py:76-87` | Paper says "first 5 sentences" with no splitter |
| Disambiguation retry | any block containing `"may refer to:"` → `self.search_step("[" + entity + "]")`, unbounded in the reference (**we cap at depth 1**); the bracketed name leaks into `Could not find [X].` | `wikienv.py:112-113`, `:109` | Explains the paper's own inconsistency between App. C.1 (bracketed) and App. E.1 (unbracketed). Verified live 2026-09-15 |
| Search-failure literal | `f"Could not find {entity}. Similar: {self.result_titles[:5]}."` — Python list `repr`, single quotes, period after `]`, no trailing newline | `wikienv.py:108-109` | Paper §3.1 says "top-5 similar entities" but its own App. C.1 exemplar lists 7 |
| Lookup literals | hit `f"(Result {cnt+1} / {len(list)}) " + sentence` (no trailing newline); exhausted `"No more results.\n"` (trailing newline) | `wikienv.py:144-146` | Neither string appears in the paper's prose |
| Lookup semantics | case-insensitive substring over whole sentences, list rebuilt only on a case-sensitive keyword change, cleared only on a **successful** search or `reset()`; never rewound; `[]` when `page is None` | `wikienv.py:59-74`, `:122`, `:139`, `:61-62` | Paper says only "next sentence in the page containing string" |
| `reset()` state | six fields cleared: `page`, `lookup_keyword`, `lookup_list`, `lookup_cnt`, `steps`, `answer` (plus `obs`) | `wikienv.py:44-57` | Not in the paper; a partial reset leaks question *k*'s page into question *k+1* |
| `think[...]` action | returns `Nice thought.` | `wikienv.py:153-154` | A 4th action outside the paper's 3-action space; appears in 0 of the 12 prompt keys; we omit it |
| Invalid action | `"Invalid action: {}".format(action)` — the bare `action`, already stripped at `:127`; no penalty, `steps` still increments, episode continues | `wikienv.py:127`, `:156` | Paper is silent; 12 occurrences in the authors' 500-question FEVER run — the model self-corrects, there is no retry |
| Terminal observation | wrapper overwrites the env's `reward = 0` string with `f"Episode finished, reward = {reward}\n"` carrying the real EM | `wikienv.py:152`; `wrappers.py:130-131`, `:190-191` | Paper shows `Episode finished` with no reward |
| Reward | `reward = 0` for every action inside `WikiEnv`, never reassigned | `wikienv.py:125` | Paper mentions no reward signal for these tasks |
| **`step()` arity (D7)** | reference returns `(obs, reward, done, info)`; **we return `(observation, done, info)`** per `prompts/claude-code/02-environment.md:1`. Safe because the dropped element is always `0`. `info = {"steps": int, "answer": str\|None}` | `wikienv.py:160`, `:41-42`, `:125` | Deliberate simplification, not a silent divergence |
| Eval index construction | `idxs = list(range(7405)); random.Random(233).shuffle(idxs); idxs[:500]` — **including for FEVER**, whose dev file has 9,999 lines; `max(idxs[:500]) == 7390` | `hotpotqa.ipynb:126-132`, `FEVER.ipynb:9362-9368` | Paper gives only "a subset of 500 validation questions" (App. A.1) and no seed, no bound, no split file |
| Data files | `HOTPOTQA_SPLIT_FILE` maps all three splits — train `hotpot_train_v1.1_simplified.json` (90,447 entries, keys `question/answer/type`, the SFT bootstrap source), dev `hotpot_dev_v1_simplified.json` (7,405), test `hotpot_test_v1_simplified.json` (7,405, **key `question` only — no answer, unusable for EM**); FEVER dev `paper_dev.jsonl` (9,999) | `wrappers.py:11-20` | Paper names neither the split files nor the HotpotQA variant |
| HotpotQA `type` field | `bridge` 5,918 / `comparison` 1,487 in dev. The reference loader **drops** it (`wrappers.py:85`); ours returns a 3-tuple, an EM-neutral deviation required by `03-data-metrics.md:3` and `07-full-runs.md:6` | `wrappers.py:84-85` | Paper never mentions the field; step 07's breakdown groups on it |
| Task-prompt seeds | `f"Question: {q}"` / `f"Claim: {c}"`, replacing WikiEnv's instruction observation entirely | `wrappers.py:97`, `:166` | Not specified |
| Prompt keys per condition | ReAct `webthink_simple6`/`webthink_simple3`; Act `webact_*`; CoT `cotqa_*`; Standard `webqa_*` (paper calls it "Original"). **`prompts_naive.json` also contains four unused legacy keys — `webthink_simple`, `webthink_simple_3`, `cotqa_simple`, `webqa_simple` — none of which may be loaded.** Beware `webthink_simple_3` (HotpotQA, underscore) vs `webthink_simple3` (FEVER): a one-character slip loads a different prompt with no error. Assert *membership* of the four required keys, never key-set equality | `hotpotqa.ipynb:76`, `FEVER.ipynb:76` (ReAct only); key dumps of both JSONs | Paper prints the prompts but names no keys |
| EM normalization | SQuAD recipe, order lower → punctuation → articles → whitespace, ASCII punctuation only | `wrappers.py:42-56` | Paper says "EM" and never defines normalization |
| F1 | `f1_score` returns the 3-tuple `(f1, precision, recall)`; the caller takes `[0]`. Full body quoted in §3 | `wrappers.py:58-78`, `:122` | Paper reports EM only |
| FEVER scoring | normalized prediction must equal the normalized gold label (`supports`/`refutes`/`not enough info`); `f1 = em = reward` | `wrappers.py:178-184`, `:193` | Paper never says how a free-form `Finish[...]` maps to a label; note the prompt header says `NOT ENOUGH INFORMATION` while every answer is `NOT ENOUGH INFO` |
| Decoding parameters | `model="text-davinci-002"`, `temperature=0`, `max_tokens=100`, `top_p=1`, `frequency_penalty=0.0`, `presence_penalty=0.0` | `hotpotqa.ipynb:23-29` | **"Greedy decoding" itself IS in the paper** for these runs (§3.3 fn 4, p.6, plus the Table 3 and Table 5 captions). Only the concrete API settings and the token budget are code-only |
| Exemplar whitespace | full measured table in §5 | `prompts/prompts_naive.json`, `prompts/fever.json` (byte-identical to `reference/prompts/`) | App. C cannot encode whitespace |
| Exemplar typos to preserve | `Answer:REFUTES` (no space) in `cotqa_simple3`/`webqa_simple3`; FEVER exemplar 3 claim ends `...in 2003.?`; HotpotQA exemplar 6 answers `Yes` in Standard/CoT but `Finish[yes]` in Act/ReAct | `prompts/fever.json`, `prompts/prompts_naive.json` | CLAUDE.md rule 2: use exemplars verbatim, never rewrite |

### Deliberate deviations we add (required by CLAUDE.md or by the prompt files)

Each row states the change and a **one-line estimate of its likely effect on results** (`08-verify.md:19`).

| Deviation | What we do | Effect estimate |
|---|---|---|
| Disk cache on every LLM call and every Wikipedia fetch | keyed by `(model, system_message, prompt, stop, temperature, max_tokens, sample_index)` for LLM, by exact URL for Wikipedia (CLAUDE.md rule 5). **`system_message` is one field more than `04-llm-client.md:4` lists (D28)** — without it rule 5's "keyed by the exact input" is false, because every request carries both the prompt and the system message | None on EM at temperature 0; at 0.7 it *freezes* sampling noise, making CoT-SC reruns reproducible rather than re-sampled. The extra key field is what stops an edit to the system message — the one knob flagged below as likely to be tuned — from silently serving the whole 500-question run from stale completions |
| Real 10-attempt fetch retry with a `User-Agent` header (D18) | the reference retry is dead code (`requests` never imported) and bare-UA requests to `en.wikipedia.org` are 403'd today. UA string, fixed: `react-langgraph-repro/0.1 (research reproduction; contact via repo)` — the same string records every fixture | None on EM; without it the run cannot complete at all |
| **`response.raise_for_status()` in `_fetch` (D18)** | the reference does none of it (`requests.get(search_url).text`, `wikienv.py:99-102`). Only `Timeout` is retried, so a non-2xx is **not** retried: it propagates out of `_fetch` and stops the run | A transient 403/429/503 aborts the run loudly instead of being parsed as an article and written into the **permanent** disk cache, where it would silently poison every rerun of a 500-question run. Cost: a rare transient error needs a restart (the cache makes the restart cheap). Asserted in C1: a 500 raises `HTTPError` and writes no cache file |
| **`timeout=30` on the `_fetch` GET (D18)** | the reference passes no `timeout=` (`wikienv.py:99-102`), so its `except requests.exceptions.Timeout` in the driver (`hotpotqa.ipynb:46-52`) can never fire — a hung socket blocks forever. A retry policy needs a timeout to retry *from*, so we set one | None on EM. Bounds a hung request at 30s instead of indefinitely; a slow-but-alive Wikipedia response under 30s is unaffected. Too small a value would convert healthy responses into 10 wasted retries, so the magnitude is asserted in C1, not just its presence |
| `hit_step_limit` flag, per-question JSONL, `results/calls.csv` | CLAUDE.md rules 5 and 6 | None on EM; `hit_step_limit` is the input to D6's `react_to_cotsc` |
| **Chat-vs-completion adaptation** | the reference calls a raw completion endpoint with a single prompt string (`hotpotqa.ipynb:22-32`); we send the identical completion-style prompt as **one user message** to a chat model (`04-llm-client.md:1`) | Chat models are RLHF-tuned to answer rather than continue text; expect more preamble and more parse failures than davinci-002, i.e. higher `n_badcalls` and a small EM drag unless the system message below compensates |
| **System message** | short, fixed, identical across all conditions and both tasks — **our wording** for the requirement at `04-llm-client.md:1` (which states it in prose and contains no such literal; `grep -rn "Continue the text exactly in the format" prompts/claude-code/` → 0 hits): *"Continue the text exactly in the format of the examples. Do not write an Observation line; stop before it."* It is part of the cache key (row 1), lives as `SYSTEM_MESSAGE` in `src/llm.py`, and C3 asserts both that the literal in this row is the literal in the module and that changing it misses the cache | Suppresses conversational answers and raises parse success; may **inflate** EM relative to a raw completion model, since it supplies format guidance the reference's prompt did not. Because it is constant across conditions it should not reorder them |
| **Client-side stop cutting** | after the provider's own stop handling, cut the returned text at the **first** occurrence of any stop string and **exclude** the stop string itself (`04-llm-client.md:3`) | Matches OpenAI completion semantics exactly when the provider honours stop; when it does not, it prevents the model writing its own `Observation`, which would otherwise poison the scratchpad and inflate EM by letting the model hallucinate evidence |
| **Temperature 0 is not determinism; the cache is** | `temperature=0` is greedy, not a provider guarantee: serving-stack routing, GPU batching and a floating model alias can all change the bytes. Reruns are reproducible because the disk cache replays them, not because the provider repeats itself. Stated in full in §5, "Temperature 0 is not determinism", and **marked for the README's Limitations section** (`12-evaluate-serve.md:14`) | None on a cached rerun, by construction: EM is bit-identical. On a **cold** cache — a clean clone, a new model string, an edited system message — every condition may move, so a README number is reproducible only together with `data/cache/llm/`, which rule 9 keeps out of git. The honest claim is "reproducible from this cache", not "deterministic" |
| `think[...]` omitted | our env implements only `search`/`lookup`/`finish`/invalid | None: `think[` appears in 0 of the 12 prompt keys, so the model is never primed to emit it; if it ever does, it lands on the invalid-action branch and the episode continues |
| Disambiguation recursion capped at depth 1 (D8) | `wikienv.py:112-113` is unbounded | None observed: every verified disambiguation resolves in one retry. Prevents a pathological page from hanging a 500-question run |
| Zero-result search pages treated as a miss with an **empty** Similar list (D10) | the reference falls through to the article branch and returns `There were no results matching the query..` plus sidebar chrome | None on real questions; affects only pathological searches, where it replaces nav chrome with a clean miss the model can act on |
| **D1 — instruction header on HotpotQA `react`/`act` only** | the header is code-only and scoped by us; `standard`/`cot`/`cotsc` get none | Should raise Standard/CoT EM slightly vs. prepending it, by not describing three actions those conditions cannot take. If it moves ReAct/Act at all it is via the header's own wrong claim ("first paragraph"), which we reproduce verbatim either way |
| **D2 — Standard: one call, `stop=["\n"]`, `.strip()`** | no reference implementation exists | Cheapest condition; the `["\n"]` stop means a chat model's conversational preamble is truncated to its first line, which scores 0 rather than a wrong answer. Expect Standard to sit at or slightly below its paper value |
| **D3 — CoT: ONE call emitting Thought and Answer together, split on the LAST literal `Answer:`** | the paper describes CoT but the notebooks implement none | Halves CoT and CoT-SC cost versus a two-call design. Risk: if Thought + Answer exceeds `max_tokens = 100` the completion truncates before `Answer:` and scores **0 rather than a wrong answer**, depressing CoT/CoT-SC EM slightly relative to the paper. Splitting on `"Answer: "` instead would silently drop `Answer:REFUTES` and cost real FEVER points |
| **D4 — CoT-SC: 21 independent requests at 0.7, vote on `normalize_answer`, lowest-sample-index tie-break** | §3.2 gives n and the temperature but no sampler, no tie-break, no raw-string rule | The tie-break is EM-neutral in expectation but makes reruns byte-identical; the per-sample cache freezes the sampling noise, so CoT-SC here has **less run-to-run variance** than the paper's. Dominates cost: ≈21× CoT |
| **D13 — empty samples excluded from the vote, denominator still 21** | §3.2 is silent on parse failures | Pushes questions with many unparsed samples into the ReAct back-off instead of letting `""` win. Raises `cotsc_to_react` EM relative to a naive tally; slightly lowers standalone CoT-SC EM by refusing a `""` plurality that would have scored 0 anyway |
| **D5 + D17 — Act loop: one call per step on `f"Action {i}:"`, action = first line, bad calls counted** | neither notebook implements Act | Without D17's first-line rule a multi-action completion becomes a garbage search with no counter — a silent EM loss concentrated in chat-model runs. With it, Act's failures show up as `n_badcalls`, comparable to ReAct's |
| **D6 — combination thresholds `hit_step_limit` and `votes <= 10`** | §3.2 gives both heuristics but no operational definition of "fails to return an answer" | `hit_step_limit` is the strictest reading (a wrong-but-returned answer is not a failure), so `react_to_cotsc` switches on fewer questions than a looser rule would — expect it closer to plain ReAct than the paper's +7.7 EM gap |
| **D22 — prompt-spec correction: the step limit is a `force_finish` NODE, not a conditional edge** | `05-react-graph.md:9` says the conditional edge should "set hit_step_limit=True, call finish[] and END". Verified against the installed langgraph 1.2.11, a path function's state writes are discarded (final state `{'step': 8, 'hit_step_limit': False}`), so the instruction is un-implementable as written. We make the router pure and add a `force_finish` node (§1) | None on EM — the decided shape is behaviourally identical to `hotpotqa.ipynb:110-111`. Implementing the prompt literally would instead lose `hit_step_limit` on **every** step-limit episode, silently zeroing `pct_hit_step_limit` and disabling D6's `react_to_cotsc` back-off entirely |
| **`clean_str` guarded (MIN1)** | the reference line raises `UnicodeDecodeError`/`UnicodeEncodeError` on text containing literal `\x`/`\u` escape sequences; we wrap it and return the block unchanged on either exception | None on any page that round-trips (all three fixtures do), so observation bytes are unaffected; it converts a run-killing crash on one pathological page into one unconverted block |
| **Fine-tuning: Qwen2.5-3B-Instruct + LoRA for 3 epochs, not PaLM-8B/62B for 4,000 steps (D20)** | `11-train-lora.md:5`; only the paper's batch size 64 is reproduced. 3 epochs over ≤3,000 examples at batch 64 is ≈141 optimizer steps | Large and unquantifiable: a 3B student distilled from a small API teacher cannot be compared to the paper's absolute numbers. Only the **relative** step-12 claims (fine-tuned > prompted, ReAct > CoT/Act, fewer prompt tokens) are meaningful here, which is why `12-evaluate-serve.md:14` requires a Limitations section |

---

## Disagreements with tests/EXPECTED.md

Cross-check of sections 2, 3 and 5 against `tests/EXPECTED.md`, line by line. **`tests/EXPECTED.md` is authoritative
(`EXPECTED.md:3-4`) and is not edited by this document.** Where it conflicts with the paper, the conflict is recorded and
escalated, not resolved here.

**Evaluation sample**
- L7-8 `idxs = list(range(7405))`, `random.Random(233)`, `idxs[:500]` — **AGREES**. `hotpotqa.ipynb:126-132`,
  `FEVER.ipynb:9362-9368`, identical text.
- L9 first five `3687, 6238, 5388, 3522, 3824` — **AGREES**. Replayed locally.
- L10-11 holds for FEVER despite 9,999 lines; "do not fix" — **AGREES**. `FEVER.ipynb:9362` really says `range(7405)`; the
  notebook's stored 500-question run has max index 7390. Pinned by C2/C8/C10.

**Data files**
- L14 `hotpot_dev_v1_simplified.json`, 7,405 entries, keys `question, answer, type` — **AGREES** (loaded and counted; types
  5918 bridge / 1487 comparison).
- L15 `hotpot_train_v1.1_simplified.json` as the fine-tuning bootstrap source — **AGREES**; present in `data/` (90,447
  entries), named at `wrappers.py:12`. Paper **§3.2** ("Finetuning") says the 3,000 trajectories come from a bootstrap but
  never names the split.
- L16 `paper_dev.jsonl`, 9,999 lines, keys `id, verifiable, label, claim, evidence` — **AGREES** (labels 3333/3333/3333).

**Prompt keys**
- L19-20 HotpotQA keys — **AGREES**. All four present in `prompts/prompts_naive.json`; `webthink_simple6` is the one the
  notebook loads (`hotpotqa.ipynb:76`); the other three are matched to App. C.1's Act / CoT / Original blocks. Caveats, not
  disagreements: (a) only the ReAct key is proven by code; (b) the file also holds four **unused legacy keys**, so any test
  must assert membership, not key-set equality.
- L21 FEVER keys — **AGREES**. All four present, and `fever.json` holds exactly those four; `webthink_simple3` loaded at
  `FEVER.ipynb:76`. `prompts/*.json` are byte-identical to `reference/prompts/*.json` (`cmp` clean).

**Loop**
- L24 step limit 7, `for i in range(1, 8)` — **AGREES with the code**, `hotpotqa.ipynb:91`, `FEVER.ipynb:84`. **Flag:** the
  *paper* (§3.2) says 5 steps for FEVER. EXPECTED.md and the code win per CLAUDE.md.
- L24-25 forced `finish[]` scoring 0 — **AGREES**. `hotpotqa.ipynb:111` → `wikienv.py:149-151` sets `answer = ""` →
  `wrappers.py:109-115` compares `""` to the gold.
- L26 stop string `f"\nObservation {i}:"` — **AGREES**, `hotpotqa.ipynb:93`.
- L26-27 parse failure → second call with `stop=["\n"]`, counted as a bad call — **AGREES**, `hotpotqa.ipynb:96-101`. Two
  additions, not conflicts: the same branch also does `n_calls += 1` (`:99`), so a bad step costs two API calls; and the
  trigger is **any** unpack failure, so ≥2 separators retries just as 0 separators does.
- L28 first character of the action lowercased — **AGREES**, `hotpotqa.ipynb:102`.

**Environment**
- L31-32 `https://en.wikipedia.org/w/index.php?search=<entity>` with spaces as `+` — **AGREES**, `wikienv.py:99-100`.
- L32 "returns the first 5 sentences" — **AGREES literally** (`wikienv.py:87`), with the precision that a "sentence" is a
  `'. '`-delimited fragment with `'.'` re-appended, over a `<p>`+`<ul>` scrape.
- L33 `Could not find <entity>. Similar: [<up to 5 titles>].` — **AGREES**, `wikienv.py:108-109`; Python `repr`, and `entity`
  is bracketed when the disambiguation retry fired. Caveat: a search with **no** near-matches produces no results page at all
  today and falls into the article branch (D10) — handled explicitly, not a disagreement.
- L34 `lookup[keyword]` → `(Result i / n) <sentence>`, then `No more results.` — **AGREES**, `wikienv.py:144-146`; the
  exhausted string carries a trailing `\n`, the hit string does not.
- L35 "Reward is 0 for every action; only the wrapper computes EM at the end" — **AGREES**, `wikienv.py:125` and
  `wrappers.py:129-133` / `:189-193`. This is what makes the D7 3-tuple safe.

**Metrics**
- L38-39 SQuAD normalization then exact string equality — **AGREES**, `wrappers.py:42-56` and `:113` / `:182`.
- L40 FEVER prediction must normalize to `supports` / `refutes` / `not enough info` — **AGREES**, `wrappers.py:178-184`.

**Fine-tuning**
- L43 3,000 bootstrapped trajectories, kept only where the final answer was correct — **AGREES**, paper **§3.2**, the
  "Finetuning" paragraph of METHODS: "we consider a bootstraping approach similar to Zelikman et al. (2022), using 3,000
  trajectories with correct answers generated by ReAct (also for other baselines) to finetune smaller language models
  (PaLM-8/62B)". **Section note:** `EXPECTED.md:42` heads this block "paper Section 3.3", but the *method* is §3.2 and only
  the *results* are §3.3 (§3.3 RESULTS AND OBSERVATIONS starts after that paragraph). Not a disagreement about substance —
  recorded so the §3.2/§3.3 citations below are not read as inconsistent.
- L44 PaLM-8B and 62B, batch size 64, 4,000 steps for ReAct and Act — **AGREES with the paper, but PARTIALLY UNTESTABLE
  here (D20).** App. B.1 verbatim: "For all finetuning we use a batch size of 64. On PaLM-8B, we finetune ReAct and Act
  methods for 4,000 steps and Standard and CoT methods for 2,000 steps. On PaLM-62B, we finetune ReAct and Act methods for
  4,000 steps and Standard and CoT methods for 1,000 steps."
  Of L44's three values, **only batch size 64 is reproduced**: `11-train-lora.md:5` specifies
  "per_device_batch 2 with grad accumulation to an effective batch of 64 (the paper's batch size), 3 epochs".
  * **PaLM-8B / 62B — not reproduced.** Our student is `Qwen2.5-3B-Instruct` in 4-bit with LoRA (`11-train-lora.md:5`); no
    PaLM model exists anywhere in this repo.
  * **4,000 steps — not reproduced.** 3 epochs over ≤3,000 examples at an effective batch of 64 is ≈**141** optimizer steps,
    not 4,000.
  * Disposition, the same shape as L45/L46: `tests/EXPECTED.md` is authoritative and is **not** amended; the substitution is
    logged as a row of the "Deliberate deviations we add" table, C12's paper-section cell marks App. B.1 as context rather
    than the setting in force, and `12-evaluate-serve.md:14` requires it in the README's Limitations section.
- L45 "Claim A: fine-tuned ReAct-8B beats every PROMPTING method on the 540B model" — **DISAGREES with the paper (D11).**
  Paper §3.3, verbatim from the extracted PDF text: "with PaLM-8B finetuned ReAct outperforming all PaLM-62B prompting
  methods, and PaLM-62B finetuned ReAct outperforming all 540B prompting methods." The 8B claim is against **62B** prompting,
  not 540B prompting; the 540B comparison is the **62B** result. Figure 3 prints no numbers, so the stronger form cannot be
  sourced to the paper.
  **Disposition: `tests/EXPECTED.md:45` misquotes paper §3.3. EXPECTED.md is authoritative and is NOT amended here;
  escalated to the repo owner.** What C12 actually prints, verbatim strings:
  - `Claim A (as written in EXPECTED.md:45): UNTESTABLE — no PaLM-8B/62B/540B in this repo.`
  - Surrogate printed in its place, `12-evaluate-serve.md:8`: `Claim 1: fine-tuned ReAct-3B > prompted ReAct-3B (6 exemplars) — PASS/FAIL`.
  - The other two step-12 claims, printed alongside: `Claim 2: fine-tuned ReAct > fine-tuned CoT and > fine-tuned Act — PASS/FAIL`
    (`12-evaluate-serve.md:9`); `Claim 3: fine-tuned ReAct-3B uses fewer prompt tokens per question than the prompted version,
    reported as a % reduction — PASS/FAIL` (`12-evaluate-serve.md:10`).
- L46 "Claim B: fine-tuning Standard or CoT is much worse than fine-tuning ReAct or Act" — **AGREES in substance, but not
  verbatim (D11).** The paper's wording (§3.3) is "In contrast, finetuning Standard or CoT is **significantly** worse than
  finetuning ReAct or Act **for both PaLM-8/62B**" — "significantly", not "much", and scoped to both model sizes;
  EXPECTED.md paraphrases and drops the scope.
  **Testability: only the CoT half is covered.** `10-collect-trajectories.md:5,7` writes `data/sft/{react,act,cot}_train.jsonl`
  and no Standard set, so no fine-tuned Standard model will ever exist in this repo. C12 therefore prints:
  `Claim B (Standard half): UNTESTABLE — no standard_train.jsonl SFT run is planned; add one to close it.`
  and `Claim B (CoT half): fine-tuned ReAct EM > fine-tuned CoT EM — PASS/FAIL` (= step-12 claim 2, minus the Act comparison).

**Run gates** — not an `EXPECTED.md` line, but a conflict of the same kind, recorded here because this is where conflicts live.
- `CLAUDE.md:32-33` (rule 8) has **two independent triggers**: "Ask before starting any run over 100 questions, and before any
  run whose estimated cost passes $5." The plan previously recorded only the cost half. Both halves are now stated where the
  runs are planned — the **C8**, **C11** and **C12** rows of §6 — because every step-07 run (500 questions × 2 tasks ×
  5 conditions, `07-full-runs.md:1`), the step-10 collection sweep (3,000 kept trajectories over the 90,447-entry train
  split, `10-collect-trajectories.md:5`) and the step-12 evaluation (the same seed-233 500 questions,
  `12-evaluate-serve.md:1`) trip the question-count trigger **regardless of cost**.
- `prompts/claude-code/10-collect-trajectories.md:5` says "stop early and ask me if the estimated cost passes **$15**",
  while rule 8 says ask above **$5** — **CONFLICT (D26)**. **Disposition: `CLAUDE.md` is the rules file and wins — ask at
  $5.** Prompt 10's $15 is a second, later checkpoint on the same run, not a replacement for the first ask. Neither file is
  edited here; the conflict is recorded and escalated, the same way D11/D20 handle the `EXPECTED.md` conflicts.

No other disagreements. Sections 2, 3 and 5 are otherwise consistent with `tests/EXPECTED.md` item by item, as listed above.

## For the README and for step 08 section 4

### What F1 actually catches (and what neither metric catches)

Claim this precisely in the README. F1 is token-overlap after SQuAD normalization, and SQuAD
normalization does **not** stem. So F1 only rewards an answer that shares EXACT tokens with the
gold; a morphological variant scores 0 on both metrics.

| prediction (question 5388, gold `torpedoes`) | EM | F1 | what it shows |
|---|---|---|---|
| `torpedoes and submarines` | 0 | **0.5** | over-complete answer — F1 catches it, EM does not |
| `torpedo boats and submarines` | 0 | **0.0** | **neither metric catches it**: `torpedo` != `torpedoes` |

The second row is the more interesting limitation and belongs in the write-up. Our live step-05
run produced exactly it: the model answered `torpedo boats and submarines`, which a human would
call substantively close, and both metrics scored it zero. So our EM figures are a LOWER bound on
"answers a human would accept", and the gap is not measurable with either number we report. Do NOT
write "F1 catches substantively correct answers" — it catches over-complete and under-complete
answers built from the gold's own tokens, and nothing else.

### Step 08 section 4 — the CoT-SC prompt-assertion gap (caught before any spend)

An entry for "anything in our pipeline the paper does not describe that could move results".

**The gap.** `cot_sc` was tested for temperature (0.7), sample count (21) and distinct
`sample_index` values, but NOTHING asserted the prompt text it actually sent. Two one-token
mutations survived the full 154-test suite: the lead token becoming `Answer:` instead of
`Thought:`, and the exemplar key falling back to `webqa_simple6`/`webqa_simple3`.

**The corruption path.** Either mutation turns CoT-SC into *Standard sampled at temperature 0.7* —
21 samples drawn with no chain of thought. It still returns 21 parseable answers, still produces
plausible `winner_votes`, and `runs/hotpotqa_cotsc_*.jsonl` plus the `cotsc` row of results.csv
keep their labels. Nothing errors and nothing looks wrong.

**The blast radius is three of the seven Table 1 rows, not one.** CoT-SC directly; and because
`cotsc_to_react` branches on `winner_votes`, which those corrupted samples produce, BOTH
combination rows inherit it. The paper's central CoT-SC > CoT > Standard ordering would become
unfalsifiable — we would be comparing Standard against Standard-at-0.7 and reporting it as CoT-SC.

**Status.** Found by mutation before any CoT-SC call was ever paid for. Now pinned by
`test_cot_sc_is_built_on_the_cot_prompt_not_standard`, which asserts the sent prompt is
byte-equal to `build_prompt(q, task, "cot")`, ends `\nThought:` and differs from the Standard
prompt, on both tasks.

## Power, and the pre-registered decision rule

Written after `standard` and `cot` completed at n=500 and BEFORE `act`, `react` and `cotsc` landed.
Every number here was independently re-derived three ways (exact enumeration, 200k-replicate Monte
Carlo, and the normal closed form) before being written down.

### What the completed runs show about resolution

| quantity | value |
|---|---|
| paired questions | 500 |
| standard EM / cot EM | 29.2 / 35.2 |
| discordant b (cot right, std wrong) / c (reverse) | 53 / 23 |
| **b + c** | **76** (15.2%) |
| concordant, carrying NO information about the difference | 424 (84.8%) |

**Minimum detectable EM difference, exact two-sided McNemar, 80% power, alpha 0.05: 5.17 points**
(rejection region k<=28 or k>=48; p80 = 0.6702). Monte Carlo agrees to 0.001; the normal form gives
4.885, and the 5.9% gap is fully explained by discreteness.

**This figure carries +/- 0.5 points of its own sampling noise.** `n_disc = 76` is an estimate, not
a design constant (sd 8.02, 95% range [61, 92]), so the MDD's own 95% range is **[4.38, 5.37]**.
Quote it as "about 5 points", never as 5.17.

**It does not transfer between pairs.** MDD grows roughly as sqrt(n_disc):

| n_disc | rate | MDD | max possible \|diff\| | MDD as % of that ceiling |
|---|---|---|---|---|
| 40 | 0.08 | 3.55 | 8.0 | 44% |
| 76 | 0.152 | 5.17 | 15.2 | 34% |
| 150 | 0.30 | 7.00 | 30.0 | 23% |
| 300 | 0.60 | 9.87 | 60.0 | 17% |

Do NOT read this as "pairs that disagree more are harder to resolve". Discordance is not a knob, it
is a property of the two systems, and it CAPS the effect (|diff| <= n_disc/N). Relative to the
effects that can exist at that discordance, high discordance is much easier. Each pair gets its own
n_disc and its own MDD, computed when its runs land. Below n_disc ~ 6 the exact two-sided test
cannot reject at 0.05 at ANY effect size, so MDD is undefined rather than small.

### The decision rule (pre-registered)

An earlier draft of this section proposed: mark a claim INCONCLUSIVE when the PAPER's reported gap
falls below our MDD. **That rule is wrong and is not used.** It is the post-hoc power fallacy, and
our own data is the counterexample: cot - standard is +6.00 points, p = 7.6e-04, 95% CI
[+2.62, +9.38]; the paper's gap for that pair is +0.70, which is below 5.17, so the rule would stamp
INCONCLUSIVE on a result we resolved at p < 0.001 whose interval excludes both zero and the paper's
value. Once a test rejects, the power to detect some other effect size is irrelevant to it. A rule
that can discard our strongest finding is not a pre-registration.

What the MDD legitimately licenses is exactly one sentence, and only when we FAIL to reject: *this
design could not have detected an effect as small as the paper reports, so the null result is
uninformative about a paper-sized effect.* It says nothing when we do reject.

**The rule in force, decided before act/react/cotsc landed, on the bootstrap 95% CI of our own
paired difference:**

| verdict | condition | strength |
|---|---|---|
| **SUPPORTED** | CI excludes 0 and the direction matches the paper | strong |
| **CONSISTENT WITH PAPER** | CI contains the paper's gap | moderate |
| **CONTRADICTS PAPER** | CI excludes the paper's gap | strong, and worth investigating |
| **INCONCLUSIVE** | CI contains BOTH 0 and the paper's gap | the honest default |

The CI is reported on EVERY row, including inconclusive ones: it carries the effect size regardless
of what the test decides, and it is the only column that stays interpretable when the test is
underpowered.

### Asymmetry of the verdicts

Calibration (300 replicates, n=200) put the exact test's type-I rate at **0.017 against a nominal
0.05** -- correctly conservative for a discrete test, never anti-conservative. Consequences:

1. **SUPPORTED is a strong verdict.** The test rejects less often than nominal, so a rejection is
   harder-won than the p-value alone suggests.
2. **Power is correspondingly reduced**, so a failure to reject is weak evidence of absence.
3. **NOT SUPPORTED is therefore never written.** The phrase is **"no evidence at this power"**, and
   it is never a refutation of the paper. We are running a different model, four years later, at
   n=500; the paper reported PaLM-540B. A null here is a statement about our experiment.

Do NOT recompute power at alpha = 0.017. That is a category error: the conservatism already lives
inside the exact rejection region, and feeding the achieved rate back in as a nominal level
double-counts it. (An earlier draft reported 5.56 points on that basis; it is withdrawn.)

### Claim 4 is structurally not an independent test

Both combinations are deterministic functions of the react and cotsc runs. `react_to_cotsc` can
differ from `react` ONLY on questions where ReAct hit the step limit, so its discordant table is
entirely conditioned on a subset selected by ReAct's own failures, and c is bounded above by the
step-limit count. McNemar there answers "does substituting CoT-SC on ReAct's failures help?", which
is worth knowing and is NOT the independent-method comparison claims 1-3 make. Against methods that
share no machinery (`react_to_cotsc` vs `cot`, vs `standard`) the comparison is clean. The claims
table reports the split, never four uniform p-values.

## Hypotheses about WHY, and their provenance

Provenance is tracked strictly here. A prediction written after the evidence is not a prediction,
and labelling it as one would be worse than not writing it at all.

### H1 — knowledge-limited (NOT pre-registered; reconstructed 2026-09-15 AFTER the cot result)

**Status: this was never recorded in the repo before the evidence arrived.** It was raised in
conversation as something that had been pre-registered; no such entry exists in notes.md, in any
other file, or in the run logs. It is written down here as a reconstruction, dated, precisely so
that the absence of an earlier record is visible rather than papered over.

The hypothesis: the binding constraint on HotpotQA is missing knowledge, not reasoning, so giving
the model tools (ReAct/Act) should help MORE here than it did in the paper, i.e. the tool gap
should exceed the paper's.

**First evidence runs against it.** `cot - standard = +6.0` points (35.2 vs 29.2) where the paper
reports +0.7 (29.4 vs 28.7). CoT supplies no tools and no new facts — only a reasoning scaffold. A
knowledge-limited model would not gain 6 points from a scaffold alone.

### H2 — reasoning-scaffold limited (revised reading of the same evidence)

Standard scores 29.2 against the paper's 28.7, so the FACTS are largely present: a bare-answer
prompt on a 2026 small model matches a 2022 540B model. What the scaffold buys is room to use them.
So the constraint is the absence of a place to reason, not the absence of knowledge.

### P1 — GENUINELY PRE-REGISTERED, 2026-09-15, before act/react/cotsc landed

Written while `act` stood at 57/500 and `react` at 20/500 (the 20 being the earlier development
run), with neither condition's n=500 EM known to anyone.

**If H2 holds, tools should add LESS here than in the paper, not more.** A model that already has
the facts and is merely short of room to reason gains little from an external retriever.
Concretely, on HotpotQA:

| quantity | paper | P1 predicts |
|---|---|---|
| ReAct - CoT | -2.0 | **at or below -2.0** (ReAct further behind CoT than in the paper) |
| ReAct - Act | +1.7 | no directional prediction; H2 says nothing about thoughts-vs-no-thoughts |

P1 is falsifiable and will be scored in the claims table against the pre-registered CI rule above,
whichever way it falls. If ReAct - CoT comes out ABOVE -2.0, P1 is wrong and gets marked wrong here.

### Candidate headline, pending act/react

`cot - standard` is +6.0 here against +0.7 in the paper: a reasoning scaffold does roughly eight
times as much for a small 2026 model as it did for a 540B 2022 model. It is above our ~5-point
resolution (p = 7.6e-04, CI [+2.62, +9.38]), so unlike most of Table 1 it is a result this
experiment can actually resolve. If it survives act/react, the README should open with it rather
than with the Table 1 ordering -- the ordering is mostly inconclusive at this power, and this is not.

