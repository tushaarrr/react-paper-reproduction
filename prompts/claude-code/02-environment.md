Implement src/wiki_env.py from section 2 of paper/notes.md: a WikiEnv class with reset(), step(action) -> (observation, done, info), and the three actions.

- search[entity]: GET https://en.wikipedia.org/w/index.php?search=<entity with spaces as +>. If it lands on an article, return the first 5 sentences of the article's paragraphs joined by a space. If it lands on a search results page, return "Could not find <entity>. Similar: [<up to 5 titles>]." Keep the fetched page text as the current page for lookup.
- lookup[keyword]: case-insensitive; return "(Result i / n) <sentence>" for successive calls over sentences of the current page containing the keyword, then "No more results.\n".
- finish[answer]: set done and store the answer. Any other string returns "Invalid action: ..." without ending the episode.
- A disk cache (sqlite or a JSON directory) keyed by the exact URL, so reruns never hit the network. Retry timeouts up to 10 times.
- Disambiguation recursion (load-bearing, NOT an edge case). If the landed page contains a block reading "may refer to:", the reference does NOT return it. It re-searches the entity wrapped in literal square brackets: `search_step("[" + entity + "]")` (reference/wikienv.py:112-113). This is why the authors' own exemplars read `Observation 1: Could not find [Adam Clayton Powell]. Similar: [...]` and `Could not find [Beautiful]. Similar: [...]` with the brackets visible in the prompt. Omit this and our observations stop matching the format the model is few-shot primed on. Cap the recursion at one retry: if the bracketed page is itself a disambiguation page, return its own result rather than searching a third time.
- `clean_str` (reference/wikienv.py:10-11), applied to every kept paragraph AND every search result title:
  ```python
  def clean_str(p):
      return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
  ```
  It round-trips the text through unicode-escape/latin1. Two consequences: real escape sequences in the page get decoded, and literal two-character `\` + `n` sequences appear in observations. That second one is why the driver runs `obs.replace('\\n', '')` — in Python source `'\\n'` is the TWO-CHARACTER sequence backslash-n, NOT a newline. An implementation that strips real newlines there is doing something different. Note that `clean_str` raises on text containing invalid escapes, so guard it and return the block unchanged on `UnicodeDecodeError` / `UnicodeEncodeError`.

Write tests/test_wiki_env.py using two recorded pages saved in tests/fixtures/ (mock the HTTP layer): search success returns exactly 5 sentences, search failure returns the Similar list, lookup counts match, finish sets done. Also add one live smoke test marked slow for search[Colorado orogeny].

Then compare behaviour with reference/wikienv.py for those three actions and list any difference in paper/notes.md.

Green tests are not enough here — the first version of this module passed 15 tests while
several behaviours above were completely unpinned. Verify by mutation: apply each of the
following to a scratch copy and confirm at least one test goes RED. A mutation that stays
green means the suite is hollow, and you strengthen it before moving on.

1. Replace the body of `clean_str` with `return p`.
2. Drop any one of the six fields from `reset()` (`page`, `obs`, `lookup_keyword`,
   `lookup_list`, `lookup_cnt`, `answer`) — each field separately.
3. Cap `lookup`'s result list at 5 (`[:5]`). There is no cap; this silently changes the
   `(Result i / n)` denominator the model reads.
4. Remove the disambiguation recursion.
5. Return 4 or 6 sentences instead of 5 from the page observation.

Report which test caught each one.
