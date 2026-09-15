Implement src/wiki_env.py from section 2 of paper/notes.md: a WikiEnv class with reset(), step(action) -> (observation, done, info), and the three actions.

- search[entity]: GET https://en.wikipedia.org/w/index.php?search=<entity with spaces as +>. If it lands on an article, return the first 5 sentences of the article's paragraphs joined by a space. If it lands on a search results page, return "Could not find <entity>. Similar: [<up to 5 titles>]." Keep the fetched page text as the current page for lookup.
- lookup[keyword]: case-insensitive; return "(Result i / n) <sentence>" for successive calls over sentences of the current page containing the keyword, then "No more results.\n".
- finish[answer]: set done and store the answer. Any other string returns "Invalid action: ..." without ending the episode.
- A disk cache (sqlite or a JSON directory) keyed by the exact URL, so reruns never hit the network. Retry timeouts up to 10 times.

Write tests/test_wiki_env.py using two recorded pages saved in tests/fixtures/ (mock the HTTP layer): search success returns exactly 5 sentences, search failure returns the Similar list, lookup counts match, finish sets done. Also add one live smoke test marked slow for search[Colorado orogeny].

Then compare behaviour with reference/wikienv.py for those three actions and list any difference in paper/notes.md.
