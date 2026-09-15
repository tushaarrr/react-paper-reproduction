"""C1 of paper/notes.md section 6, items (a)-(n). Every HTTP call is mocked except (h)."""

import hashlib
import json
import logging
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import wiki_env
from src.wiki_env import USER_AGENT, WikiEnv

FIXTURES = Path(__file__).resolve().parent / "fixtures"
HIT_URL = "https://en.wikipedia.org/w/index.php?search=Colorado+orogeny"
SIMILAR_URL = "https://en.wikipedia.org/w/index.php?search=Colorado+orogenyy"
MISS_URL = "https://en.wikipedia.org/w/index.php?search=Qwertzuiop+Zzyzxian+Orogeny"

# (a) — the literal tests/fixtures/colorado_orogeny_hit.html produces (D19), doubled
# period at the paragraph boundary included, no trailing newline.
HIT_OBS = (
    "The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado "
    "and surrounding areas. This took place from 1780 to 1650 million years ago (Mya), "
    "during the Paleoproterozoic (Statherian Period). It is recorded in the Colorado "
    "orogen, a >500-km-wide belt of oceanic arc rock that extends southward into New "
    "Mexico. The Colorado orogeny was likely part of the larger Yavapai orogeny.. The "
    "Colorado orogen, formerly called the Colorado province, is a >500-km-wide belt of "
    "oceanic arc rock (1.78\u20131.65 Ga) that extends southward into New Mexico and "
    "composes a major part of the Proterozoic provinces of southwestern United States."
)


# (b)/(c)/(d) — the two New Mexico sentences, pinned exactly like every other literal.
RESULT1 = (
    "(Result 1 / 2) It is recorded in the Colorado orogen, a >500-km-wide belt of "
    "oceanic arc rock that extends southward into New Mexico."
)
RESULT2 = (
    "(Result 2 / 2) The Colorado orogen, formerly called the Colorado province, is a "
    ">500-km-wide belt of oceanic arc rock (1.78\u20131.65 Ga) that extends southward "
    "into New Mexico and composes a major part of the Proterozoic provinces of "
    "southwestern United States."
)


class FakeResponse:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code))


def serve(monkeypatch, pages):
    """Install a fake transport over `pages` (url -> html); returns the call log."""
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers))
        return FakeResponse(pages[url])

    monkeypatch.setattr(wiki_env.requests, "get", fake_get)
    return calls


def fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def page(*blocks):
    return "<html><body>" + "".join(f"<p>{b}</p>" for b in blocks) + "</body></html>"


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(wiki_env, "CACHE_DIR", tmp_path / "wiki")
    env = WikiEnv()
    env.reset()
    return env


@pytest.fixture
def searched(env, monkeypatch):
    """An env that has just done search[Colorado orogeny] against the fixture."""
    serve(monkeypatch, {HIT_URL: fixture("colorado_orogeny_hit.html")})
    env.step("search[Colorado orogeny]")
    return env


def test_search_url_construction(env, monkeypatch):
    """(n) spaces -> +, nothing else escaped; asserted on the URL handed to _fetch."""
    seen = []
    monkeypatch.setattr(env, "_fetch", lambda url: seen.append(url) or page("x y z w"))
    env.step("search[Colorado orogeny]")
    assert seen == [HIT_URL]


def test_search_hit_returns_five_sentences(env, monkeypatch):
    """(a)"""
    serve(monkeypatch, {HIT_URL: fixture("colorado_orogeny_hit.html")})
    obs, done, info = env.step("search[Colorado orogeny]")
    assert obs == HIT_OBS
    assert obs.startswith(
        "The Colorado orogeny was an episode of mountain building (an orogeny) in "
        "Colorado and surrounding areas."
    )
    assert "larger Yavapai orogeny.. The Colorado orogen" in obs  # doubled period
    assert obs.count(". ") == 4  # 5 '. '-fragments
    assert not obs.endswith("\n")
    assert (done, info["answer"], info["steps"]) == (False, None, 1)


def test_lookup_counts(searched):
    """(b)"""
    assert searched.step("lookup[New Mexico]")[0] == RESULT1
    # The MATCH is case-insensitive (the same two sentences) but the CURSOR identity is
    # case-SENSITIVE (wikienv.py:133): a differently-cased keyword is a different lookup,
    # so it restarts at Result 1 instead of continuing to Result 2.
    assert searched.step("lookup[new mexico]")[0] == RESULT1
    assert searched.step("lookup[new mexico]")[0] == RESULT2
    assert searched.step("lookup[new mexico]")[0] == "No more results.\n"
    assert searched.step("lookup[High Plains]")[0] == (
        "(Result 1 / 1) The eastern sector extends into the High Plains and is called "
        "the Central Plains orogeny."
    )
    assert searched.step("lookup[High Plains]")[0] == "No more results.\n"
    # 0 matches: the empty-list path, not exhaustion — no result ever comes back.
    assert searched.step("lookup[elevation]")[0] == "No more results.\n"


def test_lookup_list_is_not_capped(env, monkeypatch):
    """(b) construct_lookup_list keeps every match — no `[:5]` cap (notes.md:317)."""
    serve(monkeypatch, {HIT_URL: page(*[f"Sentence number {i} mentions Kappa here." for i in range(8)])})
    env.step("search[Colorado orogeny]")
    assert env.step("lookup[Kappa]")[0] == "(Result 1 / 8) Sentence number 0 mentions Kappa here.."


def test_successful_search_resets_the_lookup_cursor(env, monkeypatch):
    """A *successful* search clears lookup_* (wikienv.py:122) — page A must not bleed into B."""
    serve(monkeypatch, {
        HIT_URL: fixture("colorado_orogeny_hit.html"),
        SIMILAR_URL: page("Gamma six seven New Mexico fresh page."),
    })
    env.step("search[Colorado orogeny]")
    env.step("lookup[New Mexico]")
    env.step("lookup[New Mexico]")
    env.step("search[Colorado orogenyy]")
    assert (env.lookup_keyword, env.lookup_list, env.lookup_cnt) == (None, None, None)
    assert env.step("lookup[New Mexico]")[0] == (
        "(Result 1 / 1) Gamma six seven New Mexico fresh page.."
    )


def test_block_scrape_keeps_lists_and_drops_short_blocks(env, monkeypatch):
    """`find_all("p") + find_all("ul")` and the `> 2`-token filter (notes.md:257,276).

    Also pins two things the block loop must NOT do: deduplicate repeated blocks (real
    pages repeat nav/footer lists, and the duplicate moves both the 5-sentence window and
    the `(Result i / n)` denominator), and leave the inner `.strip()` off each sentence
    (a double space after a period would survive into the observation byte-for-byte).
    """
    serve(monkeypatch, {HIT_URL: (
        "<html><body><p>Short block.</p>"
        "<p>A real sentence with many words here.</p>"
        "<p>Alpha one two three.  Beta four five six.</p>"
        "<ul><li>Nav item New Mexico link here.</li></ul>"
        "<ul><li>Nav item New Mexico link here.</li></ul></body></html>"
    )})
    obs, _, _ = env.step("search[Colorado orogeny]")
    assert obs == (
        "A real sentence with many words here.. Alpha one two three. "
        "Beta four five six.. Nav item New Mexico link here.. "
        "Nav item New Mexico link here.."
    )
    assert env.step("lookup[New Mexico]")[0] == (
        "(Result 1 / 2) Nav item New Mexico link here.."
    )


def test_search_miss_lists_similar_titles(searched, monkeypatch):
    """(c) the literal this recorded file produces, never a live value (D19)."""
    env = searched
    before = env.page
    env.step("lookup[New Mexico]")
    serve(monkeypatch, {SIMILAR_URL: fixture("colorado_orogenyy_similar.html")})
    obs, done, _ = env.step("search[Colorado orogenyy]")
    assert obs == (
        "Could not find Colorado orogenyy. Similar: ['Colorado orogeny', "
        "'Laramide orogeny', 'Sevier orogeny', 'Colorado Mineral Belt', "
        "'Wyoming Craton']."
    )
    assert done is False
    assert len(env.result_titles) == 20  # all kept, only 5 reach the observation
    # Miss semantics (notes.md:306): a failed search clears neither page nor lookup_*.
    assert env.page == before
    assert (env.lookup_keyword, env.lookup_cnt) == ("New Mexico", 1)
    assert env.step("lookup[New Mexico]")[0] == RESULT2
    # result_titles PERSIST across a later successful search — wikienv.py:115-122 touches
    # only page/lookup_* on a hit, and reset() does not clear them either.
    serve(monkeypatch, {
        "https://en.wikipedia.org/w/index.php?search=Fresh+page":
            page("Alpha one two three four five.")
    })
    env.step("search[Fresh page]")
    assert len(env.result_titles) == 20


def test_zero_result_page_is_an_ordinary_miss(searched, monkeypatch):
    """(d) D10/D23 — the miss literal with an empty list, and miss semantics."""
    before = (searched.page, searched.lookup_keyword, searched.lookup_cnt)
    searched.step("lookup[New Mexico]")
    # Sequenced after a Similar-list miss on purpose: the zero-result branch has to CLEAR
    # the 20 stale titles, or it offers the previous entity's results as this one's.
    serve(monkeypatch, {SIMILAR_URL: fixture("colorado_orogenyy_similar.html")})
    searched.step("search[Colorado orogenyy]")
    assert len(searched.result_titles) == 20
    serve(monkeypatch, {MISS_URL: fixture("nonexistent_miss.html")})
    obs, _, _ = searched.step("search[Qwertzuiop Zzyzxian Orogeny]")
    assert obs == "Could not find Qwertzuiop Zzyzxian Orogeny. Similar: []."
    assert searched.result_titles == []
    assert searched.page == before[0]
    assert (searched.lookup_keyword, searched.lookup_cnt) == ("New Mexico", 1)
    assert searched.lookup_list is not None
    # The failed search did not clear the page: lookup carries on where it left off.
    assert searched.step("lookup[New Mexico]")[0] == RESULT2


def test_lookup_after_reset(searched):
    """(e) all six fields cleared -> construct_lookup_list returns []."""
    searched.step("lookup[New Mexico]")  # leaves lookup_keyword/list/cnt populated
    searched.step("finish[Richard Nixon]")  # ...and `answer` set, which reset must clear
    assert searched.reset() == (
        "Interact with Wikipedia using search[], lookup[], and finish[].\n"
    )
    # The same keyword as before the reset: a surviving lookup_keyword would skip the
    # rebuild and replay page k's sentences into question k+1 (notes.md:205). A surviving
    # `answer` is worse — every later question would return done=True on step 0 carrying
    # question k's answer, so the whole run records n_steps=0 and one repeated prediction.
    assert searched.step("lookup[New Mexico]") == (
        "No more results.\n",
        False,
        {"steps": 1, "answer": None},
    )
    searched.reset()
    assert (
        searched.page,
        searched.lookup_keyword,
        searched.lookup_list,
        searched.lookup_cnt,
        searched.steps,
        searched.answer,
    ) == (None, None, None, None, 0, None)


def test_finish(env, monkeypatch):
    """(f)"""
    obs, done, info = env.step("finish[Richard Nixon]")
    assert obs == "Episode finished, reward = 0\n"
    assert done is True
    assert info["answer"] == "Richard Nixon"
    # Anything after a finish is a no-op that does not advance steps (wikienv.py:128).
    assert env.step("search[anything]") == (obs, True, info)

    # `finish[]` is routine under CLAUDE.md rule 4's 100-token cap (the answer truncated
    # away). An empty answer still ends the episode, so the guard is `is not None`, not
    # truthiness — otherwise that episode runs on to the step limit, billing every step.
    empty = WikiEnv()
    empty.reset()
    empty.step("finish[]")
    monkeypatch.setattr(empty, "_fetch", lambda url: page("one two three four"))
    assert empty.step("search[x]") == (
        "Episode finished, reward = 0\n",
        True,
        {"steps": 1, "answer": ""},
    )

    # The answer is the RAW bracket slice (wikienv.py:149), spaces included; whitespace
    # normalisation belongs to C2's EM scorer, not to the env.
    padded = WikiEnv()
    padded.reset()
    padded.step("finish[ Richard Nixon ]")
    assert padded.answer == " Richard Nixon "


def test_invalid_action(env, monkeypatch):
    """(g) plus dispatch: strip, case-sensitive prefixes, required trailing `]`."""
    # Patched first so that a mutant which *does* dispatch below never hits the network.
    monkeypatch.setattr(env, "_fetch", lambda url: page("one two three four"))
    obs, done, _ = env.step("blah")
    assert obs == "Invalid action: blah"
    assert done is False
    # think[] is not implemented here (notes.md:1120): it falls through to invalid.
    assert env.step("think[I should search]")[0] == "Invalid action: think[I should search]"
    # Prefixes are lowercase and case-sensitive (notes.md:196-199).
    assert env.step("Search[Colorado orogeny]")[0] == "Invalid action: Search[Colorado orogeny]"
    assert env.step("Lookup[New Mexico]")[0] == "Invalid action: Lookup[New Mexico]"
    # `Finish[` matters most of the three: a case regression there would silently change
    # which trajectories terminate early and which run to the step limit.
    assert env.step("Finish[Richard Nixon]")[0] == "Invalid action: Finish[Richard Nixon]"
    # A trailing `]` is required, or the entity would silently lose its last character.
    # Under the 100-token cap a truncated action is routine, and all three prefixes must
    # reject it: a truncated finish[ would end the episode on a clipped answer, and a
    # truncated lookup[ would answer for a keyword the model never asked about.
    assert env.step("search[Colorado orogeny")[0] == "Invalid action: search[Colorado orogeny"
    assert env.step("lookup[New Mexico")[0] == "Invalid action: lookup[New Mexico"
    assert env.step("finish[Richard Nixon")[0] == "Invalid action: finish[Richard Nixon"
    assert env.answer is None
    # action.strip() first: padding dispatches, and the invalid literal is the stripped action.
    assert env.step("  search[Colorado orogeny]  ")[0] == "one two three four."
    assert env.step("  blah  ")[0] == "Invalid action: blah"


def test_fetch_is_cached_on_disk(env, monkeypatch):
    """(i)"""
    calls = serve(monkeypatch, {HIT_URL: fixture("colorado_orogeny_hit.html")})
    first = env._fetch(HIT_URL)
    second = env._fetch(HIT_URL)
    assert len(calls) == 1
    assert first == second
    cached = wiki_env.CACHE_DIR / (hashlib.sha256(HIT_URL.encode()).hexdigest() + ".json")
    record = json.loads(cached.read_text(encoding="utf-8"))
    # The full record shape D18 pins. `fetched_at` is the only provenance stamp on a
    # permanent cache: without it there is no way to tell which pages predate markup drift.
    assert set(record) == {"url", "fetched_at", "html"}
    assert (record["url"], record["html"]) == (HIT_URL, first)


def test_cache_io_is_explicitly_utf8(env, monkeypatch):
    """(i) both halves of the round-trip name the encoding, never the platform locale.

    Untestable through behaviour on a UTF-8 machine, so it is pinned at the call: a
    locale-default write mangles non-ASCII HTML into a cache file that then serves every
    later run (CLAUDE.md rule 5) until someone deletes it by hand.
    """
    seen = []
    write_text, read_text = Path.write_text, Path.read_text
    monkeypatch.setattr(Path, "write_text", lambda self, data, **kw: (
        seen.append(kw.get("encoding")) or write_text(self, data, **kw)))
    monkeypatch.setattr(Path, "read_text", lambda self, **kw: (
        seen.append(kw.get("encoding")) or read_text(self, **kw)))
    serve(monkeypatch, {HIT_URL: page("Café orogeny one two three four.")})
    env._fetch(HIT_URL)  # writes
    env._fetch(HIT_URL)  # reads back
    assert seen == ["utf-8", "utf-8"]


def test_default_cache_dir():
    """(i) the store D18 pins, so C1 and C3 cannot invent two different ones."""
    root = Path(wiki_env.__file__).resolve().parents[1]
    assert wiki_env.CACHE_DIR == root / "data" / "cache" / "wiki"


def test_fetch_sends_the_user_agent(env, monkeypatch):
    """(j) a bare UA is 403'd today; the D18 header must be on the request."""
    seen = []

    def fake_get(url, headers=None, timeout=None):
        seen.append(headers)
        if (headers or {}).get("User-Agent") != USER_AGENT:
            return FakeResponse("<html>403</html>", status_code=403)
        return FakeResponse(fixture("colorado_orogeny_hit.html"))

    monkeypatch.setattr(wiki_env.requests, "get", fake_get)
    assert env.step("search[Colorado orogeny]")[0] == HIT_OBS
    assert seen == [{"User-Agent": USER_AGENT}]
    # Pinned against the literal, not the imported constant: every fixture in tests/ was
    # recorded with this exact string (D18) and comparing USER_AGENT to itself pins nothing.
    assert USER_AGENT == "react-langgraph-repro/0.1 (research reproduction; contact via repo)"


def test_fetch_retries_timeouts(env, monkeypatch):
    """(k) 3 timeouts then a body -> one observation, exactly 4 attempts."""
    attempts = []

    def fake_get(url, headers=None, timeout=None):
        attempts.append(timeout)
        if len(attempts) <= 3:
            raise requests.exceptions.Timeout()
        return FakeResponse(fixture("colorado_orogeny_hit.html"))

    monkeypatch.setattr(wiki_env.requests, "get", fake_get)
    assert env.step("search[Colorado orogeny]")[0] == HIT_OBS
    # Without a timeout= on the request, Timeout is unraisable and this loop is dead; and
    # the D18 value itself is pinned, since a too-short one burns 10 attempts and then
    # kills the run on the first cold fetch, where no fixture or cache can mask it.
    assert attempts == [30, 30, 30, 30]


def test_fetch_raises_after_ten_timeouts(env, monkeypatch):
    """(k) 10 consecutive timeouts -> a raised error, not None."""
    attempts = []

    def fake_get(url, headers=None, timeout=None):
        attempts.append(requests.exceptions.Timeout("read timed out: " + url))
        raise attempts[-1]

    monkeypatch.setattr(wiki_env.requests, "get", fake_get)
    with pytest.raises(requests.exceptions.Timeout) as excinfo:
        env.step("search[Colorado orogeny]")
    assert len(attempts) == 10
    # It re-raises the LAST REAL Timeout, not a fresh blank one: the traceback is the only
    # record of which URL died, and a blank Timeout makes a dead page indistinguishable
    # from a dead network when a cold run aborts hours in.
    assert excinfo.value is attempts[-1]
    assert str(excinfo.value) == "read timed out: " + HIT_URL


def test_fetch_raises_on_http_error(env, monkeypatch):
    """A 5xx must abort, never be parsed as an article and written to the disk cache."""
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return FakeResponse("<html><body><p>Wikimedia error one two three</p></body></html>", 500)

    monkeypatch.setattr(wiki_env.requests, "get", fake_get)
    with pytest.raises(requests.exceptions.HTTPError):
        env.step("search[Colorado orogeny]")
    assert not list(wiki_env.CACHE_DIR.glob("*.json"))
    # The retry loop catches Timeout only: an HTTP error aborts on the first attempt
    # rather than costing 10 requests per permanent 404.
    assert len(calls) == 1


def test_clean_str_converts_escape_sequences(env, monkeypatch):
    """(l) BLOCKER — the unicode-escape round-trip itself, not just its guard."""
    assert wiki_env.clean_str(r"a\nb") == "a\nb"
    serve(monkeypatch, {HIT_URL: page(r"Alpha one two\nBeta three four five.")})
    # The literal two-char `\n` becomes a real newline -> a paragraph break -> two
    # sentences. A no-op clean_str leaves `Alpha one two\nBeta ...` in the observation.
    assert env.step("search[Colorado orogeny]")[0] == "Alpha one two. Beta three four five.."


def test_result_titles_are_cleaned(env, monkeypatch):
    """wikienv.py:108 runs each Similar title through clean_str too."""
    serve(monkeypatch, {HIT_URL: (
        r'<html><body><div class="mw-search-result-heading">Caf\xc3\xa9 orogeny</div></body></html>'
    )})
    obs, _, _ = env.step("search[Colorado orogeny]")
    assert obs == "Could not find Colorado orogeny. Similar: ['Café orogeny']."
    # The echoed entity, by contrast, is the raw action text (wikienv.py:109): rewriting
    # it would show the model a search it never issued and send it round in circles.
    serve(monkeypatch, {
        wiki_env.SEARCH_URL + r"Caf\xc3\xa9":
            '<html><body><div class="mw-search-result-heading">Only hit</div></body></html>'
    })
    assert env.step(r"search[Caf\xc3\xa9]")[0] == r"Could not find Caf\xc3\xa9. Similar: ['Only hit']."


def test_no_results_phrase_must_start_the_page(env, monkeypatch):
    """(d) D23 detection is `startswith`: an article merely mentioning it is not a miss."""
    serve(monkeypatch, {HIT_URL: page(
        "Lead sentence about a page.",
        "There were no results matching the query but this is an article.",
    )})
    obs, _, _ = env.step("search[Colorado orogeny]")
    assert obs == (
        "Lead sentence about a page.. "
        "There were no results matching the query but this is an article.."
    )
    # ...and it is the WHOLE phrase, not a prefix of it: an article that happens to open
    # "There were no results ..." is still an article.
    serve(monkeypatch, {SIMILAR_URL: page(
        "There were no results reported for the 1994 election in this district."
    )})
    assert env.step("search[Colorado orogenyy]")[0] == (
        "There were no results reported for the 1994 election in this district.."
    )


def test_clean_str_guard(env, monkeypatch, caplog):
    """(l) MIN1 — a stray backslash escape kills the bare wikienv.py:10-11 body."""
    block = r"path C:\xyz here"
    serve(monkeypatch, {HIT_URL: page(block)})
    obs, _, _ = env.step("search[Colorado orogeny]")
    assert block in obs  # the block survives unconverted
    with pytest.raises(UnicodeDecodeError):  # what the unguarded reference does
        block.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    assert wiki_env.clean_str(block) == block
    encodable = r"literal \u4e2d escape"
    with pytest.raises(UnicodeEncodeError):  # the other half of the guard
        encodable.encode().decode("unicode-escape").encode("latin1")
    assert wiki_env.clean_str(encodable) == encodable
    # The guard is scoped to exactly those two types. A bare `except` would turn any other
    # defect (a non-str block, a BeautifulSoup type change) into a silently wrong page.
    with pytest.raises(AttributeError):
        wiki_env.clean_str(b"bytes block")
    # MIN1 is "return unchanged AND log": without the warning the degradation leaves no
    # trace in the run, calls.csv or the JSONL, so the EM loss is unattributable after.
    with caplog.at_level(logging.WARNING):
        wiki_env.clean_str(block)
    assert "clean_str" in caplog.text


def test_disambiguation_is_capped_at_depth_one(env, monkeypatch):
    """(m) D8/MIN2 — one retry, then take that page's own result.

    The first stub carries a hatnote before the "may refer to:" paragraph, as real
    disambiguation pages do: the trigger is `any(...)` over every block, not block[0].
    """
    calls = serve(
        monkeypatch,
        {
            HIT_URL: page(
                "For other uses see the hatnote here.",
                "Colorado orogeny may refer to: one of several things",
            ),
            "https://en.wikipedia.org/w/index.php?search=[Colorado+orogeny]": page(
                "This retry page may refer to: further ambiguous things"
            ),
        },
    )
    obs, _, _ = env.step("search[Colorado orogeny]")
    assert len(calls) == 2  # no third search
    assert obs == "This retry page may refer to: further ambiguous things."


def test_may_refer_to_needs_its_colon(env, monkeypatch):
    """(m) D8 fires on the literal "may refer to:" — the same words as ordinary prose are
    not a disambiguation page, and must not cost a wasted fetch and the wrong article."""
    calls = serve(monkeypatch, {HIT_URL: page(
        "The term may refer to several distinct historical events in Europe."
    )})
    obs, _, _ = env.step("search[Colorado orogeny]")
    assert obs == "The term may refer to several distinct historical events in Europe.."
    assert len(calls) == 1


@pytest.mark.slow
def test_live_search_smoke(monkeypatch, tmp_path):
    """(h) live, asserting only the stable parts (D19)."""
    # Throwaway cache dir, or the committed data/cache/wiki serves this test offline
    # and it can never see markup drift, a UA 403, or a ranking change.
    monkeypatch.setattr(wiki_env, "CACHE_DIR", tmp_path / "live")
    env = WikiEnv()
    env.reset()
    obs, done, _ = env.step("search[Colorado orogeny]")
    assert "orogeny" in obs and done is False
    env.step("search[Colorado orogenyy]")
    assert len(env.result_titles) == 20
    assert env.result_titles[0] == "Colorado orogeny"
