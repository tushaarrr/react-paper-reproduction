"""The paper's Wikipedia environment (paper/notes.md section 2).

Behaviour is reference/wikienv.py:44-160 with the deviations notes.md pins:
D7 (3-tuple step), D8/MIN2 (disambiguation capped at depth 1), D10/D23 (zero-result
pages emit the ordinary miss literal with an empty list), MIN1 (clean_str guarded),
D18 (disk cache + retries + User-Agent), and no think[] action.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# D18. Fixed string: every recorded fixture in tests/fixtures/ was fetched with it.
USER_AGENT = "react-langgraph-repro/0.1 (research reproduction; contact via repo)"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "wiki"
MAX_ATTEMPTS = 10
SEARCH_URL = "https://en.wikipedia.org/w/index.php?search="
RESET_OBS = "Interact with Wikipedia using search[], lookup[], and finish[].\n"


def clean_str(p):
    """wikienv.py:10-11, guarded (MIN1): literal \\x / \\u sequences raise there."""
    try:
        return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        logging.warning("clean_str: leaving block unconverted: %s", p[:80])
        return p


class WikiEnv:
    def __init__(self):
        self.page = None
        self.obs = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        self.result_titles = []

    # -- state -------------------------------------------------------------

    def _get_info(self):
        return {"steps": self.steps, "answer": self.answer}

    def reset(self):
        """wikienv.py:44-57. All six fields, or page k leaks into question k+1."""
        self.obs = RESET_OBS
        self.page = None
        self.lookup_keyword = None
        self.lookup_list = None
        self.lookup_cnt = None
        self.steps = 0
        self.answer = None
        return self.obs

    # -- text pipeline (wikienv.py:59-87) ----------------------------------

    def construct_lookup_list(self, keyword):
        if self.page is None:
            return []
        return [s for s in _sentences(self.page) if keyword.lower() in s.lower()]

    @staticmethod
    def get_page_obs(page):
        return " ".join(_sentences(page)[:5])

    # -- fetch layer (D18) -------------------------------------------------

    def _fetch(self, url):
        """Disk cache keyed by the exact URL; up to MAX_ATTEMPTS tries on timeout."""
        path = CACHE_DIR / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["html"]
        for _ in range(MAX_ATTEMPTS):
            try:
                response = requests.get(
                    url, headers={"User-Agent": USER_AGENT}, timeout=30
                )
                response.raise_for_status()
                html = response.text
                break
            except requests.exceptions.Timeout as exc:
                timeout = exc
        else:
            raise timeout  # never None, unlike hotpotqa.ipynb:46-52
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"url": url, "fetched_at": time.time(), "html": html}),
            encoding="utf-8",
        )
        os.replace(tmp, path)  # atomic: an interrupt leaves no truncated entry
        return html

    # -- actions -----------------------------------------------------------

    def search_step(self, entity, depth=0):
        """wikienv.py:98-122. A miss leaves page / lookup_* untouched."""
        soup = BeautifulSoup(
            self._fetch(SEARCH_URL + entity.replace(" ", "+")), features="html.parser"
        )
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:
            self.result_titles = [clean_str(d.get_text().strip()) for d in result_divs]
            self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
            return
        blocks = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        if depth < 1 and any("may refer to:" in p for p in blocks):
            self.search_step("[" + entity + "]", depth + 1)  # D8, capped at depth 1
            return
        page = ""
        for p in blocks:
            if len(p.split(" ")) > 2:
                # wikienv.py:119 guards `p.endswith("\n")` here; blocks are already
                # .strip()ed at :111, so that guard is unreachable and is dropped.
                page += clean_str(p) + "\n"
        if page.startswith("There were no results matching the query"):  # D10/D23
            self.result_titles = []
            self.obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
            return
        self.page = page
        self.obs = self.get_page_obs(self.page)
        self.lookup_keyword = self.lookup_list = self.lookup_cnt = None

    def step(self, action):
        """wikienv.py:124-160, minus reward (D7) and minus think[] (notes.md section 2)."""
        done = False
        action = action.strip()
        if self.answer is not None:  # already finished; steps does not increment
            return self.obs, True, self._get_info()

        if action.startswith("search[") and action.endswith("]"):
            self.search_step(action[len("search["):-1])
        elif action.startswith("lookup[") and action.endswith("]"):
            keyword = action[len("lookup["):-1]
            if self.lookup_keyword != keyword:  # rebuilt only on an exact change
                self.lookup_keyword = keyword
                self.lookup_list = self.construct_lookup_list(keyword)
                self.lookup_cnt = 0
            if self.lookup_cnt >= len(self.lookup_list):
                self.obs = "No more results.\n"
            else:
                self.obs = (
                    f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) "
                    + self.lookup_list[self.lookup_cnt]
                )
                self.lookup_cnt += 1
        elif action.startswith("finish[") and action.endswith("]"):
            self.answer = action[len("finish["):-1]
            done = True
            self.obs = "Episode finished, reward = 0\n"  # reward is always 0 here
        else:
            self.obs = "Invalid action: {}".format(action)

        self.steps += 1
        return self.obs, done, self._get_info()


def _sentences(page):
    """wikienv.py:63-74 / :79-87 — the one split both functions run."""
    sentences = []
    for p in page.split("\n"):
        # wikienv.py:65 guards `if p.strip():` here; a blank line yields the single empty
        # sentence [""], which the filter below drops anyway, so that guard is dead too.
        sentences += p.strip().split(". ")
    return [s.strip() + "." for s in sentences if s.strip()]
