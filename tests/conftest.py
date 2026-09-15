"""No test may ever spend money.

Added after the lead's own test issued 42 unauthorised live calls ($0.003318):
`fake` is a FACTORY fixture, it was used as though it were the FakeLLM instance,
so `llm.complete` was never patched and the real client ran 21 samples per case.

The per-test discipline (patch `llm.complete` or `llm._request`) is easy to get
subtly wrong. This makes the failure mode loud instead: the network path itself
is severed for the whole suite, so a test that forgets to patch ERRORS rather
than quietly billing. A test that genuinely wants the real client must opt in
explicitly with @pytest.mark.slow, which pytest.ini deselects by default.
"""
import pytest

from src import llm


@pytest.fixture(autouse=True)
def _no_real_api_calls(request, monkeypatch):
    # `slow` = the live smoke tests, deselected by default in pytest.ini.
    # `builds_client` = constructs a real client to INSPECT its payload and
    # sends nothing; that is not a spend path, so it is allowed through.
    if "slow" in request.keywords or "builds_client" in request.keywords:
        return

    def _refuse():
        raise RuntimeError(
            "This test reached the real LLM client. Patch llm.complete or "
            "llm._request, or mark the test @pytest.mark.slow if it is meant "
            "to hit the network. (tests/conftest.py)"
        )

    monkeypatch.setattr(llm, "_client", _refuse)
