"""Checks on the OpenAI-compatible path that cannot be driven from the API.

1. Citation resolution is a grounding boundary: a [n] the model invented must
   produce NO source rather than the wrong one. Only the live model can emit a
   bad marker, so _chat is stubbed to emit one deliberately.
2. run_named passes model-supplied arguments straight into the tools, so the
   path-escape guard has to hold there too (checked live in test_agent_escape).

Run: python test_citations.py
"""

import app
from rag import Hit

HITS = [Hit(path=f"note{i}.md", heading="H", text=f"body {i}", tags=[], score=0.5 - i * 0.01)
        for i in range(1, 6)]  # five hits -> valid markers are [1]..[5]


class _Stub:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None


def _with_answer(text, fn):
    original = app._chat
    app._chat = lambda *a, **k: (_Stub(text), "stub-model")
    try:
        return fn()
    finally:
        app._chat = original


def test_out_of_range_marker_yields_no_source():
    answer, sources, model = _with_answer(
        "Claim A [1]. Claim B [3]. Claim C [9]. Repeat of A [1].",
        lambda: app._openai_answer("q", HITS))
    paths = [s["path"] for s in sources]
    assert paths == ["note1.md", "note3.md"], paths      # [9] dropped, [1] deduped
    assert all(s["cited"] for s in sources)
    assert model == "stub-model"


def test_zero_and_huge_markers_are_dropped():
    _, sources, _ = _with_answer("Bad [0] and worse [999].",
                                 lambda: app._openai_answer("q", HITS))
    assert sources == [], sources


def test_uncited_answer_falls_back_to_retrieved_hits():
    # /ask returns `sources or _sources(hits)`, so an uncited answer still shows
    # what retrieval found -- but never marked cited:True.
    _, sources, _ = _with_answer("No markers at all.", lambda: app._openai_answer("q", HITS))
    assert sources == []
    assert len(app._sources(HITS)) == 5
    assert not any("cited" in s for s in app._sources(HITS))


def test_fullwidth_brackets_resolve():
    """NVIDIA Build's openai/gpt-oss-120b cites as 【1】 (U+3010/U+3011). An
    ASCII-only pattern silently returned zero sources for every answer."""
    _, sources, _ = _with_answer("Claim A 【1】 and claim B 【3】.",
                                 lambda: app._openai_answer("q", HITS))
    assert [s["path"] for s in sources] == ["note1.md", "note3.md"], sources


def test_mixed_bracket_styles_dedupe_together():
    _, sources, _ = _with_answer("A [2] then again 【2】.",
                                 lambda: app._openai_answer("q", HITS))
    assert [s["path"] for s in sources] == ["note2.md"], sources


def test_chat_falls_through_every_measured_failure_mode():
    """The fallback chain is what makes free models usable. Each failure below
    was observed against OpenRouter on 2026-09-02 and must advance the chain
    rather than reach the user."""
    import openai

    calls = []

    class _FakeResp:
        def __init__(self, code):
            self.status_code = code
            self.headers = {}
            self.request = None

    class _FakeChoice:
        def __init__(self, content, finish_reason="stop"):
            msg = _Stub(content)
            self.choices = [type("C", (), {"message": msg, "finish_reason": finish_reason})()]

    class FakeCompletions:
        def create(self, model, **kw):
            calls.append(model)
            if model == "rate-limited":
                raise openai.APIStatusError("429", response=_FakeResp(429), body=None)
            if model == "empty":
                return _FakeChoice("")                      # all budget spent thinking
            if model == "truncated":
                return _FakeChoice("Here's a thinking proc", finish_reason="length")
            if model == "artifact":
                return _FakeChoice("<|tool_call_start|>[read(path='/home/gibbon/x.md')]")
            return _FakeChoice("real answer")

    orig_oai, orig_models = app.oai, app.LLM_MODELS
    app.oai = type("O", (), {"chat": type("C", (), {"completions": FakeCompletions()})()})()
    app.LLM_MODELS = ["rate-limited", "empty", "truncated", "artifact", "good"]
    try:
        msg, model = app._chat([{"role": "user", "content": "x"}])
        assert calls == ["rate-limited", "empty", "truncated", "artifact", "good"], calls
        assert model == "good" and msg.content == "real answer"
    finally:
        app.oai, app.LLM_MODELS = orig_oai, orig_models


def test_chat_raises_when_whole_chain_fails():
    """Exhausted chain must raise, not return a garbage answer -- /ask turns it
    into a 502 naming the last failure."""
    class _FakeChoice:
        def __init__(self):
            self.choices = [type("C", (), {"message": _Stub(""), "finish_reason": "stop"})()]

    orig_oai, orig_models = app.oai, app.LLM_MODELS
    app.oai = type("O", (), {"chat": type("C", (), {"completions":
        type("X", (), {"create": lambda self, **kw: _FakeChoice()})()})()})()
    app.LLM_MODELS = ["a", "b"]
    try:
        app._chat([{"role": "user", "content": "x"}])
        raise AssertionError("exhausted chain should raise")
    except RuntimeError as e:
        assert "all models failed" in str(e), e
    finally:
        app.oai, app.LLM_MODELS = orig_oai, orig_models


def test_tool_calls_survive_a_length_finish_only_when_content_is_fine():
    """A truncated tool-calling turn is still unusable: the arguments JSON may be
    cut in half."""
    assert app._unusable(_Stub("ok"), "stop") is None
    assert app._unusable(_Stub(""), "stop") == "empty response"
    assert "truncated" in app._unusable(_Stub("half an ans"), "length")
    assert "artifact" in app._unusable(_Stub("<|tool_call_start|>x"), "stop")


def test_unknown_tool_name_raises_not_crashes():
    """run_named lives inside the /agent closure; rebuild its lookup shape here."""
    funcs = {"search_vault": lambda **k: "ok"}

    def run_named(name, args):
        fn = funcs.get(name)
        if fn is None:
            raise ValueError(f"unknown tool {name!r}")
        return fn(**args)

    try:
        run_named("rm_rf", {})
        raise AssertionError("unknown tool should raise")
    except ValueError as e:
        assert "unknown tool" in str(e)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all citation/fallback checks passed")
