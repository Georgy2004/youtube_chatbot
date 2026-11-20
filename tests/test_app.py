import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core
from streamlit.testing.v1 import AppTest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage

VIDEO_ID = "Gfr50f6ZBvo"
ROWS = [
    {"text": "AlphaFold predicts protein structures using machine learning. " * 15, "start": 0.1, "duration": 30},
    {"text": "Chess engines evaluate board positions and choose moves. " * 15, "start": 30.1, "duration": 30},
]
DATA = {"rows": ROWS, "language": "English", "code": "en", "automatic": False}


@pytest.mark.parametrize("url", [VIDEO_ID, f"https://youtu.be/{VIDEO_ID}?si=x",
    f"youtube.com/watch?v={VIDEO_ID}&t=20", f"https://m.youtube.com/shorts/{VIDEO_ID}",
    f"https://www.youtube.com/live/{VIDEO_ID}"])
def test_urls(url):
    assert core.extract_video_id(url) == VIDEO_ID


@pytest.mark.parametrize("url", ["", "not-an-id", f"https://youtube.com.evil.com/watch?v={VIDEO_ID}",
    f"https://evil.com/{VIDEO_ID}", f"ftp://youtube.com/watch?v={VIDEO_ID}",
    f"https://user@youtube.com/watch?v={VIDEO_ID}"])
def test_invalid_urls(url):
    with pytest.raises(ValueError):
        core.extract_video_id(url)


def test_chunk_times_and_keyword_retrieval():
    docs = core.make_chunks(ROWS, VIDEO_ID)
    assert len(docs) > 1
    assert all(len(d.page_content) <= 1000 for d in docs)
    assert docs[0].metadata["start"] == 0.1
    assert docs[-1].metadata["end"] == 60.1
    index = core.SearchIndex(docs, "keyword")
    assert "Chess" in index.search("chess engines")[0].page_content
    assert index.search("ZZZZZZZZ") == []


def test_faiss_without_network():
    with patch.object(core, "OpenAIEmbeddings", return_value=DeterministicFakeEmbedding(size=32)):
        index = core.SearchIndex(core.make_chunks(ROWS, VIDEO_ID), "semantic", "test", "fake")
    assert len(index.search("protein", k=2)) == 2


def track(code, generated=False):
    return SimpleNamespace(language_code=code, language=code, is_generated=generated,
                           fetch=lambda: [SimpleNamespace(**ROWS[0])])


def test_language_selection():
    tracks = [track("ml"), track("en", True), track("en")]
    with patch.object(core, "YouTubeTranscriptApi") as api:
        api.return_value.list.return_value = tracks
        assert core.fetch_transcript(VIDEO_ID)["automatic"] is False
        assert core.fetch_transcript(VIDEO_ID, "ml")["code"] == "ml"
        with pytest.raises(ValueError, match="Available codes"):
            core.fetch_transcript(VIDEO_ID, "fr")
        api.return_value.list.return_value = [track("ta")]
        assert core.fetch_transcript(VIDEO_ID)["code"] == "ta"


def test_followup_and_grounded_prompt():
    calls = []
    def respond(prompt):
        calls.append(prompt.to_string())
        return AIMessage(content="What does AlphaFold predict?" if len(calls) == 1 else "Protein structures [1].")
    index = core.SearchIndex(core.make_chunks(ROWS, VIDEO_ID), "keyword")
    with patch.object(core, "ChatOpenAI", return_value=RunnableLambda(respond)):
        answer, sources = core.answer_question(index, "What does it predict?", [
            {"role": "user", "content": "Explain AlphaFold"},
            {"role": "assistant", "content": "AlphaFold is discussed."}], "test", "fake", "English")
    assert len(calls) == 2
    assert "What does AlphaFold predict?" in calls[1]
    assert "untrusted data" in calls[1]
    assert sources and answer == "Protein structures [1]."


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_ui_full_flow_and_reset():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    assert app.chat_input == []
    app.text_input(key="api_key_input").set_value("test-key")
    next(w for w in app.text_input if w.label == "YouTube URL or video ID").set_value(VIDEO_ID)
    with patch.object(core, "fetch_transcript", return_value=DATA):
        button(app, "Load captions").click().run()
    assert not app.exception
    button(app, "Build search index").click().run()
    assert not app.exception
    assert not app.chat_input[0].disabled
    cached = app.session_state["index"]
    with patch.object(core, "answer_question", return_value=("Protein structures [1].", cached.docs[:1])):
        app.chat_input[0].set_value("What is AlphaFold?").run()
    assert not app.exception
    assert len(app.session_state["messages"]) == 2
    assert app.session_state["index"] is cached
    with patch.object(core, "answer_question", side_effect=RuntimeError("SECRET MUST NOT DISPLAY")):
        app.chat_input[0].set_value("Another question").run()
    assert "SECRET" not in app.error[0].value
    assert len(app.session_state["messages"]) == 2
    button(app, "Clear conversation").click().run()
    assert not app.session_state["messages"]
    # A failed load retains the current video, and tells the user.
    with patch.object(core, "fetch_transcript", side_effect=ValueError("No captions")):
        button(app, "Load captions").click().run()
    assert app.session_state["video"]["id"] == VIDEO_ID
    assert app.error
    button(app, "Unload video").click().run()
    assert not app.exception
    assert "video" not in app.session_state
