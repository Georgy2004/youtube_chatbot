import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

from core import (SearchIndex, answer_question, extract_video_id, fetch_transcript,
                  friendly_error, make_chunks)

load_dotenv()
st.set_page_config(page_title="Video Notes | YouTube Chat", page_icon="▶", layout="wide")


def setting(name, default=""):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default


def clear_video():
    for key in ("video", "index", "index_signature", "messages"):
        st.session_state.pop(key, None)


def stamp(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes}:{seconds:02}"


def show_sources(sources):
    if sources:
        with st.expander("View supporting transcript passages"):
            for i, source in enumerate(sources, 1):
                metadata = source["metadata"]
                url = f'https://www.youtube.com/watch?v={metadata["video_id"]}&t={int(metadata["start"])}s'
                st.markdown(f'**[{i}] [{stamp(metadata["start"])} – {stamp(metadata["end"])}]({url})**')
                st.text(source["text"])


st.title("Video Notes")
st.caption("Ask questions about a YouTube video. Check answers against timestamped captions.")

with st.sidebar:
    st.header("Connection")
    entered_key = st.text_input("OpenRouter API key", type="password", key="api_key_input",
                                help="Leave blank to use your .env or Streamlit secret.")
    api_key = entered_key.strip() or setting("OPENROUTER_API_KEY")
    st.caption("Your key stays in this session. Requests go to OpenRouter and its model providers.")
    chat_model = st.text_input("Chat model", value=setting("CHAT_MODEL", "openrouter/free")).strip()
    st.header("Search")
    mode_label = st.selectbox("Retrieval method", ["Local keyword search (no embedding API)",
                                                   "Semantic search (FAISS + API embeddings)"])
    mode = "semantic" if mode_label.startswith("Semantic") else "keyword"
    embedding_model = setting("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    if mode == "semantic":
        embedding_model = st.text_input("Embedding model", value=embedding_model).strip()
        st.warning("Embedding requests can require credits, even with a free chat model.")
    else:
        st.caption("Lightweight local search; no model download. Use the caption language for best results. Cross-language semantic matching is not supported in this mode.")
    k = st.slider("Passages per answer", 2, 8, 4)
    response_language = st.text_input("Answer language", value="Same language as the question", max_chars=80)
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state["messages"] = []
    if st.button("Unload video", use_container_width=True):
        clear_video()
    st.caption("Free chat models have quotas and may be unavailable. No automatic switch to paid models.")

with st.form("video_form"):
    url = st.text_input("YouTube URL or video ID", placeholder="https://www.youtube.com/watch?v=...")
    preference = st.text_input("Preferred caption language code (optional)",
                               help="For example: ml, ta, kn, es, fr. Leave blank to prefer English, then any available track.")
    load = st.form_submit_button("Load captions", type="primary")

if load:
    try:
        video_id = extract_video_id(url)
        with st.spinner("Fetching captions…"):
            data = fetch_transcript(video_id, preference)
            docs = make_chunks(data["rows"], video_id)
        # Commit a new video only after fetching and chunking succeed.
        clear_video()
        st.session_state["video"] = {**data, "id": video_id, "docs": docs}
        st.success("Captions loaded. Build the search index below to start chatting.")
    except Exception as error:
        st.error(friendly_error(error))
        if "video" in st.session_state:
            st.info("The previously loaded video remains active below.")

video = st.session_state.get("video")
if not video:
    st.info("Start by loading a video with captions. Any available caption language is supported.")
    st.stop()

left, right = st.columns([2, 1])
with left:
    st.subheader("Loaded video")
    st.video(f'https://www.youtube.com/watch?v={video["id"]}')
    st.markdown(f'[Open video on YouTube](https://www.youtube.com/watch?v={video["id"]})')
    st.caption("If playback is restricted by the video owner, use the YouTube link above.")
    st.caption(f'Video ID: {video["id"]} · Captions: {video["language"]} ({video["code"]}) · '
               f'{"Auto-generated" if video["automatic"] else "Manual"}')
with right:
    st.metric("Transcript chunks", len(video["docs"]))
transcript = " ".join(row["text"] for row in video["rows"])
with st.expander("Read or download transcript"):
    st.text(transcript[:20_000])
    if len(transcript) > 20_000:
        st.caption("Preview limited to 20,000 characters. Download includes the complete transcript.")
    st.download_button("Download transcript", transcript,
                       file_name=f'{video["id"]}_transcript.txt', mime="text/plain")

# Index reuse is private to this browser session. Keys are never global cache arguments.
signature = (video["id"], video["code"], mode, embedding_model if mode == "semantic" else "",
             hashlib.sha256(api_key.encode()).hexdigest() if mode == "semantic" else "")
ready = st.session_state.get("index_signature") == signature
if not ready:
    st.info("Build the search index for the selected retrieval settings.")
if st.button("Build search index", disabled=ready):
    try:
        with st.spinner("Building search index…"):
            index = SearchIndex(video["docs"], mode, api_key, embedding_model)
        st.session_state["index"] = index
        st.session_state["index_signature"] = signature
        st.session_state["messages"] = []
        st.rerun()
    except Exception as error:
        st.error(friendly_error(error))

st.subheader("Conversation")
messages = st.session_state.setdefault("messages", [])
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

if not api_key:
    st.info("Add your OpenRouter API key in the sidebar or .env to generate answers.")
question = st.chat_input("Ask a question about the video…", disabled=not ready or not api_key or not chat_model,
                         max_chars=2000)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.spinner("Finding passages and preparing an answer…"):
            answer, sources = answer_question(
                st.session_state["index"], question, messages, api_key,
                chat_model, response_language or "the question's language", k
            )
        source_rows = [{"text": d.page_content, "metadata": d.metadata} for d in sources]
        messages.extend([{"role": "user", "content": question},
                         {"role": "assistant", "content": answer, "sources": source_rows}])
        # Bounded per-session conversation storage.
        st.session_state["messages"] = messages[-40:]
        st.rerun()
    except Exception as error:
        st.error(friendly_error(error))
        st.caption("Your question was not saved. You can submit it again after fixing the issue.")

if messages:
    chat_text = "\n\n".join(f'{m["role"].upper()}: {m["content"]}' for m in messages)
    st.download_button("Download conversation", chat_text, file_name="video_chat.txt", mime="text/plain")
st.caption("Answers depend on caption quality and retrieval. Check the linked passages; citations are model-generated.")
