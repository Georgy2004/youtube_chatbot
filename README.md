# YouTube Chatbot — Streamlit edition

A runnable, production-oriented starter based on your transcript → chunks → retrieval → LLM pipeline. It is not a fully hardened public multi-user service.

## Run on Windows (Python 3.11 recommended)

Extract this folder, open it in VS Code, and use its terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `OPENROUTER_API_KEY=your_key_here`. Alternatively enter the key in the password field in the app. Then:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed in the terminal (usually http://localhost:8501). Activation is unnecessary, so PowerShell execution-policy issues do not block these commands.

1. Paste a video link or ID and optionally enter a caption language code.
2. Click **Load captions**. Blank language prefers English, then any available language; manual tracks are preferred within that choice. This is selection, not translation.
3. Choose retrieval mode and click **Build search index**.
4. Ask questions. Expand source passages and use their timestamp links to verify answers.
   An embedded YouTube player above the chat lets you watch the loaded video on the same page. Some videos restrict embedded playback; use the Open video on YouTube link in that case. The chatbot still uses captions, not visual video analysis.
5. Clear conversation to start fresh on the same index, or unload/change the video to reset it.

## Retrieval and costs

The default local TF-IDF keyword mode needs no embedding API, GPU, or downloaded model. It uses character n-grams for multilingual text, but cannot translate queries or reliably match synonyms across languages. Ask in the caption language for best results.

The optional semantic mode keeps your FAISS design and uses `openai/text-embedding-3-small` through OpenRouter. Indexing and each retrieval can incur embedding charges. Semantic quality varies by language. There is no silent fallback to paid chat models.

The default chat model is `openrouter/free`, OpenRouter's free-model router. You can enter a specific currently available model ID. Provider quotas, availability and language quality vary. Follow-up questions make an additional model call to resolve references before retrieval. New independent questions after clearing the conversation need one chat call.

## Improvements over the original

- Real chat input and visible answers; no fixed `"Whats your query:"` string.
- Video ID validation with host allowlist; URL Markdown/escape errors removed.
- Automatic caption language selection or explicit preference.
- Timestamp metadata preserved through splitting; numbered retrieved passages.
- Session-private index reused across questions; changing video or retrieval configuration resets the applicable state.
- Follow-up question rewriting, bounded history, clear/reset and downloads.
- API timeouts/retries, readable caption/auth/credit/rate-limit errors, no raw provider payloads displayed.
- Separate UI and retrieval modules; secrets via environment, Streamlit secrets or session password input.
- 300,000 transcript character limit, 2,000 question character limit, bounded response and retrieval sizes.
- Transcript treated as untrusted data in prompts. This reduces prompt injection risk but does not guarantee grounded answers.

## Files

- `app.py`: Streamlit interface and session state.
- `core.py`: URL parsing, captions, chunks, retrieval and answer generation.
- `requirements.txt`: dependency ranges.
- `requirements-tested.txt`: exact versions installed during validation (if supplied).
- `.env.example`: configuration template; never commit your real `.env`.
- `tests/test_app.py`: offline regression and Streamlit UI tests.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use fixtures/mocks for external services. They do not prove current provider availability or YouTube accessibility. Actual caption fetches and model/embedding calls require live testing with your own key; no key is included.

Validation performed: 16 tests passed using Python 3.12 on Linux; `pip check` reported no broken requirements. Windows setup instructions are provided but were not executed on Windows. `requirements-tested.txt` records that Linux environment; use `requirements.txt` for a fresh Windows install so pip can resolve compatible platform versions. The installed LangChain community package emits a deprecation warning; this starter retains your FAISS integration, and a maintained integration migration should be reviewed before long-term public deployment.

## Deployment limits

This version is suitable for local learning, portfolio demos and a controlled pilot. Before exposing it publicly, add authentication, per-user quotas and spending budgets, secret management, telemetry with redaction, dependency/security review, and load testing. Server-wide keys in a public deployment let visitors consume the owner's API quota. Prefer per-user credentials or enforce quotas before enabling a shared key.

YouTube may block cloud-hosted transcript requests. Captions must already exist: there is no audio transcription fallback. The app reports blocks; it does not promise access to every video. It only knows transcript text, not visuals.

All indexes and history live in browser-session-associated server RAM and disappear on session loss/restart. They are not globally cached or persisted. Download your transcript/chat if needed. Caption text is sent to the embedding provider in semantic mode; retrieved passages and recent conversation context are sent to the chat provider. Session credentials are held in server memory, not stored on disk by the app.

Similarity search is designed for specific questions. Four retrieved passages cannot reliably summarize an entire long video. Whole-video summarization and evaluation across languages remain separate enhancements. Citations reflect retrieved passages and require human verification.

## API references

- https://github.com/jdepoix/youtube-transcript-api
- https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings
- https://openrouter.ai/openrouter/free
- https://docs.streamlit.io/develop/api-reference/app-testing
