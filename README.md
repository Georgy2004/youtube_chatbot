# YouTube Chatbot

Ask a question about a YouTube video and jump to the part where the speaker discusses it. The app, called **Video Notes**, uses the video's captions to find relevant passages and write an answer. Each answer comes with a list of passages and timestamps you can check yourself.

For example, while watching a Python lesson, you could ask **“Why does the instructor use a virtual environment?”** Open **View supporting transcript passages** below the answer and click a timestamp to hear the explanation in context.

## Run the app on Windows

You’ll need Git, Python 3.11 (recommended), an internet connection, and an OpenRouter API key to generate answers.

1. Open PowerShell. Download the project and enter its folder. If you already have the project, open a terminal in its folder and skip this step.

   ```powershell
   git clone https://github.com/Georgy2004/youtube_chatbot.git
   cd youtube_chatbot
   ```

2. Create a Python environment for the project and install its packages.

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Start the app and open the local address printed in the terminal.

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

4. Enter your key in **OpenRouter API key** in the sidebar. To reuse it across local sessions, you can instead create a file named `.env` beside `app.py` and add the line below. Replace the placeholder on your computer and keep the file out of Git.

   ```dotenv
   OPENROUTER_API_KEY=your_key_here
   ```

## Ask your first question

1. Paste a YouTube link and click **Load captions**. Leave the caption language blank to prefer English and fall back to an available track. To request Malayalam captions, enter `ml` if the video offers that track.
2. Leave **Retrieval method** on **Local keyword search** for your first try. Click **Build search index** so the app can look up passages in this video.
3. Ask a specific question about something the speaker said. You can ask follow-ups in the same conversation. The default **Answer language** follows your question, and you can also enter a language explicitly.
4. Open **View supporting transcript passages** to compare the answer with the captions. Use **Download transcript** or **Download conversation** to keep a copy before the session ends.

## How it works

The app splits the captions into short passages and keeps their timestamps. When you ask a question, it finds relevant passages and sends them to a chat model through OpenRouter. The model uses those passages to compose an answer. This approach is called *retrieval-augmented generation* (RAG).

| Search option | When to use it | Cost considerations |
| --- | --- | --- |
| Local keyword search (TF-IDF) | Start here when your question uses words from the captions. Search runs on your computer without a GPU. | Finding passages uses no API credits. Generating the answer still uses OpenRouter. |
| Semantic search (FAISS + embeddings) | Try this when your wording differs from the speaker’s. It uses numerical representations called embeddings to find passages with similar meanings. | Creating the index and searching it use an embedding API and may require credits, even with a free chat model. |

For Malayalam captions and an English question, try semantic search and set **Answer language** to English. Finding the right passage across languages depends on the embedding model. With local keyword search, ask in the caption language for better matches.

## What to expect

- **Captions must be available.** The app reads an existing caption track. Details shown only on screen, such as a diagram the speaker never describes, are outside its scope. YouTube can also restrict access to captions.
- **Specific questions work best.** Each answer draws on a few passages. A request to summarize a long video may miss topics, and errors in automatic captions can carry into an answer.
- **Chat uses an external service.** Relevant passages and recent conversation context go to the chat provider through OpenRouter. Semantic search also sends caption text to the embedding provider. The default chat setting is `openrouter/free`, subject to model availability and quotas.
- **Downloads preserve your work.** The app keeps the search index and conversation in session memory. Download anything you want to keep before ending the session.

## Project files and tests

| File or folder | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, video player, chat, and downloads |
| `core.py` | Caption loading, timestamped passages, search, and answer generation |
| `requirements.txt` | Packages needed to run the app |
| `tests/` | Automated checks that use sample data in place of live services |

To run the tests from the project folder after installation:

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
```
