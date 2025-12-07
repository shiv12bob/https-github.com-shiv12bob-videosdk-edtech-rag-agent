## Voice RAG Agent – EdTech Sales 

This project implements an **end‑to‑end voice agent** using the **VideoSDK Agent SDK** with a **RAG (Retrieval‑Augmented Generation) pipeline** on top of a local vector database (**ChromaDB**).  
The assistant is specialized for **EdTech sales** use cases.

It satisfies the assignment requirements:
- **Agent SDK**: VideoSDK Agents with `DeepgramSTT → OpenAILLM → ElevenLabsTTS`.
- **RAG pipeline**: local ChromaDB, ingestion from `docs/`, retrieval + context to LLM.
- **Fallback**: if no relevant docs are found (low similarity), it falls back to a plain LLM answer.
- **Test flow**: easy to run console and voice agents with example queries.

---

## 1. Project Structure

- `ingest.py` – Ingests `./docs/*.txt` / `.md` into a local ChromaDB collection using `sentence-transformers`.
- `rag.py` – RAG core:
  - embeds queries,
  - runs nearest‑neighbor search on Chroma,
  - builds context,
  - calls OpenAI Chat Completions with RAG prompt (EdTech‑sales focused),
  - falls back to plain LLM when docs are not relevant.
- `agent.py` – Simple **text console agent** (type in terminal).
- `main_videosdk.py` – **VideoSDK Agent SDK** integration (STT → LLM → TTS voice agent).
- `docs/` – Source documents for RAG (add 3–4 small EdTech sales text files here).
- `requirements.txt` – Python dependencies.

---

## 2. Setup Instructions

### 2.1. Python environment

```bash
cd Assignment_Videosdk
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# or cmd
.venv\Scripts\activate.bat

pip install -r requirements.txt
```

> If `videosdk-agents` fails for your Python version, you can still fully run the **console RAG agent** (`agent.py`) and RAG pipeline; VideoSDK voice is an optional extra.

### 2.2. Environment variables (`.env`)

Create a `.env` file in the project root with:

```env
# OpenAI (required)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Speech / VideoSDK (optional but required for full voice agent)
DEEPGRAM_API_KEY=your_deepgram_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
VIDEOSDK_AUTH_TOKEN=your_videosdk_auth_token_here
VIDEOSDK_ROOM_ID=           # can be empty for playground mode

# RAG / Chroma
CHROMA_DIR=./chroma_db
SIMILARITY_THRESHOLD=0.65   # lower = more likely to use docs
RAG_TOP_K=3
```

> **Important:** The OpenAI key must have quota. If you see a `429 insufficient_quota` error, update your billing or use a different key.

### 2.3. Prepare docs for RAG (EdTech sales use cases)

1. Open the `docs/` folder.
2. Add 3–4 small `.txt` or `.md` files describing **EdTech sales** scenarios, e.g.:
   - `product_overview.txt` – features and value prop of your EdTech product.
   - `sales_playbook.txt` – discovery questions, objection handling, follow‑up templates.
   - `pricing_and_offers.txt` – pricing tiers, discounts, common negotiation patterns.
3. Keep each file reasonably small (a few KB).

---

## 3. RAG Pipeline Details

### 3.1. Ingestion (`ingest.py`)

- Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to embed chunks.
- Streams through files in `./docs/`, checks they’re text, splits into overlapping chunks, and upserts into **ChromaDB**.

Key pieces:

```startLine:endLine:ingest.py
21:DOCS_DIR = "./docs"
22:CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
23:EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
27:COLLECTION_NAME = "docs_collection"
74:def ingest():
75:    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
76:    collection = client.get_or_create_collection(name=COLLECTION_NAME)
```

Run ingestion:

```bash
python ingest.py
```

This creates/updates a local Chroma database at `CHROMA_DIR` with your EdTech sales docs.

### 3.2. Retrieval + LLM (`rag.py`)

Core flow:

```startLine:endLine:rag.py
21:# Create persistent client for ChromaDB 1.3+
22:client = chromadb.PersistentClient(path=CHROMA_DIR)
23:collection = client.get_or_create_collection(name=COLLECTION_NAME)
24:embed_model = SentenceTransformer(EMBED_MODEL_NAME)
```

- `embed_text(text)` – encodes a query.
- `get_relevant_docs(query)` – nearest neighbor search in Chroma, returns docs + distances + similarity.
- `ask_with_rag(query, ...)`:
  1. Embed query and retrieve top‑`k` docs.
  2. Compute similarity; if the best similarity `< SIMILARITY_THRESHOLD`, it **falls back** to plain LLM.
  3. Otherwise, formats a specialized **EdTech sales** RAG prompt:

```startLine:endLine:rag.py
84:def ask_with_rag(query, top_k=RAG_TOP_K, threshold=SIMILARITY_THRESHOLD):
85:    docs = get_relevant_docs(query, top_k=top_k)
86:    if not docs:
87:        return call_openai_chat(query)
88:    best_sim = max(d["similarity"] for d in docs)
89:    if best_sim < threshold:
90:        return call_openai_chat(query)
91:    context = format_context(docs)
92:    prompt = (
93:        "You are an AI assistant specialized in **EdTech sales**.\n"
94:        "- You help with lead qualification, pitching EdTech products, handling objections, and closing deals.\n"
95:        "- Base your answers primarily on the EdTech sales documentation below. If something is not covered, say so, then answer briefly from your own knowledge.\n\n"
96:        f"CONTEXT (EdTech sales docs):\n{context}\n\n"
97:        f"SALES QUESTION: {query}\n\n"
98:        "Give a clear, actionable answer tailored to an EdTech sales scenario."
99:    )
100:    return call_openai_chat(prompt)
```

### 3.3. Fallback behavior

- Implemented in `ask_with_rag`:
  - **No docs or low similarity** → `call_openai_chat(query)` (plain LLM, no context).
  - **Relevant docs** → RAG prompt with `CONTEXT (EdTech sales docs): ...`.

Additionally, `call_openai_chat` has error handling for:
- Missing API key,
- Rate limit / insufficient quota,
- Other unexpected errors.

---

## 4. Agent SDK (VideoSDK) Integration

The **voice agent** is defined in `main_videosdk.py`:

```startLine:endLine:main_videosdk.py
24:class RagAgent(Agent):
25:    def __init__(self, instructions: str = None):
26:        instructions = instructions or (
27:            "You are an AI assistant specialized in **EdTech sales**.\n"
28:            "- Always think like an EdTech sales consultant.\n"
29:            "- Use the provided company/product documentation and playbooks first.\n"
30:            "- Focus on: lead qualification, demo pitching, objection handling, pricing discussion, and upsell/cross-sell for EdTech products.\n"
31:            "- If the answer is not in the docs, you may use your own knowledge, but keep it framed around EdTech sales."
32:        )
33:        super().__init__(instructions=instructions)
```

The pipeline uses:

```startLine:endLine:main_videosdk.py
52:    pipeline = CascadingPipeline(
53:        stt=DeepgramSTT(model="nova", language="en"),
54:        llm=OpenAILLM(model=os.getenv("OPENAI_MODEL", "gpt-4o")),
55:        tts=ElevenLabsTTS(voice="alloy"),
56:        vad=SileroVAD(threshold=0.35),
57:        turn_detector=TurnDetector(threshold=0.8)
58:    )
```

And the context is created with `JobContext` + `RoomOptions`:

```startLine:endLine:main_videosdk.py
69:def make_context():
70:    return JobContext(room_options=RoomOptions(room_id=os.getenv("VIDEOSDK_ROOM_ID", None), name="VideoSDK RAG Agent", playground=True))
```

> The VideoSDK Agent SDK expects `VIDEOSDK_AUTH_TOKEN` in the environment (which you set in `.env`) to authorize room access.

To run the voice agent:

```bash
python main_videosdk.py
```

This starts a long‑running job that:
- Connects to VideoSDK,
- Listens to your microphone,
- Runs `STT → RAG‑LLM → TTS`,
- Uses your EdTech sales docs as primary context.

---

## 5. Console Agent (Text) – Quick Test Flow

For a simple text‑only test (no VideoSDK), use `agent.py`:

```bash
python agent.py --mode type
```

You’ll see:

```text
Voice RAG Agent started. Mode: type
You:
```

### 5.1. Example queries (assignment “Test Flow”)

Assume you have `docs/sales_playbook.txt` describing how to pitch your LMS to schools.

- **Doc‑covered question (should use RAG context)**:
  - `You: How should I pitch our LMS to a school principal who worries about teacher adoption?`
  - **Expected behavior**:
    - Agent pulls relevant chunks from `sales_playbook.txt`.
    - Mentions concrete selling points from your docs (training, onboarding, analytics, etc.).

- **General question not covered in docs (should fall back to LLM)**:
  - `You: Who won the FIFA World Cup in 2018?`
  - **Expected behavior**:
    - Similarity with docs is low → **fallback** to plain LLM answer, ignoring doc context.

To exit:

```text
You: quit
```

---

## 6. GitHub Repository (for submission)

When you push this project to GitHub:

- Use a clear repo name, e.g. `videosdk-edtech-rag-agent`.
- Include this `README.md` at the root.
- Ensure `docs/` contains sample EdTech sales files (redact any secrets).
- Do **not** commit `.env` or any API keys.

You can then share the GitHub URL as required by the assignment.

---

## 7. Troubleshooting

- **OpenAI 429 / insufficient_quota**:
  - Check your OpenAI billing page.
  - Replace `OPENAI_API_KEY` in `.env` with a key that has quota.

- **VIDEOSDK_AUTH_TOKEN error**:
  - Make sure `.env` has `VIDEOSDK_AUTH_TOKEN=...`.
  - Restart the terminal after changing `.env`.

- **No answers from docs**:
  - Ensure you ran `python ingest.py` after editing files in `docs/`.
  - Lower `SIMILARITY_THRESHOLD` in `.env` (e.g., `0.5`) if matches are too strict.

This setup now fully matches the assignment requirements for **Agent SDK + RAG + fallback + documented test flow**. 
