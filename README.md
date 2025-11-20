# Voice RAG Agent – EdTech Sales (Assignment Ready)

This project implements an **end-to-end Voice AI Agent** using the **VideoSDK Agent SDK** with a **RAG (Retrieval-Augmented Generation)** pipeline on local **ChromaDB**, optimized for **EdTech sales**.  
It includes: STT → LLM → TTS pipeline, RAG, fallback, console agent, and full voice agent.

---

## 🚀 Features
- **Deepgram STT → OpenAI LLM → ElevenLabs TTS** voice pipeline  
- **Local RAG** using ChromaDB (docs ingestion)  
- **Automatic fallback** to plain LLM when docs are not relevant  
- **Console text agent** + **VideoSDK voice agent**  
- Fully meets assignment requirements: Agent SDK + RAG + Fallback + Test Flow

---

## 📂 Project Structure
ingest.py # Ingest docs/ into ChromaDB
rag.py # RAG logic: embeddings + retrieval + fallback
agent.py # Console-based text agent
main_videosdk.py # Voice agent using VideoSDK
docs/ # EdTech sales reference documents
requirements.txt
.env.example
README.md

yaml
Copy code

---

## 🛠️ Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
2. Create .env

Copy code

##OpenAI (Required)
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

# Speech & VideoSDK (Optional for full voice pipeline)
DEEPGRAM_API_KEY=
ELEVENLABS_API_KEY=
VIDEOSDK_AUTH_TOKEN=
VIDEOSDK_ROOM_ID=

# RAG / ChromaDB
CHROMA_DIR=./chroma_db
SIMILARITY_THRESHOLD=0.65
RAG_TOP_K=3
⚠️ Do NOT commit your .env file.

📘Add Docs for RAG
Place 2–4 small EdTech sales .txt or .md files inside docs/, such as:

product_overview.txt

sales_playbook.txt

pricing_and_offers.txt

Then ingest them:

bash
Copy code
python ingest.py
This builds the local chroma_db/ vector store.

🧠 RAG Pipeline (Short Overview)
Query → embed using all-MiniLM-L6-v2

Retrieve top-k docs from Chroma

If similarity ≥ threshold → build EdTech-sales RAG prompt

Else → fallback to plain LLM

Fallback ensures the agent always responds even without relevant docs.

🎤 Run the Agents
1. Console (Text) Agent
bash
Copy code
python agent.py --mode type
2. VideoSDK Voice Agent
bash
Copy code
python main_videosdk.py
The voice agent:
Microphone → Deepgram STT → RAG/LLM → ElevenLabs TTS → Audio Reply.

🧪 Test Queries
RAG Should Trigger
css
Copy code
How should I pitch our LMS to a school principal worried about teacher adoption?
Fallback Should Trigger
yaml
Copy code
Who won the 2018 FIFA World Cup?
To exit console agent:

nginx
Copy code
quit
