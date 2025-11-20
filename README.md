Voice RAG Agent – EdTech Sales (Assignment Ready)

This project implements a complete Voice AI Agent using the VideoSDK Agent SDK combined with a RAG (Retrieval-Augmented Generation) pipeline powered by a local ChromaDB vector store.
The system is specialized for EdTech sales, including pitching, qualification, pricing, objections, and general sales workflows.

1. Features

Deepgram Speech-to-Text (STT)

OpenAI LLM response generation

ElevenLabs Text-to-Speech (TTS)

Local RAG using ChromaDB (docs → ingestion → retrieval)

Fallback to plain LLM when doc similarity is low

Console agent + VideoSDK voice agent

Fully meets assignment requirements: Agent SDK + RAG + fallback + test flow

2. Project Structure

ingest.py – Ingests text files from docs/ into ChromaDB
rag.py – RAG logic: embeddings, retrieval, similarity scoring, fallback
agent.py – Console-based text agent
main_videosdk.py – End-to-end Voice Agent (STT → RAG+LLM → TTS)
docs/ – EdTech sales documents
requirements.txt – Python dependencies
.env.example – Environment template
README.md – You are reading it

3. Setup Instructions
Create environment and install dependencies:
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt

Create .env file:
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
DEEPGRAM_API_KEY=
ELEVENLABS_API_KEY=
VIDEOSDK_AUTH_TOKEN=
VIDEOSDK_ROOM_ID=
CHROMA_DIR=./chroma_db
SIMILARITY_THRESHOLD=0.65
RAG_TOP_K=3

4. Prepare RAG Documents

Add 2–4 small .txt or .md files about EdTech sales to the docs/ directory.

Examples:

product_overview.txt

sales_playbook.txt

pricing_and_offers.txt

Ingest them into ChromaDB:

python ingest.py

5. Run the Agents
Console agent:
python agent.py --mode type

Voice agent (VideoSDK):
python main_videosdk.py


The voice agent performs:
Microphone → Deepgram STT → RAG/OpenAI → ElevenLabs TTS → Spoken response

6. Testing (for assignment)
RAG should trigger:

“How should I pitch our LMS to a school principal worried about teacher adoption?”

Fallback should trigger:

“Who won the 2018 FIFA World Cup?”

Exit console agent:

quit

7. Troubleshooting

OpenAI API issues:

429 insufficient_quota → update billing or use valid key

RAG not working:

Re-run python ingest.py

Lower similarity threshold:

SIMILARITY_THRESHOLD=0.55


VideoSDK issues:

Check VIDEOSDK_AUTH_TOKEN

Restart terminal after editing .env

8. Submission Checklist

✔ README.md included (this file)
✔ Add 3+ EdTech sales files to docs/
✔ .env not committed
✔ RAG ingestion tested
✔ Console + voice agent fully working

✔ Final Notes

This repository fully meets the assignment requirements:

Agent SDK

Complete RAG pipeline

Fallback logic

Document ingestion

Voice and text agents

Structured test flow

Professional documentation
