#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import openai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "docs_collection"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.65))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 3))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Initialize OpenAI client (v1.0+ API)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Create persistent client for ChromaDB 1.3+
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def embed_text(text):
    return embed_model.encode([text], show_progress_bar=False)[0].tolist()

def get_relevant_docs(query, top_k=RAG_TOP_K):
    q_emb = embed_text(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents","metadatas","distances"]
    )
    docs=[]
    docs_list = results.get("documents", [[]])[0]
    metas_list = results.get("metadatas", [[]])[0]
    dists_list = results.get("distances", [[]])[0]
    for d,m,dist in zip(docs_list, metas_list, dists_list):
        sim = 1.0/(1.0+float(dist))
        docs.append({"doc":d,"meta":m,"distance":dist,"similarity":sim})
    return docs

def format_context(docs):
    parts=[]
    for d in docs:
        src = d.get("meta", {}).get("source", "unknown")
        idx = d.get("meta", {}).get("chunk_index", -1)
        parts.append(f"Source: {src} (chunk {idx})\n{d['doc']}\n---")
    return "\n".join(parts)

def call_openai_chat(prompt, model=OPENAI_MODEL, max_tokens=512, temperature=0.2):
    if not openai_client:
        return "OpenAI API key not configured; cannot answer questions. Please set OPENAI_API_KEY in your .env file."

    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    except openai.RateLimitError:
        # This is the error you're currently hitting: insufficient_quota (HTTP 429)
        return (
            "OpenAI is rejecting requests because this API key has no remaining quota.\n"
            "Please check your OpenAI billing/plan or use a different API key, then run the project again."
        )
    except openai.AuthenticationError:
        return (
            "OpenAI authentication failed. Please double‑check your OPENAI_API_KEY "
            "value in the .env file and try again."
        )
    except Exception as e:
        # Catch-all so that the app shows a message instead of crashing
        return f"OpenAI request failed: {e}"

def ask_with_rag(query, top_k=RAG_TOP_K, threshold=SIMILARITY_THRESHOLD):
    docs = get_relevant_docs(query, top_k=top_k)
    if not docs:
        return call_openai_chat(query)
    best_sim = max(d["similarity"] for d in docs)
    if best_sim < threshold:
        return call_openai_chat(query)
    context = format_context(docs)
    prompt = (
        "You are an AI assistant specialized in **EdTech sales**.\n"
        "- You help with lead qualification, pitching EdTech products, handling objections, and closing deals.\n"
        "- Base your answers primarily on the EdTech sales documentation below. If something is not covered, say so, then answer briefly from your own knowledge.\n\n"
        f"CONTEXT (EdTech sales docs):\n{context}\n\n"
        f"SALES QUESTION: {query}\n\n"
        "Give a clear, actionable answer tailored to an EdTech sales scenario."
    )
    return call_openai_chat(prompt)
