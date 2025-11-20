#!/usr/bin/env python3
"""
Memory-safe ingest.py

- Streams text files from ./docs
- Chunks each file incrementally
- Encodes chunks in batches (not all at once)
- Upserts batches to ChromaDB
- Skips non-text/binary files and large PDFs (recommend converting PDFs to text first)
"""
import os
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DOCS_DIR = "./docs"
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
BATCH_EMBED = int(os.getenv("EMBED_BATCH_SIZE", 32))
COLLECTION_NAME = "docs_collection"
MAX_FILE_MB = float(os.getenv("MAX_FILE_MB", 100))  # warn/skip files larger than this (adjust)

def is_probably_text(path, sample_bytes=4096):
    """
    Heuristic: read a small prefix and check for NUL bytes and decoding ability.
    """
    try:
        with open(path, "rb") as fh:
            prefix = fh.read(sample_bytes)
        # NUL bytes are a good sign it's binary
        if b"\x00" in prefix:
            return False
        # Try decode as utf-8 with replacement; check ratio of non-printables
        text = prefix.decode("utf-8", errors="replace")
        non_print = sum(1 for ch in text if ord(ch) < 9 or (32 <= ord(ch) <= 126) == False)
        # if too many non-printable characters, it's likely binary
        if non_print / max(1, len(text)) > 0.30:
            return False
        return True
    except Exception:
        return False

def chunk_generator_from_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    start = 0
    L = len(text)
    while start < L:
        end = min(start + size, L)
        yield text[start:end]
        start = end - overlap
        if start < 0:
            start = 0
        if start >= L:
            break

def iter_file_chunks(path):
    """
    Read file in a streaming way and produce chunks.
    This tries to handle large files by reading them fully but not storing all chunks.
    If files are huge (multi-GB), you should pre-process into smaller files.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    # If file is too big, we still process but warn
    for chunk in chunk_generator_from_text(text):
        yield chunk

def ingest():
    # Use a persistent Chroma client so embeddings are stored on disk
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    model = SentenceTransformer(EMBED_MODEL_NAME)

    # Walk docs and build a generator of (id, text, metadata)
    def gen_all_chunks():
        for fname in sorted(os.listdir(DOCS_DIR)):
            path = os.path.join(DOCS_DIR, fname)
            if not os.path.isfile(path):
                continue
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                print(f"WARNING: Skipping {fname} as it is {size_mb:.1f} MB (> MAX_FILE_MB={MAX_FILE_MB}). Consider splitting or converting.")
                continue
            if not is_probably_text(path):
                print(f"Skipping non-text or binary file: {fname}")
                continue
            # stream chunks for this file
            chunk_index = 0
            for chunk in iter_file_chunks(path):
                if not chunk.strip():
                    continue
                doc_id = f"{fname}__chunk_{chunk_index}"
                meta = {"source": fname, "chunk_index": chunk_index}
                yield doc_id, chunk, meta
                chunk_index += 1

    # Batch and upsert to Chroma
    batch_ids, batch_docs, batch_metas = [], [], []
    total = 0
    gen = gen_all_chunks()
    for doc_id, doc_text, meta in tqdm(gen, desc="Preparing chunks"):
        batch_ids.append(doc_id)
        batch_docs.append(doc_text)
        batch_metas.append(meta)
        if len(batch_ids) >= BATCH_EMBED:
            embs = model.encode(batch_docs, show_progress_bar=False).tolist()
            collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=embs)
            total += len(batch_ids)
            batch_ids, batch_docs, batch_metas = [], [], []
    # leftover
    if batch_ids:
        embs = model.encode(batch_docs, show_progress_bar=False).tolist()
        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=embs)
        total += len(batch_ids)

    print(f"Ingest complete. Indexed {total} chunks into Chroma at {CHROMA_DIR}")

if __name__ == "__main__":
    if not os.path.isdir(DOCS_DIR):
        print(f"Docs dir '{DOCS_DIR}' not found. Create it and add small text files (.txt/.md).")
    else:
        ingest()
