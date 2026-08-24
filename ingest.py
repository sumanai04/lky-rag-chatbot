"""Chunk prepared source texts, embed them, and store them in a Chroma vector DB.

Run after prepare.py:
    python ingest.py
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import chromadb  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

CLEAN_DIR = Path("data/clean")
CHROMA_DIR = Path("chroma_db")
COLLECTION = "lky"


def load_sources():
    with open("data/sources.json", encoding="utf-8") as f:
        return json.load(f)


def chunk_text(text, size, overlap):
    """Pack sentences into chunks of roughly `size` chars.

    Each new chunk re-includes the tail of the previous chunk (up to `overlap`
    chars) so quote boundaries are not lost between chunks.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks = []
    current = ""
    for sentence in sentences:
        if len(sentence) > size:
            sentence = sentence[:size]
        if current and len(current) + len(sentence) + 1 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap and len(current) > overlap else ""
            current = (tail + " " + sentence).strip() if tail else sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence
    if current:
        chunks.append(current)
    return chunks


def main():
    sources = load_sources()
    size = int(os.getenv("CHUNK_SIZE", "600"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    print(f"Loading embedding model {model_name} ...")
    embedder = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    existing = set(collection.get(include=[])["ids"])
    print(f"Collection '{COLLECTION}' has {len(existing)} chunks already.")

    for src in sources:
        path = CLEAN_DIR / src["file"]
        if not path.exists():
            print(f"  skip {src['file']} (missing)")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, size, overlap)
        print(f"  {src['id']}: {len(chunks)} chunks")

        batch = 64
        meta = {"title": src["title"], "date": src["date"], "source": src["source"], "url": src["url"]}
        for i in range(0, len(chunks), batch):
            ids, docs, metas = [], [], []
            for j, chunk in enumerate(chunks[i:i + batch]):
                chunk_id = f"{src['id']}:{i + j}"
                if chunk_id in existing:
                    continue
                ids.append(chunk_id)
                docs.append(chunk)
                metas.append(meta)
            if not ids:
                continue
            embeddings = embedder.encode(docs).tolist()
            collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
            print(f"    embedded {i + len(ids)}/{len(chunks)}")

    total = collection.count()
    print(f"Done. Collection now holds {total} chunks.")


if __name__ == "__main__":
    main()
