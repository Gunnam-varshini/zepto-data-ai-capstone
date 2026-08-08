"""
ingest.py - Loads the Zepto policy corpus, embeds it locally with
sentence-transformers (all-MiniLM-L6-v2), and stores/retrieves it from a
persistent ChromaDB collection. No API key or network call is required.
"""

from pathlib import Path
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).parent / "docs"
PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "zepto_policies"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # characters; corpus docs are short, so this yields one chunk/doc in practice

_model = None
_client = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_model().encode(texts, convert_to_numpy=True).tolist()


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_collection():
    client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def ensure_ingested(force: bool = False) -> int:
    """Embeds and stores all corpus documents in ChromaDB if not already present.
    Returns the number of chunks currently in the collection."""
    collection = get_collection()

    if force:
        client = get_client()
        client.delete_collection(COLLECTION_NAME)
        collection = get_collection()

    if collection.count() > 0:
        return collection.count()

    doc_paths = sorted(DOCS_DIR.glob("doc_*.txt"))
    if not doc_paths:
        raise FileNotFoundError(f"No corpus documents found in {DOCS_DIR}")

    ids, texts, metadatas = [], [], []
    for path in doc_paths:
        content = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(_chunk_text(content)):
            ids.append(f"{path.stem}_chunk{i}")
            texts.append(chunk)
            metadatas.append({"source": path.stem})

    embeddings = embed_texts(texts)
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return collection.count()


if __name__ == "__main__":
    n = ensure_ingested(force=True)
    print(f"Ingested {n} chunks into ChromaDB collection '{COLLECTION_NAME}' at {PERSIST_DIR}")
