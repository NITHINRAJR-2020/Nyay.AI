"""
Vector store using FAISS + sentence-transformers.
Stores embeddings locally and supports semantic search.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# We use sentence-transformers (free, local) as primary embedder
# Falls back to OpenAI if OPENAI_API_KEY is set and ST not available

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


MODEL_NAME     = "all-MiniLM-L6-v2"   # 80MB, fast, good for legal text
EMBED_DIM      = 384
INDEX_FILE     = "faiss.index"
CHUNKS_FILE    = "chunks.pkl"
OPENAI_MODEL   = "text-embedding-3-small"


class VectorStore:
    def __init__(self, embed_dir: str):
        self.embed_dir   = Path(embed_dir)
        self.index_path  = self.embed_dir / INDEX_FILE
        self.chunks_path = self.embed_dir / CHUNKS_FILE

        self._model  = None     # lazy-loaded sentence-transformer
        self._index  = None     # FAISS index
        self._chunks: List[Dict[str, Any]] = []

        self._load()

    # ── lazy model load ────────────────────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            if _ST_AVAILABLE:
                print(f"[vector_store] Loading sentence-transformer: {MODEL_NAME}")
                self._model = SentenceTransformer(MODEL_NAME)
            else:
                raise RuntimeError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
        return self._model

    # ── embedding ──────────────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Return float32 numpy array of shape (N, EMBED_DIM)."""
        openai_key = os.getenv("OPENAI_API_KEY")

        if openai_key:
            return self._embed_openai(texts, openai_key)
        else:
            model = self._get_model()
            vectors = model.encode(texts, show_progress_bar=False, batch_size=32)
            return np.array(vectors, dtype=np.float32)

    def _embed_openai(self, texts: List[str], api_key: str) -> np.ndarray:
        import requests
        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": OPENAI_MODEL, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        vectors = [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]
        return np.array(vectors, dtype=np.float32)

    # ── FAISS helpers ──────────────────────────────────────────────────────────

    def _get_index(self):
        if self._index is None:
            if not _FAISS_AVAILABLE:
                raise RuntimeError("faiss-cpu not installed. Run: pip install faiss-cpu")
            self._index = faiss.IndexFlatIP(EMBED_DIM)   # Inner product (cosine after normalize)
        return self._index

    # ── persistence ────────────────────────────────────────────────────────────

    def _load(self):
        if self.chunks_path.exists():
            with open(self.chunks_path, "rb") as f:
                self._chunks = pickle.load(f)
            print(f"[vector_store] Loaded {len(self._chunks)} chunks from disk")

        if self.index_path.exists() and _FAISS_AVAILABLE:
            import faiss
            self._index = faiss.read_index(str(self.index_path))
            print(f"[vector_store] Loaded FAISS index ({self._index.ntotal} vectors)")

    def _save(self):
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)
        if self._index is not None and _FAISS_AVAILABLE:
            import faiss
            faiss.write_index(self._index, str(self.index_path))

    # ── public API ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Embed and index a list of chunk dicts."""
        if not chunks:
            return

        texts   = [c["text"] for c in chunks]
        vectors = self._embed(texts)

        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors = vectors / norms

        index = self._get_index()
        index.add(vectors)

        # Store vectors alongside chunks for retrieval
        for i, chunk in enumerate(chunks):
            chunk["_vector"] = vectors[i].tolist()
            self._chunks.append(chunk)

        self._save()
        print(f"[vector_store] Added {len(chunks)} chunks. Total: {len(self._chunks)}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        case_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search. Returns top_k chunks with similarity scores."""
        if not self._chunks:
            return []

        q_vec = self._embed([query])
        norm  = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        index = self._get_index()
        if index.ntotal == 0:
            return []

        # Search wider if filtering by case_id
        search_k = min(index.ntotal, top_k * 10 if case_id else top_k * 3)
        scores, indices = index.search(q_vec, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]

            if case_id and chunk.get("case_id") != case_id:
                continue

            results.append({
                "chunk_id":  chunk["id"],
                "case_id":   chunk["case_id"],
                "section":   chunk.get("section", ""),
                "text":      chunk["text"],
                "score":     float(score),
                "metadata":  chunk.get("metadata", {}),
            })

            if len(results) >= top_k:
                break

        return results

    def get_case_chunks(self, case_id: str) -> List[Dict[str, Any]]:
        """Return all chunks belonging to a case, ordered by index."""
        chunks = [c for c in self._chunks if c.get("case_id") == case_id]
        chunks.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))
        return chunks

    def get_case_embedding(self, case_id: str) -> Optional[np.ndarray]:
        """Return mean embedding for a case (used for similarity)."""
        chunks = [c for c in self._chunks if c.get("case_id") == case_id]
        if not chunks:
            return None
        vectors = [c["_vector"] for c in chunks if "_vector" in c]
        if not vectors:
            return None
        arr = np.array(vectors, dtype=np.float32)
        mean_vec = arr.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec /= norm
        return mean_vec

    def remove_case(self, case_id: str):
        """Remove all chunks for a case. Rebuilds FAISS index."""
        self._chunks = [c for c in self._chunks if c.get("case_id") != case_id]

        # Rebuild index
        if _FAISS_AVAILABLE:
            import faiss
            self._index = faiss.IndexFlatIP(EMBED_DIM)
            if self._chunks:
                vectors = np.array(
                    [c["_vector"] for c in self._chunks if "_vector" in c],
                    dtype=np.float32,
                )
                self._index.add(vectors)

        self._save()
