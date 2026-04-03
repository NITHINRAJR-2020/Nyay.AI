"""
Vector store using FAISS + sentence-transformers.
Stores embeddings locally and supports semantic search.

Upgrades:
  - Hybrid search: BM25 (0.3) + embedding cosine (0.7)
  - Optional cross-encoder reranking
  - Embedding cache: skips re-embedding duplicate case_ids
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

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

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    _CROSSENCODER_AVAILABLE = True
except ImportError:
    _CROSSENCODER_AVAILABLE = False


MODEL_NAME       = "all-MiniLM-L6-v2"
CROSSENC_NAME    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_DIM        = 384
INDEX_FILE       = "faiss.index"
CHUNKS_FILE      = "chunks.pkl"
OPENAI_MODEL     = "text-embedding-3-small"

# Hybrid search weights
EMBED_WEIGHT     = 0.7
BM25_WEIGHT      = 0.3


class VectorStore:
    def __init__(self, embed_dir: str):
        self.embed_dir   = Path(embed_dir)
        self.index_path  = self.embed_dir / INDEX_FILE
        self.chunks_path = self.embed_dir / CHUNKS_FILE

        self._model        = None   # lazy sentence-transformer
        self._cross_encoder = None  # lazy cross-encoder
        self._index        = None   # FAISS index
        self._chunks: List[Dict[str, Any]] = []

        # BM25 state — rebuilt from _chunks on load
        self._bm25: Optional["BM25Okapi"] = None
        self._bm25_corpus: List[List[str]] = []   # tokenized chunk texts

        # ── Embedding cache: case_id → True ──────────────────────────────────
        # Prevents re-embedding the same document if uploaded twice.
        self._indexed_case_ids: set = set()

        self._load()

    # ── lazy model loading ─────────────────────────────────────────────────────

    def _get_model(self) -> "SentenceTransformer":
        if self._model is None:
            if not _ST_AVAILABLE:
                raise RuntimeError(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
            print(f"[vector_store] Loading sentence-transformer: {MODEL_NAME}")
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    def _get_cross_encoder(self) -> "CrossEncoder":
        if self._cross_encoder is None:
            if not _CROSSENCODER_AVAILABLE:
                raise RuntimeError(
                    "sentence-transformers not installed (CrossEncoder). "
                    "Run: pip install sentence-transformers"
                )
            print(f"[vector_store] Loading cross-encoder: {CROSSENC_NAME}")
            self._cross_encoder = CrossEncoder(CROSSENC_NAME)
        return self._cross_encoder

    # ── embedding ──────────────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Return float32 array of shape (N, EMBED_DIM)."""
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            return self._embed_openai(texts, openai_key)

        model = self._get_model()
        vectors = model.encode(
            texts,
            show_progress_bar=False,
            batch_size=32,         # efficient batching
            normalize_embeddings=True,  # pre-normalize → skip manual norm step
        )
        return np.array(vectors, dtype=np.float32)

    def _embed_openai(self, texts: List[str], api_key: str) -> np.ndarray:
        import requests
        # Send in a single batched request (OpenAI supports up to 2048 inputs)
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

    # ── FAISS ──────────────────────────────────────────────────────────────────

    def _get_index(self):
        if self._index is None:
            if not _FAISS_AVAILABLE:
                raise RuntimeError("faiss-cpu not installed. Run: pip install faiss-cpu")
            self._index = faiss.IndexFlatIP(EMBED_DIM)
        return self._index

    # ── BM25 ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        return text.lower().split()

    def _rebuild_bm25(self):
        """Rebuild BM25 index from current _chunks."""
        if not _BM25_AVAILABLE or not self._chunks:
            return
        self._bm25_corpus = [self._tokenize(c["text"]) for c in self._chunks]
        self._bm25 = BM25Okapi(self._bm25_corpus)

    def _bm25_scores(self, query: str) -> np.ndarray:
        """Return BM25 score array aligned with self._chunks. All zeros if unavailable."""
        if not _BM25_AVAILABLE or self._bm25 is None or not self._chunks:
            return np.zeros(len(self._chunks), dtype=np.float32)
        raw = np.array(self._bm25.get_scores(self._tokenize(query)), dtype=np.float32)
        # Min-max normalise into [0, 1]
        rng = raw.max() - raw.min()
        if rng > 0:
            raw = (raw - raw.min()) / rng
        return raw

    # ── persistence ────────────────────────────────────────────────────────────

    def _load(self):
        if self.chunks_path.exists():
            with open(self.chunks_path, "rb") as f:
                self._chunks = pickle.load(f)
            self._indexed_case_ids = {c["case_id"] for c in self._chunks}
            self._rebuild_bm25()
            print(f"[vector_store] Loaded {len(self._chunks)} chunks from disk")

        if self.index_path.exists() and _FAISS_AVAILABLE:
            import faiss as _faiss
            self._index = _faiss.read_index(str(self.index_path))
            print(f"[vector_store] Loaded FAISS index ({self._index.ntotal} vectors)")

    def _save(self):
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)
        if self._index is not None and _FAISS_AVAILABLE:
            import faiss as _faiss
            _faiss.write_index(self._index, str(self.index_path))

    # ── public API ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Embed and index a list of chunk dicts.
        Skips entirely if the case_id is already indexed (embedding cache).
        """
        if not chunks:
            return

        case_id = chunks[0].get("case_id")

        # ── Embedding cache check ─────────────────────────────────────────────
        if case_id and case_id in self._indexed_case_ids:
            print(f"[vector_store] case_id '{case_id}' already indexed — skipping re-embed.")
            return

        texts   = [c["text"] for c in chunks]
        vectors = self._embed(texts)

        # Normalize for cosine similarity via inner product (skip if ST already normalized)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors = vectors / norms

        index = self._get_index()
        index.add(vectors)

        for i, chunk in enumerate(chunks):
            chunk["_vector"] = vectors[i].tolist()
            self._chunks.append(chunk)

        if case_id:
            self._indexed_case_ids.add(case_id)

        # Rebuild BM25 with new chunks included
        self._rebuild_bm25()
        self._save()
        print(f"[vector_store] Added {len(chunks)} chunks. Total: {len(self._chunks)}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        case_id: Optional[str] = None,
        rerank: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid semantic + BM25 search with optional cross-encoder reranking.

        Scoring:  final = 0.7 * embed_score + 0.3 * bm25_score
        Reranking: top 10 results re-scored by cross-encoder, best 5 returned.

        Args:
            query:   Natural language query.
            top_k:   Number of results to return.
            case_id: If set, restrict to chunks of this case.
            rerank:  If True, apply cross-encoder reranking on top candidates.
        """
        if not self._chunks:
            return []

        # ── 1. Embedding scores via FAISS ──────────────────────────────────────
        q_vec = self._embed([query])
        norm  = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        index = self._get_index()
        if index.ntotal == 0:
            return []

        # Retrieve a wide candidate pool; we'll rerank / filter down
        pool_k  = min(index.ntotal, max(top_k * 20, 50))
        scores_embed, indices = index.search(q_vec, pool_k)
        embed_scores = scores_embed[0]          # shape (pool_k,)
        faiss_indices = indices[0]              # shape (pool_k,)

        # Build a full-corpus embedding score array (default 0)
        full_embed = np.zeros(len(self._chunks), dtype=np.float32)
        for score, idx in zip(embed_scores, faiss_indices):
            if 0 <= idx < len(self._chunks):
                full_embed[idx] = float(score)

        # ── 2. BM25 scores ────────────────────────────────────────────────────
        full_bm25 = self._bm25_scores(query)    # already normalised [0,1]

        # Normalise embed scores to [0,1] as well
        if full_embed.max() > 0:
            full_embed = (full_embed - full_embed.min()) / (full_embed.max() - full_embed.min())

        # ── 3. Hybrid combination ─────────────────────────────────────────────
        combined = EMBED_WEIGHT * full_embed + BM25_WEIGHT * full_bm25

        # ── 4. Apply case_id filter and collect candidates ────────────────────
        candidates = []
        for idx in np.argsort(combined)[::-1]:
            if idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            if case_id and chunk.get("case_id") != case_id:
                continue
            candidates.append((combined[idx], idx, chunk))
            # Collect enough for reranking (top 10) or direct return
            if len(candidates) >= (10 if rerank else top_k):
                break

        if not candidates:
            return []

        # ── 5. Optional cross-encoder reranking ───────────────────────────────
        if rerank and _CROSSENCODER_AVAILABLE and len(candidates) > 1:
            try:
                cross = self._get_cross_encoder()
                pairs = [(query, c["text"]) for _, _, c in candidates]
                ce_scores = cross.predict(pairs)
                # Re-sort by cross-encoder score
                reranked = sorted(
                    zip(ce_scores, candidates),
                    key=lambda x: x[0],
                    reverse=True,
                )
                candidates = [cand for _, cand in reranked]
            except Exception as e:
                print(f"[vector_store] Cross-encoder reranking failed ({e}), skipping.")

        # ── 6. Build final result list ─────────────────────────────────────────
        results = []
        for item in candidates[:top_k]:
            score, idx, chunk = item if len(item) == 3 else (*item,)  # unpack safely
            results.append({
                "chunk_id": chunk["id"],
                "case_id":  chunk["case_id"],
                "section":  chunk.get("section", ""),
                "text":     chunk["text"],
                "score":    float(combined[idx]),
                "metadata": chunk.get("metadata", {}),
            })

        return results

    def get_case_chunks(self, case_id: str) -> List[Dict[str, Any]]:
        """Return all chunks belonging to a case, ordered by chunk index."""
        chunks = [c for c in self._chunks if c.get("case_id") == case_id]
        chunks.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))
        return chunks

    def get_case_embedding(self, case_id: str) -> Optional[np.ndarray]:
        """Return mean embedding for a case (used for similarity comparison)."""
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
        """Remove all chunks for a case and rebuild FAISS + BM25 indices."""
        self._chunks = [c for c in self._chunks if c.get("case_id") != case_id]
        self._indexed_case_ids.discard(case_id)

        # Rebuild FAISS
        if _FAISS_AVAILABLE:
            import faiss as _faiss
            self._index = _faiss.IndexFlatIP(EMBED_DIM)
            if self._chunks:
                vectors = np.array(
                    [c["_vector"] for c in self._chunks if "_vector" in c],
                    dtype=np.float32,
                )
                self._index.add(vectors)

        # Rebuild BM25
        self._rebuild_bm25()
        self._save()