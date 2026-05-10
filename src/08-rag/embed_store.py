"""
Embedding generation and vector store ingestion for RAG.
Covers: TF-IDF-style embeddings from scratch, ChromaDB ingestion,
        collection management, batch upsert, similarity search, metadata filtering.
pip install numpy chromadb
"""

import os
import re
import math
import tempfile
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Chunk dataclass (mirrors ingest.py — standalone here)
# ---------------------------------------------------------------------------

class Chunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata


# ---------------------------------------------------------------------------
# From-scratch TF-IDF embedding
# ---------------------------------------------------------------------------

class TFIDFEmbedder:
    """
    Sparse → dense projection via TF-IDF + random projection.
    Produces unit-normalised dense vectors of configurable dimension.
    Deterministic given a fixed corpus and seed.
    """

    def __init__(self, dim: int = 128, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.idf: Dict[str, float] = {}
        self.vocab: Dict[str, int] = {}
        self.projection: Optional[np.ndarray] = None
        self._fitted = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z]+", text.lower())

    def fit(self, texts: List[str]) -> "TFIDFEmbedder":
        N = len(texts)
        df: Counter = Counter()
        for text in texts:
            unique_terms = set(self._tokenize(text))
            df.update(unique_terms)

        vocab_terms = sorted(df.keys())
        self.vocab = {t: i for i, t in enumerate(vocab_terms)}
        V = len(self.vocab)

        # IDF with Laplace smoothing: log((N + 1) / (df(t) + 1)) + 1
        self.idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in self.vocab}

        # Random projection from vocab space → dim
        self.projection = self.rng.standard_normal((V, self.dim)) / math.sqrt(self.dim)
        self._fitted = True
        return self

    def _tfidf_vec(self, text: str) -> np.ndarray:
        """Compute TF-IDF vector in vocab space."""
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(len(self.vocab))
        tf: Counter = Counter(tokens)
        max_tf = max(tf.values())
        vec = np.zeros(len(self.vocab))
        for term, count in tf.items():
            if term in self.vocab:
                tf_norm = count / max_tf           # raw TF normalised by max
                idf_val = self.idf.get(term, 1.0)
                vec[self.vocab[term]] = tf_norm * idf_val
        return vec

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Returns (n, dim) array of unit-normalised embeddings.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")
        vecs = []
        for text in texts:
            tfidf = self._tfidf_vec(text)
            dense = tfidf @ self.projection   # (dim,)
            norm = np.linalg.norm(dense)
            if norm > 1e-9:
                dense /= norm
            vecs.append(dense)
        return np.array(vecs)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# In-memory vector store (scratch — no ChromaDB dependency)
# ---------------------------------------------------------------------------

class InMemoryVectorStore:
    """
    Simple HNSW-free brute-force vector store.
    Stores embeddings, documents, metadata. Queries via cosine similarity.
    """

    def __init__(self):
        self._ids: List[str] = []
        self._embeddings: List[np.ndarray] = []
        self._documents: List[str] = []
        self._metadatas: List[Dict] = []

    def upsert(self, ids: List[str], embeddings: np.ndarray,
               documents: List[str], metadatas: List[Dict]) -> None:
        for i, doc_id in enumerate(ids):
            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                self._embeddings[idx] = embeddings[i]
                self._documents[idx] = documents[i]
                self._metadatas[idx] = metadatas[i]
            else:
                self._ids.append(doc_id)
                self._embeddings.append(embeddings[i])
                self._documents.append(documents[i])
                self._metadatas.append(metadatas[i])

    def query(self, query_embedding: np.ndarray, n_results: int = 5,
              where: Optional[Dict] = None) -> Dict:
        if not self._ids:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        E = np.stack(self._embeddings)    # (N, dim)
        scores = E @ query_embedding      # cosine (unit vecs): (N,)
        distances = 1.0 - scores          # convert to distance

        # Metadata filter
        mask = np.ones(len(self._ids), dtype=bool)
        if where:
            for i, meta in enumerate(self._metadatas):
                for k, v in where.items():
                    if meta.get(k) != v:
                        mask[i] = False

        filtered_indices = np.where(mask)[0]
        if len(filtered_indices) == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        filtered_distances = distances[filtered_indices]
        top_k_local = np.argsort(filtered_distances)[:n_results]
        top_k = filtered_indices[top_k_local]

        return {
            "ids":       [[self._ids[i] for i in top_k]],
            "documents": [[self._documents[i] for i in top_k]],
            "metadatas": [[self._metadatas[i] for i in top_k]],
            "distances": [[float(distances[i]) for i in top_k]],
        }

    def get(self, ids: List[str]) -> Dict:
        result = {"ids": [], "documents": [], "metadatas": []}
        for doc_id in ids:
            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                result["ids"].append(doc_id)
                result["documents"].append(self._documents[idx])
                result["metadatas"].append(self._metadatas[idx])
        return result

    def delete(self, ids: List[str]) -> None:
        for doc_id in ids:
            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                self._ids.pop(idx)
                self._embeddings.pop(idx)
                self._documents.pop(idx)
                self._metadatas.pop(idx)

    def count(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# ChromaDB wrapper (graceful if not installed)
# ---------------------------------------------------------------------------

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("chromadb not installed — ChromaDB demos will be skipped")
    print("pip install chromadb")


class ChromaStore:
    """
    Thin wrapper around ChromaDB for RAG ingestion and retrieval.
    Uses in-memory client for zero-setup demo.
    """

    def __init__(self, collection_name: str = "rag_demo", dim: int = 128):
        if not CHROMA_AVAILABLE:
            raise RuntimeError("pip install chromadb")
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: List[Chunk], embedder: TFIDFEmbedder,
                      batch_size: int = 64) -> int:
        texts = [c.text for c in chunks]
        embeddings = embedder.embed(texts)
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embs = embeddings[i:i + batch_size]
            self.collection.upsert(
                ids=[f"chunk_{i+j}" for j in range(len(batch_chunks))],
                embeddings=batch_embs.tolist(),
                documents=[c.text for c in batch_chunks],
                metadatas=[c.metadata for c in batch_chunks],
            )
            total += len(batch_chunks)
        return total

    def query(self, query_embedding: np.ndarray, n_results: int = 5,
              where: Optional[Dict] = None) -> Dict:
        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": min(n_results, self.collection.count()),
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def count(self) -> int:
        return self.collection.count()


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

def build_corpus() -> List[Chunk]:
    """Build a small demo corpus of chunks with source metadata."""
    texts = [
        ("The attention mechanism computes query-key-value weighted sums. "
         "Scaled dot-product attention divides by sqrt(d_k) to stabilise gradients.",
         "transformers.txt", 1),
        ("Multi-head attention projects queries, keys, and values h times "
         "to different subspaces and concatenates the results.",
         "transformers.txt", 2),
        ("BM25 is a probabilistic retrieval function based on term frequency "
         "and inverse document frequency with document length normalisation.",
         "retrieval.txt", 1),
        ("Dense retrieval uses neural embeddings to encode semantic meaning. "
         "Cosine similarity finds semantically similar documents regardless of exact wording.",
         "retrieval.txt", 2),
        ("Hybrid search combines BM25 sparse scores with dense embedding scores. "
         "Reciprocal Rank Fusion merges ranked lists without score normalisation.",
         "retrieval.txt", 3),
        ("RAG injects retrieved context into the LLM prompt before generation. "
         "The model answers from evidence rather than memorised parametric knowledge.",
         "rag_overview.txt", 1),
        ("Faithfulness measures whether the generated answer is grounded in the context. "
         "Answer relevancy checks if the answer addresses the question.",
         "evaluation.txt", 1),
        ("LoRA fine-tunes only low-rank adapter matrices, freezing the base model weights. "
         "This reduces trainable parameters by orders of magnitude.",
         "finetuning.txt", 1),
        ("Chunking splits documents into segments for embedding. Fixed-size chunks use "
         "a sliding window. Recursive splitting respects paragraph and sentence boundaries.",
         "chunking.txt", 1),
        ("FAISS provides efficient approximate nearest neighbour search. "
         "HNSW builds a hierarchical graph for sub-linear query time.",
         "vectordb.txt", 1),
    ]
    chunks = []
    for i, (text, source, page) in enumerate(texts):
        chunks.append(Chunk(
            text=text,
            metadata={"source": source, "page": page, "chunk_index": i}
        ))
    return chunks


def main():
    corpus = build_corpus()

    section("TF-IDF EMBEDDER — FIT ON CORPUS")
    embedder = TFIDFEmbedder(dim=64, seed=42)
    embedder.fit([c.text for c in corpus])
    print(f"\n  Vocab size   : {len(embedder.vocab)}")
    print(f"  Embedding dim: {embedder.dim}")
    print(f"  Projection   : {embedder.projection.shape}")

    # Sample embeddings
    embs = embedder.embed([c.text for c in corpus])
    print(f"\n  Embeddings shape: {embs.shape}")
    print(f"  L2 norms (should be ≈1.0): ", end="")
    norms = np.linalg.norm(embs, axis=1)
    print("  ".join(f"{n:.4f}" for n in norms[:5]))

    section("COSINE SIMILARITY MATRIX (TOP 5 CHUNKS)")
    sim_matrix = embs[:5] @ embs[:5].T
    print(f"\n  (rows/cols = chunk 0..4)")
    header = "       " + "  ".join(f"  [{i}]" for i in range(5))
    print(f"\n  {header}")
    for i in range(5):
        row = "  ".join(f"{sim_matrix[i,j]:+.3f}" for j in range(5))
        print(f"  [{i}]  {row}")
    print(f"\n  Diagonal = 1.0 (self-similarity). Off-diag shows semantic overlap.")

    section("IN-MEMORY VECTOR STORE — UPSERT + QUERY")
    store = InMemoryVectorStore()
    ids = [f"chunk_{i}" for i in range(len(corpus))]
    store.upsert(ids, embs, [c.text for c in corpus], [c.metadata for c in corpus])
    print(f"\n  Upserted {store.count()} chunks")

    # Query
    query = "How does attention work in transformers?"
    q_emb = embedder.embed_one(query)
    results = store.query(q_emb, n_results=3)
    print(f"\n  Query: '{query}'")
    print(f"  Top-3 results:")
    for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0])):
        print(f"\n    [{i+1}] dist={dist:.4f}  source={meta['source']}  page={meta['page']}")
        print(f"         {doc[:100]}...")

    # Metadata filter
    print(f"\n  Filtered query (source='retrieval.txt'):")
    results_filtered = store.query(q_emb, n_results=5,
                                   where={"source": "retrieval.txt"})
    for doc, meta in zip(results_filtered["documents"][0],
                         results_filtered["metadatas"][0]):
        print(f"    source={meta['source']} | {doc[:80]}...")

    section("BATCH UPSERT — PERFORMANCE")
    import time
    large_corpus = corpus * 20   # 200 chunks
    large_embs = embedder.embed([c.text for c in large_corpus])
    large_ids = [f"large_{i}" for i in range(len(large_corpus))]
    large_store = InMemoryVectorStore()

    t0 = time.perf_counter()
    large_store.upsert(large_ids, large_embs, [c.text for c in large_corpus],
                       [c.metadata for c in large_corpus])
    t_upsert = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for _ in range(100):
        large_store.query(q_emb, n_results=5)
    t_query = (time.perf_counter() - t0) / 100 * 1000

    print(f"\n  Corpus size  : {large_store.count()} chunks")
    print(f"  Upsert time  : {t_upsert:.1f} ms")
    print(f"  Query time   : {t_query:.3f} ms/query (brute-force cosine)")

    section("CHROMADB INTEGRATION")
    if not CHROMA_AVAILABLE:
        print("\n  Skipping — install chromadb: pip install chromadb")
    else:
        chroma = ChromaStore("rag_demo", dim=64)
        n_upserted = chroma.upsert_chunks(corpus, embedder)
        print(f"\n  Upserted {n_upserted} chunks to ChromaDB")
        print(f"  Collection count: {chroma.count()}")

        q_results = chroma.query(q_emb, n_results=3)
        print(f"\n  Query: '{query}'")
        print(f"  Top-3 from ChromaDB:")
        for doc, dist in zip(q_results["documents"][0], q_results["distances"][0]):
            print(f"    dist={dist:.4f} | {doc[:90]}...")

    section("EMBEDDING QUALITY: NEAREST NEIGHBOURS")
    queries_test = [
        "attention and transformer architecture",
        "keyword search and term frequency",
        "retrieval augmented generation pipeline",
    ]
    print()
    for q in queries_test:
        q_e = embedder.embed_one(q)
        res = store.query(q_e, n_results=2)
        print(f"  Query: {q!r}")
        for doc, dist in zip(res["documents"][0], res["distances"][0]):
            print(f"    [{dist:.4f}] {doc[:80]}...")
        print()


if __name__ == "__main__":
    main()
