"""
Retrieval strategies for RAG: cosine similarity (dense), BM25 (sparse),
hybrid fusion (RRF + linear interpolation), and MMR (diversity).
Covers: BM25 from scratch, Reciprocal Rank Fusion, MMR algorithm.
pip install numpy
"""

import re
import math
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Shared data structure
# ---------------------------------------------------------------------------

class ScoredChunk:
    def __init__(self, doc_id: str, text: str, score: float, metadata: Dict[str, Any]):
        self.doc_id = doc_id
        self.text = text
        self.score = score
        self.metadata = metadata

    def __repr__(self):
        return f"ScoredChunk(id={self.doc_id!r}, score={self.score:.4f})"


# ---------------------------------------------------------------------------
# BM25 from scratch
# ---------------------------------------------------------------------------

class BM25:
    """
    Robertson & Sparck-Jones BM25 ranking function.

    BM25(q, d) = Σ_t IDF(t) * [ f(t,d) * (k1+1) ] / [ f(t,d) + k1*(1-b+b*|d|/avgdl) ]

    IDF(t) = log( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )
      N    = corpus size
      n(t) = number of docs containing term t
      k1   = [1.2, 2.0]  TF saturation
      b    = [0, 1]       length normalisation (0.75 typical)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        self.tf: List[Dict[str, int]] = []
        self.N: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z]+", text.lower())

    def fit(self, texts: List[str], ids: List[str]) -> "BM25":
        self.doc_texts = texts
        self.doc_ids = ids
        self.corpus = [self._tokenize(t) for t in texts]
        self.N = len(self.corpus)
        self.doc_lengths = [len(tokens) for tokens in self.corpus]
        self.avgdl = sum(self.doc_lengths) / self.N if self.N else 1.0
        self.tf = [Counter(tokens) for tokens in self.corpus]

        # Document frequency
        df: Counter = Counter()
        for tokens in self.corpus:
            df.update(set(tokens))

        # IDF (Robertson variant with +1 smoothing to keep positive)
        self.idf = {}
        for term, n in df.items():
            self.idf[term] = math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

        return self

    def score(self, query: str) -> np.ndarray:
        """Return BM25 scores for all documents."""
        q_tokens = self._tokenize(query)
        scores = np.zeros(self.N)
        for t in q_tokens:
            if t not in self.idf:
                continue
            idf_t = self.idf[t]
            for i in range(self.N):
                tf_td = self.tf[i].get(t, 0)
                if tf_td == 0:
                    continue
                denom = tf_td + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[i] / self.avgdl
                )
                scores[i] += idf_t * (tf_td * (self.k1 + 1)) / denom
        return scores

    def retrieve(self, query: str, n: int = 5) -> List[ScoredChunk]:
        scores = self.score(query)
        top_idx = np.argsort(scores)[::-1][:n]
        return [
            ScoredChunk(self.doc_ids[i], self.doc_texts[i], float(scores[i]), {})
            for i in top_idx
            if scores[i] > 0
        ]


# ---------------------------------------------------------------------------
# Dense retriever (cosine similarity)
# ---------------------------------------------------------------------------

class DenseRetriever:
    """Brute-force cosine similarity retrieval over a pre-embedded corpus."""

    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.metadatas: List[Dict] = []

    def index(self, embeddings: np.ndarray, ids: List[str],
              texts: List[str], metadatas: List[Dict]) -> None:
        # Ensure unit norm
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        self.embeddings = embeddings / norms
        self.doc_ids = ids
        self.doc_texts = texts
        self.metadatas = metadatas

    def retrieve(self, query_emb: np.ndarray, n: int = 5) -> List[ScoredChunk]:
        q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        scores = self.embeddings @ q   # cosine similarity
        top_idx = np.argsort(scores)[::-1][:n]
        return [
            ScoredChunk(self.doc_ids[i], self.doc_texts[i],
                        float(scores[i]), self.metadatas[i])
            for i in top_idx
        ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(ranked_lists: List[List[ScoredChunk]],
                            k: int = 60) -> List[ScoredChunk]:
    """
    RRF(d) = Σ_ranker  1 / (k + rank(d))
    Immune to score scale mismatch. k=60 is the standard default.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, ScoredChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            rrf_scores[chunk.doc_id] = rrf_scores.get(chunk.doc_id, 0.0) + 1.0 / (k + rank)
            chunk_map[chunk.doc_id] = chunk

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    result = []
    for doc_id in sorted_ids:
        c = chunk_map[doc_id]
        result.append(ScoredChunk(c.doc_id, c.text, rrf_scores[doc_id], c.metadata))
    return result


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Combines BM25 (sparse) + Dense (semantic) via RRF or linear interpolation.
    """

    def __init__(self, dense: DenseRetriever, bm25: BM25):
        self.dense = dense
        self.bm25 = bm25

    def retrieve_rrf(self, query: str, query_emb: np.ndarray,
                     n: int = 5, candidate_n: int = 20) -> List[ScoredChunk]:
        dense_results = self.dense.retrieve(query_emb, candidate_n)
        bm25_results = self.bm25.retrieve(query, candidate_n)
        fused = reciprocal_rank_fusion([dense_results, bm25_results])
        return fused[:n]

    def retrieve_linear(self, query: str, query_emb: np.ndarray,
                        n: int = 5, alpha: float = 0.5) -> List[ScoredChunk]:
        """
        alpha * dense_score + (1 - alpha) * bm25_score.
        Both score arrays are min-max normalised before fusion.
        """
        dense_scores_arr = self.dense.embeddings @ (
            query_emb / (np.linalg.norm(query_emb) + 1e-9)
        )
        bm25_scores_arr = self.bm25.score(query)

        def minmax(arr):
            lo, hi = arr.min(), arr.max()
            if hi - lo < 1e-9:
                return np.zeros_like(arr)
            return (arr - lo) / (hi - lo)

        dense_norm = minmax(dense_scores_arr)
        bm25_norm = minmax(bm25_scores_arr)
        combined = alpha * dense_norm + (1 - alpha) * bm25_norm

        top_idx = np.argsort(combined)[::-1][:n]
        return [
            ScoredChunk(
                self.dense.doc_ids[i], self.dense.doc_texts[i],
                float(combined[i]), self.dense.metadatas[i]
            )
            for i in top_idx
        ]


# ---------------------------------------------------------------------------
# Maximum Marginal Relevance
# ---------------------------------------------------------------------------

def mmr_retrieve(query_emb: np.ndarray, candidate_embs: np.ndarray,
                 candidates: List[ScoredChunk], k: int = 5,
                 lam: float = 0.7) -> List[ScoredChunk]:
    """
    MMR_i = argmax_{d in R\S} [ λ·sim(q,d) - (1-λ)·max_{s in S} sim(d,s) ]

    lam=1 → pure relevance, lam=0 → pure diversity.
    candidate_embs: (n_candidates, dim) unit-normalised embeddings.
    """
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    embs = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-9)

    relevance = embs @ q   # (n_candidates,)
    selected_idx: List[int] = []
    remaining = list(range(len(candidates)))

    while len(selected_idx) < k and remaining:
        if not selected_idx:
            # First: pick highest relevance
            best = max(remaining, key=lambda i: relevance[i])
        else:
            # MMR score for each remaining candidate
            sel_embs = embs[selected_idx]       # (|S|, dim)
            best_score = -1e9
            best = remaining[0]
            for i in remaining:
                rel_score = lam * relevance[i]
                redundancy = (1 - lam) * float((embs[i] @ sel_embs.T).max())
                mmr_score = rel_score - redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = i
        selected_idx.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected_idx]


# ---------------------------------------------------------------------------
# Demo corpus and embedder
# ---------------------------------------------------------------------------

CORPUS = [
    ("chunk_0",  "Attention mechanism computes weighted sum of values using query-key similarities."),
    ("chunk_1",  "Multi-head attention splits the embedding into h heads for parallel attention."),
    ("chunk_2",  "BM25 ranks documents using term frequency and inverse document frequency."),
    ("chunk_3",  "Dense retrieval embeds queries and documents into the same vector space."),
    ("chunk_4",  "Hybrid search combines sparse BM25 and dense embedding scores."),
    ("chunk_5",  "RAG retrieves context chunks and injects them into the LLM prompt."),
    ("chunk_6",  "Faithfulness measures whether the answer is supported by retrieved context."),
    ("chunk_7",  "Chunking splits documents into smaller segments for embedding and retrieval."),
    ("chunk_8",  "Cosine similarity measures the angle between two embedding vectors."),
    ("chunk_9",  "HNSW builds a hierarchical graph for fast approximate nearest neighbour search."),
    ("chunk_10", "BM25 term frequency saturation is controlled by the k1 hyperparameter."),
    ("chunk_11", "Dense retrievers fail on rare proper nouns and exact keyword queries."),
    ("chunk_12", "Reciprocal Rank Fusion merges ranked lists without score normalisation."),
    ("chunk_13", "MMR trades off relevance and diversity to reduce redundancy in retrieval."),
    ("chunk_14", "The attention formula divides by sqrt(d_k) to prevent vanishing softmax gradients."),
]


def _deterministic_embed(text: str, dim: int = 64) -> np.ndarray:
    """Character bigram TF-IDF proxy embedding (no external deps)."""
    tokens = re.findall(r"[a-z]+", text.lower())
    vec = np.zeros(dim)
    for token in tokens:
        for i in range(len(token) - 1):
            h = (ord(token[i]) * 31 + ord(token[i+1])) % dim
            vec[h] += 1.0
    # IDF proxy: divide by token count
    n = max(len(tokens), 1)
    vec /= n
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec


def build_demo() -> Tuple[DenseRetriever, BM25, HybridRetriever, np.ndarray]:
    texts = [t for _, t in CORPUS]
    ids = [i for i, _ in CORPUS]
    embs = np.array([_deterministic_embed(t) for t in texts])

    dense = DenseRetriever()
    dense.index(embs, ids, texts, [{} for _ in texts])

    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(texts, ids)

    hybrid = HybridRetriever(dense, bm25)
    return dense, bm25, hybrid, embs


def main():
    dense, bm25, hybrid, corpus_embs = build_demo()

    queries = [
        ("attention mechanism transformer",   "Semantic — transformer concept"),
        ("BM25 term frequency k1 parameter",  "Lexical — exact keyword match"),
        ("retrieve documents context chunks", "Mixed — RAG pipeline"),
    ]

    for query_text, query_label in queries:
        q_emb = _deterministic_embed(query_text)

        section(f"DENSE RETRIEVAL — {query_label}")
        dense_results = dense.retrieve(q_emb, n=5)
        print(f"\n  Query: {query_text!r}")
        for r in dense_results:
            print(f"    [{r.doc_id}] score={r.score:+.4f} | {r.text[:70]}...")

        section(f"BM25 RETRIEVAL — {query_label}")
        bm25_results = bm25.retrieve(query_text, n=5)
        print(f"\n  Query: {query_text!r}")
        if bm25_results:
            for r in bm25_results:
                print(f"    [{r.doc_id}] score={r.score:.4f} | {r.text[:70]}...")
        else:
            print("  No BM25 matches (no overlapping terms)")

        section(f"HYBRID RRF — {query_label}")
        hybrid_rrf = hybrid.retrieve_rrf(query_text, q_emb, n=5, candidate_n=10)
        print(f"\n  Query: {query_text!r}  (RRF k=60)")
        for r in hybrid_rrf:
            print(f"    [{r.doc_id}] rrf={r.score:.5f} | {r.text[:70]}...")

        section(f"HYBRID LINEAR (α=0.5) — {query_label}")
        hybrid_lin = hybrid.retrieve_linear(query_text, q_emb, n=5, alpha=0.5)
        print(f"\n  Query: {query_text!r}  (α=0.5 dense, 0.5 BM25)")
        for r in hybrid_lin:
            print(f"    [{r.doc_id}] score={r.score:.4f} | {r.text[:70]}...")

        section(f"MMR (λ=0.7) — {query_label}")
        # Use top-10 dense as candidates for MMR
        candidates = dense.retrieve(q_emb, n=10)
        cand_embs = np.array([_deterministic_embed(c.text) for c in candidates])
        mmr_results = mmr_retrieve(q_emb, cand_embs, candidates, k=5, lam=0.7)
        print(f"\n  Query: {query_text!r}  (λ=0.7, candidates=10)")
        for r in mmr_results:
            print(f"    [{r.doc_id}] {r.text[:70]}...")

    section("BM25 INTERNALS — IDF TABLE (TOP 15 TERMS)")
    top_terms = sorted(bm25.idf.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"\n  {'Term':<20} {'IDF':>8}")
    print(f"  {'-'*30}")
    for term, idf in top_terms:
        print(f"  {term:<20} {idf:>8.4f}")

    section("RRF vs LINEAR vs MMR — ALPHA SENSITIVITY")
    q_text = "retrieval and embedding vectors"
    q_e = _deterministic_embed(q_text)
    print(f"\n  Query: {q_text!r}")
    print(f"\n  RRF top-3:")
    for r in hybrid.retrieve_rrf(q_text, q_e, n=3):
        print(f"    [{r.doc_id}] rrf={r.score:.5f}")
    for alpha in [0.2, 0.5, 0.8]:
        print(f"\n  Linear α={alpha} top-3:")
        for r in hybrid.retrieve_linear(q_text, q_e, n=3, alpha=alpha):
            print(f"    [{r.doc_id}] score={r.score:.4f}")

    section("MMR LAMBDA SWEEP — DIVERSITY vs RELEVANCE")
    q_text = "BM25 retrieval search"
    q_e = _deterministic_embed(q_text)
    cands = dense.retrieve(q_e, n=12)
    cand_embs = np.array([_deterministic_embed(c.text) for c in cands])
    print(f"\n  Query: {q_text!r}")
    print(f"\n  {'λ':>6}  {'Selected chunk IDs'}")
    print(f"  {'-'*50}")
    for lam in [1.0, 0.7, 0.5, 0.3, 0.0]:
        selected = mmr_retrieve(q_e, cand_embs, cands, k=4, lam=lam)
        ids_str = ", ".join(r.doc_id for r in selected)
        print(f"  {lam:>6.1f}  [{ids_str}]")
    print(f"\n  λ=1.0 → pure relevance (likely repeated similar chunks)")
    print(f"  λ=0.0 → pure diversity (maximally different chunks)")

    section("RETRIEVAL LATENCY BENCHMARK")
    N_RUNS = 200
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        dense.retrieve(q_e, n=5)
    t_dense = (time.perf_counter() - t0) / N_RUNS * 1000

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        bm25.retrieve(q_text, n=5)
    t_bm25 = (time.perf_counter() - t0) / N_RUNS * 1000

    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        hybrid.retrieve_rrf(q_text, q_e, n=5, candidate_n=10)
    t_hybrid = (time.perf_counter() - t0) / N_RUNS * 1000

    print(f"\n  Corpus size: {len(CORPUS)} chunks")
    print(f"  Dense retrieval    : {t_dense:.3f} ms/query")
    print(f"  BM25 retrieval     : {t_bm25:.3f} ms/query")
    print(f"  Hybrid RRF         : {t_hybrid:.3f} ms/query")


if __name__ == "__main__":
    main()
