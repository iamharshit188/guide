"""
Full RAG pipeline Flask API: ingest, query, health, collection management.
Covers: end-to-end RAG (ingest → embed → store → retrieve → generate),
        REST API design, retrieval strategy selection, source attribution.
pip install flask numpy  (chromadb optional)
"""

import os
import re
import json
import math
import time
import tempfile
import hashlib
from typing import List, Dict, Any, Optional
from collections import Counter

try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("flask not installed — demo will run in CLI mode")
    print("pip install flask")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Inline implementations (self-contained — no cross-file imports)
# ---------------------------------------------------------------------------

# --- Chunking ---

def chunk_fixed(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    chunks, stride = [], chunk_size - overlap
    i = 0
    while i < len(text):
        end = min(i + chunk_size, len(text))
        t = text[i:end].strip()
        if t:
            chunks.append(t)
        i += stride
        if end == len(text):
            break
    return chunks


def chunk_recursive(text: str, max_chars: int = 400, overlap: int = 50,
                    seps: List[str] = None) -> List[str]:
    if seps is None:
        seps = ["\n\n", "\n", ". ", " ", ""]
    if len(text) <= max_chars:
        t = text.strip()
        return [t] if t else []
    sep = seps[0]
    rest = seps[1:]
    if sep == "":
        return chunk_fixed(text, max_chars, overlap)
    pieces = text.split(sep)
    chunks, current = [], ""
    for piece in pieces:
        candidate = current + (sep if current else "") + piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current.strip():
                chunks.extend(chunk_recursive(current, max_chars, overlap, rest))
            current = piece
    if current.strip():
        chunks.extend(chunk_recursive(current, max_chars, overlap, rest))
    return chunks


# --- Embedding ---

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


class TFIDFEmbedder:
    def __init__(self, dim: int = 128, seed: int = 42):
        self.dim = dim
        self._rng_seed = seed
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.projection = None
        self._fitted = False

    def fit(self, texts: List[str]) -> "TFIDFEmbedder":
        import numpy as np
        rng = np.random.default_rng(self._rng_seed)
        N = len(texts)
        df: Counter = Counter()
        for t in texts:
            df.update(set(_tokenize(t)))
        vocab_terms = sorted(df.keys())
        self.vocab = {t: i for i, t in enumerate(vocab_terms)}
        V = len(self.vocab)
        self.idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in self.vocab}
        self.projection = rng.standard_normal((V, self.dim)) / math.sqrt(self.dim)
        self._fitted = True
        return self

    def embed(self, texts: List[str]):
        import numpy as np
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        vecs = []
        for text in texts:
            tokens = _tokenize(text)
            tf = Counter(tokens)
            max_tf = max(tf.values()) if tf else 1
            vec = [0.0] * len(self.vocab)
            for term, cnt in tf.items():
                if term in self.vocab:
                    vec[self.vocab[term]] = (cnt / max_tf) * self.idf.get(term, 1.0)
            v = np.array(vec) @ self.projection
            n = np.linalg.norm(v)
            vecs.append(v / n if n > 1e-9 else v)
        return np.array(vecs)

    def embed_one(self, text: str):
        return self.embed([text])[0]


# --- BM25 ---

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus_tokens: List[List[str]] = []
        self.doc_texts: List[str] = []
        self.idf: Dict[str, float] = {}
        self.tf: List[Counter] = []
        self.avgdl: float = 1.0
        self.N: int = 0

    def fit(self, texts: List[str]) -> "BM25":
        self.doc_texts = texts
        self.corpus_tokens = [_tokenize(t) for t in texts]
        self.N = len(self.corpus_tokens)
        lens = [len(t) for t in self.corpus_tokens]
        self.avgdl = sum(lens) / self.N if self.N else 1.0
        self.tf = [Counter(t) for t in self.corpus_tokens]
        df: Counter = Counter()
        for t in self.corpus_tokens:
            df.update(set(t))
        self.idf = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)
                    for t, n in df.items()}
        return self

    def scores(self, query: str):
        import numpy as np
        s = np.zeros(self.N)
        for term in _tokenize(query):
            if term not in self.idf:
                continue
            idf_t = self.idf[term]
            for i in range(self.N):
                f = self.tf[i].get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * len(self.corpus_tokens[i]) / self.avgdl)
                s[i] += idf_t * f * (self.k1 + 1) / denom
        return s


# --- In-memory vector store ---

class VectorStore:
    def __init__(self):
        import numpy as np
        self._ids: List[str] = []
        self._embs = None
        self._docs: List[str] = []
        self._metas: List[Dict] = []

    def upsert(self, ids, embs, docs, metas):
        import numpy as np
        for i, doc_id in enumerate(ids):
            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                self._embs[idx] = embs[i]
                self._docs[idx] = docs[i]
                self._metas[idx] = metas[i]
            else:
                self._ids.append(doc_id)
                self._docs.append(docs[i])
                self._metas.append(metas[i])
        # Rebuild embedding matrix
        all_embs = ([self._embs[j] for j in range(len(self._ids) - len(ids))]
                    if self._embs is not None and len(self._ids) > len(ids)
                    else []) + list(embs)
        # Simpler: always rebuild
        import numpy as np
        collect = []
        for i in range(len(self._ids)):
            if i < len(self._ids) - len(ids):
                collect.append(self._embs[i] if self._embs is not None else embs[0])
            else:
                collect.append(embs[i - (len(self._ids) - len(ids))])
        self._embs = np.array(collect) if collect else np.empty((0, embs.shape[1]))

    def _rebuild(self, ids, embs, docs, metas):
        import numpy as np
        self._ids = list(ids)
        self._docs = list(docs)
        self._metas = list(metas)
        self._embs = np.array(embs)

    def query(self, q_emb, n=5):
        import numpy as np
        if self._embs is None or len(self._ids) == 0:
            return []
        sims = self._embs @ q_emb
        top = np.argsort(sims)[::-1][:n]
        return [(self._ids[i], self._docs[i], self._metas[i], float(sims[i])) for i in top]

    def count(self) -> int:
        return len(self._ids)

    def clear(self):
        import numpy as np
        self._ids = []
        self._embs = None
        self._docs = []
        self._metas = []


# --- Mock LLM ---

class MockLLM:
    def generate(self, query: str, context_chunks: List[str]) -> str:
        q_tokens = set(_tokenize(query))
        best_score, best_sent = 0.0, ""
        for chunk in context_chunks:
            for sent in re.split(r"(?<=[.!?])\s+", chunk):
                if len(sent) < 20:
                    continue
                sent_tokens = set(_tokenize(sent))
                score = len(q_tokens & sent_tokens) / (len(q_tokens | sent_tokens) + 1e-9)
                if score > best_score:
                    best_score, best_sent = score, sent
        if best_score < 0.05:
            return "I don't know based on the provided context."
        return best_sent.strip() + (" " if not best_sent.endswith(".") else "")


# ---------------------------------------------------------------------------
# RAG Pipeline state (in-memory, reset per demo run)
# ---------------------------------------------------------------------------

class RAGPipeline:
    def __init__(self, chunk_size: int = 400, overlap: int = 50,
                 embed_dim: int = 128, top_k: int = 5):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embed_dim = embed_dim
        self.top_k = top_k

        self.store = VectorStore()
        self.embedder: Optional[TFIDFEmbedder] = None
        self.bm25: Optional[BM25] = None
        self.llm = MockLLM()

        self._all_texts: List[str] = []
        self._all_metas: List[Dict] = []
        self._doc_count = 0

    def ingest(self, text: str, source: str = "document",
               strategy: str = "recursive") -> Dict[str, Any]:
        """Chunk, embed, and store a document."""
        import numpy as np
        t0 = time.perf_counter()

        if strategy == "fixed":
            chunks = chunk_fixed(text, self.chunk_size, self.overlap)
        else:
            chunks = chunk_recursive(text, self.chunk_size, self.overlap)

        metas = [{"source": source, "chunk_index": i, "doc_id": self._doc_count}
                 for i in range(len(chunks))]

        self._all_texts.extend(chunks)
        self._all_metas.extend(metas)

        # Refit embedder and BM25 on full corpus
        self.embedder = TFIDFEmbedder(dim=self.embed_dim, seed=42)
        self.embedder.fit(self._all_texts)
        self.bm25 = BM25()
        self.bm25.fit(self._all_texts)

        embs = self.embedder.embed(self._all_texts)
        ids = [f"doc{m['doc_id']}_chunk{m['chunk_index']}" for m in self._all_metas]
        self.store._rebuild(ids, embs, self._all_texts, self._all_metas)

        self._doc_count += 1
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "chunks_added": len(chunks),
            "total_chunks": self.store.count(),
            "source": source,
            "strategy": strategy,
            "ingest_ms": round(elapsed, 1),
        }

    def query(self, query: str, top_k: int = None,
              strategy: str = "hybrid") -> Dict[str, Any]:
        """Run full RAG: retrieve → build context → generate."""
        import numpy as np
        if self.embedder is None or self.store.count() == 0:
            return {"error": "No documents ingested yet"}

        k = top_k or self.top_k
        t0 = time.perf_counter()

        q_emb = self.embedder.embed_one(query)

        if strategy == "dense":
            results = self.store.query(q_emb, n=k)
        elif strategy == "bm25":
            bm25_scores = self.bm25.scores(query)
            top_idx = np.argsort(bm25_scores)[::-1][:k]
            all_docs = self._all_texts
            all_metas = self._all_metas
            ids = [f"doc{m['doc_id']}_chunk{m['chunk_index']}" for m in all_metas]
            results = [(ids[i], all_docs[i], all_metas[i], float(bm25_scores[i]))
                       for i in top_idx if bm25_scores[i] > 0]
        elif strategy == "mmr":
            candidates = self.store.query(q_emb, n=min(k * 3, self.store.count()))
            results = self._mmr_select(q_emb, candidates, k)
        else:  # hybrid
            dense_results = self.store.query(q_emb, n=k * 2)
            bm25_scores = self.bm25.scores(query)
            top_bm25_idx = np.argsort(bm25_scores)[::-1][:k * 2]
            all_ids = [f"doc{m['doc_id']}_chunk{m['chunk_index']}"
                       for m in self._all_metas]
            bm25_results = [(all_ids[i], self._all_texts[i], self._all_metas[i],
                             float(bm25_scores[i])) for i in top_bm25_idx]
            results = self._rrf_fuse(dense_results, bm25_results, k=k)

        retrieval_ms = (time.perf_counter() - t0) * 1000

        context_chunks = [doc for _, doc, _, _ in results[:k]]
        sources = [{"source": meta.get("source"), "chunk_index": meta.get("chunk_index"),
                    "score": round(score, 4)}
                   for _, _, meta, score in results[:k]]

        t1 = time.perf_counter()
        answer = self.llm.generate(query, context_chunks)
        gen_ms = (time.perf_counter() - t1) * 1000

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_strategy": strategy,
            "chunks_retrieved": len(results[:k]),
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(gen_ms, 1),
        }

    def _mmr_select(self, q_emb, candidates, k):
        import numpy as np
        if not candidates:
            return []
        cand_embs = np.array([self.embedder.embed_one(doc) for _, doc, _, _ in candidates])
        q = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        embs = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-9)
        relevance = embs @ q
        selected, remaining = [], list(range(len(candidates)))
        while len(selected) < k and remaining:
            if not selected:
                best = max(remaining, key=lambda i: relevance[i])
            else:
                sel_embs = embs[selected]
                scores = [0.7 * relevance[i] - 0.3 * float((embs[i] @ sel_embs.T).max())
                          for i in remaining]
                best = remaining[int(np.argmax(scores))]
            selected.append(best)
            remaining.remove(best)
        return [candidates[i] for i in selected]

    def _rrf_fuse(self, list_a, list_b, k=5, rrf_k=60):
        scores = {}
        refs = {}
        for rank, (doc_id, doc, meta, score) in enumerate(list_a, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank)
            refs[doc_id] = (doc_id, doc, meta, score)
        for rank, (doc_id, doc, meta, score) in enumerate(list_b, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rrf_k + rank)
            refs.setdefault(doc_id, (doc_id, doc, meta, score))
        top = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
        return [(doc_id, refs[doc_id][1], refs[doc_id][2], scores[doc_id])
                for doc_id in top]

    def collection_info(self) -> Dict[str, Any]:
        return {
            "total_chunks": self.store.count(),
            "embed_dim": self.embed_dim,
            "vocab_size": len(self.embedder.vocab) if self.embedder else 0,
            "documents_ingested": self._doc_count,
        }

    def clear(self):
        self.store.clear()
        self.embedder = None
        self.bm25 = None
        self._all_texts = []
        self._all_metas = []
        self._doc_count = 0


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

pipeline = RAGPipeline(chunk_size=300, overlap=50, embed_dim=128, top_k=5)


def create_app() -> "Flask":
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "chunks": pipeline.store.count()})

    @app.route("/ingest", methods=["POST"])
    def ingest():
        data = request.get_json(force=True) or {}
        text = data.get("text", "")
        source = data.get("source", "api_document")
        strategy = data.get("strategy", "recursive")
        if not text:
            return jsonify({"error": "text field required"}), 400
        result = pipeline.ingest(text, source=source, strategy=strategy)
        return jsonify(result), 201

    @app.route("/query", methods=["POST"])
    def query_endpoint():
        data = request.get_json(force=True) or {}
        query_text = data.get("query", "")
        top_k = int(data.get("top_k", 5))
        strategy = data.get("strategy", "hybrid")
        if not query_text:
            return jsonify({"error": "query field required"}), 400
        if pipeline.store.count() == 0:
            return jsonify({"error": "No documents ingested"}), 400
        result = pipeline.query(query_text, top_k=top_k, strategy=strategy)
        return jsonify(result)

    @app.route("/collection", methods=["GET"])
    def collection_info():
        return jsonify(pipeline.collection_info())

    @app.route("/collection", methods=["DELETE"])
    def clear_collection():
        pipeline.clear()
        return jsonify({"status": "cleared"})

    return app


# ---------------------------------------------------------------------------
# CLI demo (runs without Flask server)
# ---------------------------------------------------------------------------

DEMO_CORPUS = {
    "transformers.txt": (
        "The attention mechanism is the core of transformer models. "
        "Scaled dot-product attention computes queries, keys and values. "
        "The scores are divided by sqrt(d_k) to prevent gradient vanishing in softmax. "
        "Multi-head attention projects into h subspaces for richer representations.\n\n"
        "The encoder processes the full input sequence with bidirectional attention. "
        "The decoder uses masked self-attention to prevent looking at future tokens. "
        "Cross-attention in the decoder attends to encoder outputs. "
        "Layer normalisation is applied before each sub-layer in the Pre-LN variant."
    ),
    "retrieval.txt": (
        "BM25 is a probabilistic retrieval function based on term frequency statistics. "
        "The k1 hyperparameter controls term frequency saturation. "
        "The b parameter controls document length normalisation, typically set to 0.75. "
        "Inverse document frequency penalises common terms that appear in many documents.\n\n"
        "Dense retrieval embeds queries and documents into shared vector spaces. "
        "Cosine similarity measures semantic relatedness between embedding vectors. "
        "Hybrid retrieval combines sparse BM25 and dense embedding scores. "
        "Reciprocal Rank Fusion merges ranked lists without requiring score normalisation."
    ),
    "rag.txt": (
        "Retrieval-Augmented Generation retrieves context before language model generation. "
        "The retrieved chunks are injected into the prompt as context for the LLM. "
        "RAG grounds answers in external evidence, reducing hallucination rates. "
        "Source attribution is trivial with RAG because chunk metadata tracks origins.\n\n"
        "RAGAS evaluates RAG pipelines using faithfulness and answer relevancy metrics. "
        "Faithfulness checks if every claim in the answer is supported by the context. "
        "Answer relevancy measures whether the response addresses the original question. "
        "Context precision measures the fraction of retrieved chunks that are relevant."
    ),
}


def run_cli_demo():
    section("RAG PIPELINE — CLI DEMO (no Flask server)")

    section("DOCUMENT INGESTION")
    for source, text in DEMO_CORPUS.items():
        result = pipeline.ingest(text, source=source, strategy="recursive")
        print(f"\n  Ingested '{source}': {result['chunks_added']} chunks "
              f"({result['ingest_ms']:.1f}ms)")
    print(f"\n  Total chunks in store: {pipeline.store.count()}")
    info = pipeline.collection_info()
    print(f"  Vocab size: {info['vocab_size']}  |  Embed dim: {info['embed_dim']}")

    section("QUERY — RETRIEVAL STRATEGIES")
    queries = [
        ("How does the attention mechanism prevent gradient issues?", "dense"),
        ("BM25 term frequency k1 saturation parameter",              "bm25"),
        ("What makes RAG reduce hallucinations?",                     "hybrid"),
        ("How does cross attention work in the decoder?",             "mmr"),
    ]

    for query_text, strategy in queries:
        result = pipeline.query(query_text, top_k=3, strategy=strategy)
        print(f"\n  Strategy={strategy!r}")
        print(f"  Q: {query_text!r}")
        print(f"  A: {result['answer']}")
        print(f"  Sources:")
        for s in result["sources"]:
            print(f"    - {s['source']} [chunk {s['chunk_index']}] score={s['score']}")
        print(f"  Retrieval: {result['retrieval_ms']:.1f}ms | "
              f"Generation: {result['generation_ms']:.1f}ms")

    section("COLLECTION INFO")
    print(f"\n  {json.dumps(pipeline.collection_info(), indent=4)}")

    section("PIPELINE LATENCY BENCHMARK")
    import time
    N = 50
    queries_bench = [q for q, _ in queries]
    strategies_bench = ["dense", "bm25", "hybrid", "mmr"]

    print(f"\n  {'Strategy':<10} {'Mean (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
    print(f"  {'-'*44}")
    for strat in strategies_bench:
        times = []
        for i in range(N):
            q = queries_bench[i % len(queries_bench)]
            t0 = time.perf_counter()
            pipeline.query(q, top_k=3, strategy=strat)
            times.append((time.perf_counter() - t0) * 1000)
        print(f"  {strat:<10} {sum(times)/N:>10.2f} {min(times):>10.2f} {max(times):>10.2f}")

    section("CLEAR AND RE-INGEST")
    pipeline.clear()
    print(f"\n  After clear: {pipeline.store.count()} chunks")
    pipeline.ingest(DEMO_CORPUS["rag.txt"], source="rag.txt")
    print(f"  After re-ingest: {pipeline.store.count()} chunks")
    r = pipeline.query("What does RAG reduce?", top_k=2, strategy="dense")
    print(f"  Quick query: {r['answer']}")


def main():
    run_cli_demo()

    if FLASK_AVAILABLE:
        section("FLASK APP — ROUTES REGISTERED")
        app = create_app()
        print("\n  Routes:")
        for rule in app.url_map.iter_rules():
            print(f"    {', '.join(rule.methods - {'HEAD', 'OPTIONS'}):>10}  {rule.rule}")
        print("\n  To run: python3.14 app.py --serve")
        print("  Then:   curl -X POST http://localhost:5000/ingest "
              "-H 'Content-Type: application/json' "
              "-d '{\"text\": \"...\", \"source\": \"doc.txt\"}'")

        import sys
        if "--serve" in sys.argv:
            app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        section("FLASK NOT INSTALLED")
        print("\n  pip install flask")


if __name__ == "__main__":
    main()
