"""
Pinecone demo — runs in MOCK MODE using a FAISS-backed local client.
The mock exposes the identical API surface as the real Pinecone SDK.

To switch to real Pinecone:
  pip install pinecone
  Replace MockPinecone() with:
      from pinecone import Pinecone, ServerlessSpec
      pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
  All subsequent calls (upsert, query, fetch, delete) are identical.

No API key or internet required in mock mode.
"""

import numpy as np
import time
import uuid
from typing import Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Mock Pinecone Client (FAISS-backed) ───────────────────────

class MockQueryResult:
    def __init__(self, matches):
        self.matches = matches


class MockMatch:
    def __init__(self, id, score, values=None, metadata=None):
        self.id = id
        self.score = score
        self.values = values or []
        self.metadata = metadata or {}


class MockIndex:
    """
    Mirrors the Pinecone Index API:
      .upsert(), .query(), .fetch(), .delete(),
      .describe_index_stats(), .update()
    Backed by a dict store + numpy for exact cosine search.
    Namespaces are isolated sub-stores.
    """

    def __init__(self, name, dimension, metric="cosine"):
        self.name = name
        self.dimension = dimension
        self.metric = metric
        # namespace → {id: {"values": [...], "metadata": {...}, "sparse": {...}}}
        self._store = {}
        self._total_ops = 0

    def _ns(self, namespace=""):
        return self._store.setdefault(namespace, {})

    def _similarity(self, q, v):
        q, v = np.array(q), np.array(v)
        if self.metric == "cosine":
            return float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-10))
        elif self.metric == "dotproduct":
            return float(np.dot(q, v))
        elif self.metric == "euclidean":
            return -float(np.linalg.norm(q - v))  # negative so higher = closer
        return 0.0

    def upsert(self, vectors, namespace=""):
        """
        Pinecone: index.upsert(vectors=[
            {"id": "v1", "values": [...], "metadata": {...}},
            ...
        ], namespace="ns")
        """
        ns = self._ns(namespace)
        count = 0
        for vec in vectors:
            vid = vec["id"]
            ns[vid] = {
                "values": vec.get("values", []),
                "metadata": vec.get("metadata", {}),
                "sparse": vec.get("sparse_values", {}),
            }
            count += 1
        self._total_ops += count
        return {"upserted_count": count}

    def query(self, vector, top_k=10, namespace="", filter=None,
              include_values=False, include_metadata=True,
              sparse_vector=None, alpha=1.0):
        """
        Pinecone: index.query(
            vector=[...],
            top_k=5,
            namespace="ns",
            filter={"category": {"$eq": "ml"}},
            include_metadata=True,
        )
        alpha: weight for dense vector in hybrid search (0=sparse, 1=dense)
        """
        ns = self._ns(namespace)
        if not ns:
            return MockQueryResult([])

        def _matches_filter(metadata, filt):
            if not filt:
                return True
            for key, cond in filt.items():
                val = metadata.get(key)
                if isinstance(cond, dict):
                    for op, operand in cond.items():
                        if op == "$eq"  and val != operand: return False
                        if op == "$ne"  and val == operand: return False
                        if op == "$gt"  and not (val > operand): return False
                        if op == "$gte" and not (val >= operand): return False
                        if op == "$lt"  and not (val < operand): return False
                        if op == "$lte" and not (val <= operand): return False
                        if op == "$in"  and val not in operand: return False
                        if op == "$nin" and val in operand: return False
                else:
                    if val != cond: return False
            return True

        def _sparse_score(sparse_q, sparse_doc):
            if not sparse_q or not sparse_doc:
                return 0.0
            q_idx = set(sparse_q.get("indices", []))
            total = 0.0
            for idx, val in zip(sparse_doc.get("indices", []), sparse_doc.get("values", [])):
                if idx in q_idx:
                    q_val = sparse_q["values"][sparse_q["indices"].index(idx)]
                    total += q_val * val
            return total

        scored = []
        for vid, data in ns.items():
            if not _matches_filter(data["metadata"], filter):
                continue
            dense_score = self._similarity(vector, data["values"]) if data["values"] else 0.0
            sparse_score = _sparse_score(sparse_vector, data["sparse"]) if sparse_vector else 0.0
            score = alpha * dense_score + (1 - alpha) * sparse_score
            match = MockMatch(
                id=vid,
                score=score,
                values=data["values"] if include_values else None,
                metadata=data["metadata"] if include_metadata else None,
            )
            scored.append(match)

        scored.sort(key=lambda m: m.score, reverse=True)
        return MockQueryResult(scored[:top_k])

    def fetch(self, ids, namespace=""):
        """Pinecone: index.fetch(ids=["id1","id2"], namespace="ns")"""
        ns = self._ns(namespace)
        vectors = {}
        for vid in ids:
            if vid in ns:
                vectors[vid] = {
                    "id": vid,
                    "values": ns[vid]["values"],
                    "metadata": ns[vid]["metadata"],
                }
        return {"vectors": vectors, "namespace": namespace}

    def delete(self, ids=None, delete_all=False, namespace="", filter=None):
        """Pinecone: index.delete(ids=[...], namespace="ns")"""
        ns = self._ns(namespace)
        if delete_all:
            self._store[namespace] = {}
            return {}
        if ids:
            for vid in ids:
                ns.pop(vid, None)
        if filter:
            to_del = [vid for vid, data in ns.items()
                      if all(data["metadata"].get(k) == v for k, v in filter.items())]
            for vid in to_del:
                del ns[vid]
        return {}

    def update(self, id, values=None, metadata=None, namespace=""):
        """Pinecone: index.update(id="v1", set_metadata={...}, namespace="ns")"""
        ns = self._ns(namespace)
        if id in ns:
            if values is not None:
                ns[id]["values"] = values
            if metadata is not None:
                ns[id]["metadata"].update(metadata)
        return {}

    def describe_index_stats(self):
        total = sum(len(ns) for ns in self._store.values())
        ns_stats = {ns: {"vector_count": len(vecs)} for ns, vecs in self._store.items()}
        return {
            "dimension": self.dimension,
            "index_fullness": total / 1_000_000,
            "total_vector_count": total,
            "namespaces": ns_stats,
        }


class MockPinecone:
    """Mirrors pinecone.Pinecone client."""

    def __init__(self, api_key="mock"):
        self.api_key = api_key
        self._indexes = {}
        self._specs = {}

    def create_index(self, name, dimension, metric="cosine", spec=None):
        """
        Real: pc.create_index(name="idx", dimension=384, metric="cosine",
                              spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        """
        self._indexes[name] = MockIndex(name, dimension, metric)
        self._specs[name] = spec or {"type": "mock"}
        print(f"  [mock] Index '{name}' created (dim={dimension}, metric={metric})")

    def Index(self, name):
        """Real: index = pc.Index("my-index")"""
        if name not in self._indexes:
            raise KeyError(f"Index '{name}' does not exist")
        return self._indexes[name]

    def list_indexes(self):
        return [{"name": n, "dimension": i.dimension, "metric": i.metric}
                for n, i in self._indexes.items()]

    def delete_index(self, name):
        self._indexes.pop(name, None)


# ── Helpers ───────────────────────────────────────────────────

def random_embedding(dim=128, seed=None):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return (v / np.linalg.norm(v)).tolist()


def text_to_fake_embedding(text, dim=128):
    seed = sum(ord(c) for c in text) % (2**31)
    return random_embedding(dim, seed)


def bm25_sparse(text, vocab_size=10000):
    """Fake BM25 sparse vector for hybrid search demo."""
    words = text.lower().split()
    indices, values = [], []
    seen = set()
    for word in words:
        idx = hash(word) % vocab_size
        if idx not in seen:
            indices.append(idx)
            values.append(1.0 / len(words))
            seen.add(idx)
    return {"indices": indices, "values": values}


# ── Main ──────────────────────────────────────────────────────

def main():
    section("1. CLIENT SETUP & INDEX CREATION")
    pc = MockPinecone(api_key="mock-key-xxxx")

    # Create serverless index (384 dim = all-MiniLM-L6-v2 output size)
    pc.create_index(
        name="ml-knowledge",
        dimension=128,    # 128 for demo; use 384/768/1536 in production
        metric="cosine",
        # Real Pinecone: spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    index = pc.Index("ml-knowledge")

    section("2. UPSERT VECTORS")

    # Build a corpus
    docs = [
        {"id": "doc_001", "text": "Transformers use self-attention for sequence modeling",
         "category": "nlp", "year": 2017, "score": 9.5},
        {"id": "doc_002", "text": "BERT pre-trains on masked language modeling objectives",
         "category": "nlp", "year": 2019, "score": 9.0},
        {"id": "doc_003", "text": "GPT generates text by predicting the next token autoregressively",
         "category": "llm", "year": 2020, "score": 9.2},
        {"id": "doc_004", "text": "LoRA fine-tunes models with low-rank weight decomposition",
         "category": "finetuning", "year": 2022, "score": 8.8},
        {"id": "doc_005", "text": "RAG retrieves relevant documents to ground LLM generation",
         "category": "rag", "year": 2020, "score": 8.5},
        {"id": "doc_006", "text": "Vector databases store and retrieve dense embeddings efficiently",
         "category": "databases", "year": 2023, "score": 7.5},
        {"id": "doc_007", "text": "Convolutional networks extract image features through local filters",
         "category": "cv", "year": 2015, "score": 9.8},
        {"id": "doc_008", "text": "Diffusion models denoise Gaussian noise to generate images",
         "category": "generative", "year": 2021, "score": 9.0},
        {"id": "doc_009", "text": "Reinforcement learning from human feedback aligns LLMs to preferences",
         "category": "rlhf", "year": 2022, "score": 9.1},
        {"id": "doc_010", "text": "Quantization reduces model size by using lower-precision weights",
         "category": "efficiency", "year": 2023, "score": 8.0},
    ]

    vectors = [{
        "id": d["id"],
        "values": text_to_fake_embedding(d["text"]),
        "metadata": {"category": d["category"], "year": d["year"],
                     "score": d["score"], "text": d["text"]},
    } for d in docs]

    t0 = time.perf_counter()
    result = index.upsert(vectors=vectors, namespace="prod")
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"Upserted {result['upserted_count']} vectors in {elapsed:.2f}ms")
    print(f"Stats: {index.describe_index_stats()}")

    section("3. BASIC SIMILARITY QUERY")

    queries = [
        "how do language models predict text",
        "efficient fine-tuning of large models",
        "visual recognition with deep learning",
    ]
    for q_text in queries:
        q_vec = text_to_fake_embedding(q_text)
        result = index.query(
            vector=q_vec,
            top_k=3,
            namespace="prod",
            include_metadata=True,
        )
        print(f"\nQuery: '{q_text}'")
        print(f"  {'ID':>10}  {'Score':>8}  {'Category':>12}  text snippet")
        print("  " + "-" * 70)
        for match in result.matches:
            print(f"  {match.id:>10}  {match.score:>8.4f}  {match.metadata['category']:>12}  "
                  f"{match.metadata['text'][:45]}...")

    section("4. METADATA FILTERING")

    # $eq filter
    result_nlp = index.query(
        vector=text_to_fake_embedding("neural language model pretraining"),
        top_k=5,
        namespace="prod",
        filter={"category": {"$eq": "nlp"}},
        include_metadata=True,
    )
    print(f"Filter: category == 'nlp'")
    for m in result_nlp.matches:
        print(f"  {m.id}: score={m.score:.4f} | {m.metadata['text'][:55]}...")

    # Range filter: year >= 2021
    result_recent = index.query(
        vector=text_to_fake_embedding("modern AI techniques"),
        top_k=5,
        namespace="prod",
        filter={"year": {"$gte": 2021}},
        include_metadata=True,
    )
    print(f"\nFilter: year >= 2021")
    for m in result_recent.matches:
        print(f"  {m.id}: score={m.score:.4f} | year={m.metadata['year']} | {m.metadata['category']}")

    # Combined filter
    result_combined = index.query(
        vector=text_to_fake_embedding("text generation"),
        top_k=5,
        namespace="prod",
        filter={"$and": [
            {"year": {"$gte": 2020}},
            {"score": {"$gte": 9.0}},
        ]},
        include_metadata=True,
    )
    print(f"\nFilter: year>=2020 AND score>=9.0")
    for m in result_combined.matches:
        print(f"  {m.id}: score={m.score:.4f} | year={m.metadata['year']} | "
              f"rating={m.metadata['score']}")

    section("5. NAMESPACES")

    # Upsert same IDs into different namespaces
    prod_vecs = [{"id": "shared_001", "values": text_to_fake_embedding("production model v1"),
                  "metadata": {"env": "prod", "version": 1}}]
    dev_vecs  = [{"id": "shared_001", "values": text_to_fake_embedding("development model v2"),
                  "metadata": {"env": "dev", "version": 2}}]

    index.upsert(vectors=prod_vecs, namespace="prod")
    index.upsert(vectors=dev_vecs, namespace="dev")
    index.upsert(vectors=prod_vecs, namespace="staging")

    stats = index.describe_index_stats()
    print(f"Namespace isolation:")
    for ns, info in stats["namespaces"].items():
        print(f"  namespace='{ns}': {info['vector_count']} vectors")

    q_vec = text_to_fake_embedding("model")
    for ns in ["prod", "dev"]:
        r = index.fetch(ids=["shared_001"], namespace=ns)
        meta = r["vectors"].get("shared_001", {}).get("metadata", {})
        print(f"  fetch shared_001 from '{ns}': env={meta.get('env')}, v={meta.get('version')}")

    section("6. HYBRID SEARCH (SPARSE + DENSE)")
    print("Hybrid search combines dense (semantic) + sparse (keyword/BM25) scores:")
    print("  final_score = α * dense_score + (1-α) * sparse_score")

    hybrid_docs = [
        {"id": "h1", "text": "LoRA low-rank adaptation for efficient fine-tuning",
         "category": "finetuning"},
        {"id": "h2", "text": "Attention mechanism in transformer encoder decoder",
         "category": "nlp"},
        {"id": "h3", "text": "LoRA rank decomposition reduces trainable parameters",
         "category": "finetuning"},
    ]
    hybrid_vecs = [{
        "id": d["id"],
        "values": text_to_fake_embedding(d["text"]),
        "sparse_values": bm25_sparse(d["text"]),
        "metadata": {"category": d["category"], "text": d["text"]},
    } for d in hybrid_docs]
    index.upsert(vectors=hybrid_vecs, namespace="hybrid")

    q_text = "LoRA fine-tuning"
    q_dense = text_to_fake_embedding(q_text)
    q_sparse = bm25_sparse(q_text)

    for alpha in [0.0, 0.5, 1.0]:
        result_h = index.query(
            vector=q_dense,
            sparse_vector=q_sparse,
            top_k=3,
            namespace="hybrid",
            alpha=alpha,
            include_metadata=True,
        )
        label = {0.0: "sparse only", 0.5: "hybrid (50/50)", 1.0: "dense only"}[alpha]
        top = result_h.matches[0] if result_h.matches else None
        if top:
            print(f"  α={alpha:.1f} ({label}): top={top.id}, score={top.score:.4f} | {top.metadata['text'][:45]}...")

    section("7. FETCH, UPDATE, DELETE")

    # Fetch specific IDs
    fetched = index.fetch(ids=["doc_001", "doc_002", "doc_999"], namespace="prod")
    print(f"Fetched IDs: {list(fetched['vectors'].keys())}  (doc_999 missing — not returned)")

    # Update metadata (e.g., after re-evaluation)
    index.update(id="doc_001", metadata={"score": 9.9, "updated": True}, namespace="prod")
    updated = index.fetch(ids=["doc_001"], namespace="prod")
    print(f"After update doc_001: score={updated['vectors']['doc_001']['metadata']['score']}")

    # Delete by ID
    index.delete(ids=["doc_010"], namespace="prod")
    print(f"After delete doc_010: {index.describe_index_stats()['total_vector_count']} total vectors")

    # Delete by filter
    index.delete(filter={"category": "cv"}, namespace="prod")

    # Delete all in namespace
    index.delete(delete_all=True, namespace="staging")
    print(f"After clear 'staging': {index.describe_index_stats()['namespaces'].get('staging', {})}")

    section("8. PINECONE BEST PRACTICES")
    print("""
  Production tips:
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Always L2-normalize embeddings before upsert when        │
  │    using cosine metric (dot product ≡ cosine for ||v||=1).  │
  │                                                             │
  │ 2. Batch upserts in chunks of 100 vectors max per request.  │
  │    Parallel batches: use asyncio or ThreadPoolExecutor.     │
  │                                                             │
  │ 3. Namespaces for multi-tenancy: one namespace per customer  │
  │    → zero cross-contamination, efficient delete_all.        │
  │                                                             │
  │ 4. Metadata filtering on high-cardinality fields is slower   │
  │    than filtering on low-cardinality fields (year, category).│
  │                                                             │
  │ 5. Hybrid search: set α=0.7-0.9 for semantic-heavy queries; │
  │    α=0.3-0.5 for keyword-heavy (e.g., code search, IDs).   │
  │                                                             │
  │ 6. Pod sizing: p1.x1 → 1M vecs 768-dim; p2 for < 100ms     │
  │    p99 latency; s1 for cost-efficient storage.              │
  └─────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
