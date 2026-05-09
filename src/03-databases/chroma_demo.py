"""
ChromaDB deep dive — in-memory mode (no API key, no server needed).
pip install chromadb

Uses random numpy embeddings so no sentence-transformer download required.
Swap `fake_embed()` for any real embedding function in production.
"""

import numpy as np
import time
import random

try:
    import chromadb
    from chromadb import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("chromadb not installed. Run: pip install chromadb")
    print("Showing code structure and expected output.\n")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Fake embeddings (replace with real model in production) ───

rng_g = np.random.default_rng(42)

def fake_embed(texts, dim=128):
    """
    Production replacement:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(texts).tolist()
    """
    embeddings = []
    for text in texts:
        seed = sum(ord(c) for c in text) % (2**31)
        local_rng = np.random.default_rng(seed)
        vec = local_rng.standard_normal(dim)
        vec = vec / (np.linalg.norm(vec) + 1e-10)   # L2 normalize → cosine ≡ dot
        embeddings.append(vec.tolist())
    return embeddings


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ── Sample corpus ─────────────────────────────────────────────

CORPUS = [
    # ML papers
    {"id": "p001", "text": "Attention mechanisms in transformer models enable parallel sequence processing.",
     "source": "arxiv", "year": 2017, "topic": "transformers", "citations": 80000},
    {"id": "p002", "text": "BERT uses bidirectional pre-training for deep language understanding.",
     "source": "arxiv", "year": 2019, "topic": "transformers", "citations": 45000},
    {"id": "p003", "text": "GPT models are trained autoregressively to predict the next token.",
     "source": "arxiv", "year": 2020, "topic": "llm", "citations": 12000},
    {"id": "p004", "text": "LoRA adapts large language models by injecting low-rank matrices.",
     "source": "arxiv", "year": 2022, "topic": "finetuning", "citations": 8000},
    {"id": "p005", "text": "Retrieval augmented generation grounds LLM outputs in retrieved documents.",
     "source": "arxiv", "year": 2020, "topic": "rag", "citations": 5000},
    {"id": "p006", "text": "Vector databases enable efficient similarity search over dense embeddings.",
     "source": "blog", "year": 2023, "topic": "databases", "citations": 0},
    {"id": "p007", "text": "Convolutional neural networks extract hierarchical visual features.",
     "source": "arxiv", "year": 2015, "topic": "cv", "citations": 60000},
    {"id": "p008", "text": "Gradient descent optimizes neural network weights through backpropagation.",
     "source": "textbook", "year": 2016, "topic": "optimization", "citations": 0},
    {"id": "p009", "text": "Diffusion models generate images by reversing a noising process.",
     "source": "arxiv", "year": 2021, "topic": "generative", "citations": 15000},
    {"id": "p010", "text": "Reinforcement learning from human feedback aligns LLMs with user intent.",
     "source": "arxiv", "year": 2022, "topic": "rlhf", "citations": 9000},
    # Code-related
    {"id": "c001", "text": "Python list comprehensions provide concise syntax for list operations.",
     "source": "docs", "year": 2020, "topic": "programming", "citations": 0},
    {"id": "c002", "text": "NumPy vectorized operations avoid Python loops for array computations.",
     "source": "docs", "year": 2020, "topic": "programming", "citations": 0},
    {"id": "c003", "text": "Flask routes handle HTTP requests and return JSON responses.",
     "source": "docs", "year": 2021, "topic": "backend", "citations": 0},
]


def main():
    if not CHROMA_AVAILABLE:
        return

    section("1. CLIENT SETUP & COLLECTION CREATION")

    # In-memory client (no persistence)
    client = chromadb.Client()

    # Create collection with cosine distance
    collection = client.create_collection(
        name="ml_knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Collection created: {collection.name}")
    print(f"Distance metric: cosine")
    print(f"Available collections: {[c.name for c in client.list_collections()]}")

    section("2. ADDING DOCUMENTS WITH EMBEDDINGS")

    ids       = [d["id"] for d in CORPUS]
    texts     = [d["text"] for d in CORPUS]
    metadatas = [{"source": d["source"], "year": d["year"],
                  "topic": d["topic"], "citations": d["citations"]}
                 for d in CORPUS]

    t0 = time.perf_counter()
    embeddings = fake_embed(texts, dim=128)
    embed_time = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    add_time = (time.perf_counter() - t1) * 1000

    print(f"Embedded {len(texts)} docs: {embed_time:.1f}ms")
    print(f"Indexed  {len(texts)} docs: {add_time:.1f}ms")
    print(f"Collection count: {collection.count()}")

    section("3. BASIC SIMILARITY QUERY")

    queries = [
        "how do transformers process text sequences",
        "training large language models efficiently",
        "image generation with deep learning",
    ]
    for query_text in queries:
        q_embed = fake_embed([query_text], dim=128)
        results = collection.query(
            query_embeddings=q_embed,
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        print(f"\nQuery: '{query_text}'")
        print(f"  {'ID':>6}  {'Dist':>8}  {'Topic':>12}  {'Text snippet'}")
        print("  " + "-" * 72)
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            print(f"  {doc_id:>6}  {dist:>8.4f}  {meta['topic']:>12}  {doc[:55]}...")

    section("4. METADATA FILTERING")

    # $eq filter: only arxiv sources
    results_arxiv = collection.query(
        query_embeddings=fake_embed(["neural network training"], dim=128),
        n_results=5,
        where={"source": {"$eq": "arxiv"}},
        include=["documents", "metadatas", "distances"],
    )
    print("Filter: source == 'arxiv'")
    for doc_id, doc, meta, dist in zip(
        results_arxiv["ids"][0],
        results_arxiv["documents"][0],
        results_arxiv["metadatas"][0],
        results_arxiv["distances"][0],
    ):
        print(f"  [{doc_id}] dist={dist:.4f} | {doc[:55]}...")

    # $gte filter: citations >= 10000
    results_cited = collection.query(
        query_embeddings=fake_embed(["language model pretraining"], dim=128),
        n_results=5,
        where={"citations": {"$gte": 10000}},
        include=["documents", "metadatas", "distances"],
    )
    print(f"\nFilter: citations >= 10000")
    for doc_id, meta, dist in zip(
        results_cited["ids"][0],
        results_cited["metadatas"][0],
        results_cited["distances"][0],
    ):
        print(f"  [{doc_id}] dist={dist:.4f} | citations={meta['citations']:>6} | topic={meta['topic']}")

    # $and: arxiv + year >= 2020
    results_and = collection.query(
        query_embeddings=fake_embed(["large language models"], dim=128),
        n_results=4,
        where={"$and": [
            {"source": {"$eq": "arxiv"}},
            {"year": {"$gte": 2020}},
        ]},
        include=["documents", "metadatas", "distances"],
    )
    print(f"\nFilter: source=='arxiv' AND year>=2020")
    for doc_id, meta, dist in zip(
        results_and["ids"][0],
        results_and["metadatas"][0],
        results_and["distances"][0],
    ):
        print(f"  [{doc_id}] dist={dist:.4f} | {meta['year']} | {meta['topic']}")

    # where_document: content filter
    results_content = collection.query(
        query_embeddings=fake_embed(["attention and transformers"], dim=128),
        n_results=3,
        where_document={"$contains": "transformer"},
        include=["documents", "distances"],
    )
    print(f"\nwhere_document contains 'transformer':")
    for doc_id, doc, dist in zip(
        results_content["ids"][0],
        results_content["documents"][0],
        results_content["distances"][0],
    ):
        print(f"  [{doc_id}] dist={dist:.4f} | {doc[:60]}...")

    section("5. GET, UPDATE, DELETE")

    # get by ID
    fetched = collection.get(ids=["p001", "p002"], include=["documents", "metadatas"])
    print("get(['p001', 'p002']):")
    for doc_id, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"]):
        print(f"  {doc_id}: {doc[:50]}... | year={meta['year']}")

    # update metadata
    collection.update(
        ids=["p001"],
        metadatas=[{"source": "arxiv", "year": 2017, "topic": "transformers",
                    "citations": 82000}]  # updated citation count
    )
    updated = collection.get(ids=["p001"], include=["metadatas"])
    print(f"\nAfter update p001 citations: {updated['metadatas'][0]['citations']}")

    # upsert — insert or update
    new_doc_text = "Chain-of-thought prompting improves reasoning in large language models."
    collection.upsert(
        ids=["p011"],
        embeddings=fake_embed([new_doc_text], dim=128),
        documents=[new_doc_text],
        metadatas=[{"source": "arxiv", "year": 2022, "topic": "prompting", "citations": 3000}],
    )
    print(f"After upsert: collection count = {collection.count()}")

    # delete
    collection.delete(ids=["c001", "c002"])
    print(f"After delete 2 docs: count = {collection.count()}")

    # delete by metadata filter
    collection.delete(where={"topic": {"$eq": "backend"}})
    print(f"After delete topic==backend: count = {collection.count()}")

    section("6. DISTANCE METRIC COMPARISON")
    print("Creating collections with different distance metrics:")

    sample_vecs = fake_embed(["test doc one", "test doc two", "query vector"], dim=64)
    doc_vecs, query_vec = sample_vecs[:2], sample_vecs[2]

    for metric, space in [("Cosine", "cosine"), ("L2", "l2"), ("Inner Product", "ip")]:
        coll = client.create_collection(f"metric_{space}", metadata={"hnsw:space": space})
        coll.add(
            ids=["d1", "d2"],
            embeddings=doc_vecs,
            documents=["doc one", "doc two"],
        )
        res = coll.query(query_embeddings=[query_vec], n_results=2,
                         include=["distances"])
        dists = res["distances"][0]
        ids_res = res["ids"][0]
        print(f"  {metric:14s}: top1={ids_res[0]}({dists[0]:.4f}), top2={ids_res[1]}({dists[1]:.4f})")

    section("7. BATCH OPERATIONS & PERFORMANCE")

    perf_client = chromadb.Client()
    perf_col = perf_client.create_collection("perf_test", metadata={"hnsw:space": "cosine"})

    for batch_size in [10, 100, 500]:
        vecs = rng_g.standard_normal((batch_size, 128))
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        t0 = time.perf_counter()
        perf_col.add(
            ids=[f"b{batch_size}_{i}" for i in range(batch_size)],
            embeddings=vecs.tolist(),
            documents=[f"document {i}" for i in range(batch_size)],
        )
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  batch_add({batch_size:4d} docs): {elapsed:7.1f}ms  ({elapsed/batch_size:.2f}ms/doc)")

    # Query latency at scale
    q_vec = rng_g.standard_normal(128)
    q_vec /= np.linalg.norm(q_vec)

    n_queries = 100
    t0 = time.perf_counter()
    for _ in range(n_queries):
        perf_col.query(query_embeddings=[q_vec.tolist()], n_results=5)
    avg_lat = (time.perf_counter() - t0) * 1000 / n_queries
    print(f"\n  Query latency ({perf_col.count()} docs, n_results=5): {avg_lat:.2f}ms avg over {n_queries} queries")

    section("8. COLLECTION MANAGEMENT")
    print(f"All collections: {[c.name for c in client.list_collections()]}")
    client.delete_collection("metric_cosine")
    client.delete_collection("metric_l2")
    client.delete_collection("metric_ip")
    print(f"After deleting metric collections: {[c.name for c in client.list_collections()]}")

    # Collection metadata
    meta = collection.metadata
    print(f"\nml_knowledge_base metadata: {meta}")
    print(f"Final document count: {collection.count()}")


if __name__ == "__main__":
    main()
