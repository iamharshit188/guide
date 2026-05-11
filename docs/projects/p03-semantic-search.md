# Project 03 — Semantic Code Search Engine

> **Difficulty:** Intermediate · **Module:** 03 — Databases & Vector DBs
> **Skills:** TF-IDF, FAISS IVFFlat, BM25, hybrid retrieval, latency benchmarking

---

## What You'll Build

A semantic search engine over ~200 Python stdlib docstrings. Two indexes (TF-IDF dense + BM25 sparse), merged with Reciprocal Rank Fusion (RRF), with metadata filtering by module. Includes a latency comparison table across all three retrieval strategies.

---

## Skills Exercised

- TF-IDF from scratch: $\text{IDF}(t) = \log\frac{N+1}{df_t+1}+1$
- FAISS IVFFlat: `nlist` Voronoi cells, `nprobe` search breadth
- BM25: $\text{score}(q,d) = \sum_{t\in q} \text{IDF}(t) \cdot \frac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b\frac{|d|}{\text{avgdl}})}$, $k_1=1.5$, $b=0.75$
- RRF: $\text{score}(d) = \sum_r \frac{1}{k+\text{rank}_r(d)}$, $k=60$
- Metadata filtering: restrict results to a specific stdlib module

---

## Approach

### Phase 1 — Build corpus
```
import inspect, pkgutil
for stdlib module in [os, sys, json, re, pathlib, collections, itertools, functools,
                      math, statistics, datetime, typing]:
    for name, obj in inspect.getmembers(module):
        if callable(obj) and obj.__doc__:
            corpus.append({
                "id": f"{module.__name__}.{name}",
                "text": obj.__doc__[:500],
                "module": module.__name__
            })
# target: ≥150 documents
```

### Phase 2 — TF-IDF index + FAISS
```
tokenize: lowercase, split on non-alphanumeric, remove stopwords
build vocab, compute TF-IDF matrix (n_docs × vocab_size)
reduce to 64 dims with TruncatedSVD (or random projection)
build FAISS IVFFlat(nlist=8, metric=METRIC_L2)
train on embeddings, add all vectors
```

### Phase 3 — BM25 index
```
compute avgdl (average document length in tokens)
for each query:
    for each term in query:
        retrieve documents containing term
        compute BM25 score
    rank by total score
```

### Phase 4 — Hybrid RRF
```
dense_results = faiss_search(query_vec, top_k=20)    # list of (id, rank)
bm25_results  = bm25_search(query_tokens, top_k=20)  # list of (id, rank)
rrf_score[id] = 1/(60+rank_dense) + 1/(60+rank_bm25)
sort by rrf_score descending, return top_k
```

### Phase 5 — Metadata filter + latency table
```
filter: only return results where doc["module"] == requested_module
benchmark 100 random queries × 3 strategies:
    strategy   | p50 (ms) | p95 (ms) | Recall@5
    BM25       | ...      | ...      | ...
    Dense      | ...      | ...      | ...
    Hybrid RRF | ...      | ...      | ...
```

---

## Checkpoints

| Phase | What correct output looks like |
|-------|-------------------------------|
| 1 | `len(corpus) >= 150`; each doc has id, text, module keys |
| 2 | FAISS index trained; `index.ntotal == len(corpus)` |
| 3 | BM25 query "read file path" → top result is `pathlib.Path.read_text` or similar |
| 4 | RRF top-1 should match or beat BM25 top-1 for 70%+ of queries |
| 5 | Dense latency < BM25 latency for large vocab; Hybrid recall ≥ both individual |

---

## Extensions

1. **ChromaDB backend** — replace FAISS with a ChromaDB persistent collection; compare CRUD latency vs. FAISS in-memory.
2. **Query expansion** — for each query term, add WordNet synonyms or top-3 similar TF-IDF vocabulary terms; measure recall@5 improvement.
3. **MMR re-ranking** — after retrieval, apply Maximal Marginal Relevance to diversify results when the query returns many near-duplicate docs.

---

## Hints

<details><summary>Hint 1 — Corpus is too small? Use ast</summary>
Parse the CPython source with <code>ast.parse</code> to extract all function/class docstrings from .py files in your Python stdlib path (<code>import sysconfig; sysconfig.get_paths()['stdlib']</code>).
</details>

<details><summary>Hint 2 — FAISS IVFFlat needs training before adding</summary>
<code>index.train(X)</code> before <code>index.add(X)</code>. If ntotal is 0 after add, you forgot to train. nlist should be ≤ sqrt(n_docs).
</details>

<details><summary>Hint 3 — BM25 speed</summary>
Pre-build an inverted index: dict mapping term → list of (doc_id, freq). Query only touches documents that contain at least one query term.
</details>

<details><summary>Hint 4 — RRF k=60 is not a hyperparameter to tune</summary>
k=60 is the standard value from the original RRF paper. It prevents a rank-1 document from dominating. Keep it fixed.
</details>

<details><summary>Hint 5 — Metadata filter placement</summary>
Filter after retrieval (post-filter) for speed, or build a separate FAISS index per module (pre-filter) to avoid recall loss when the module is small.
</details>

---

*Back to [Module 03 — Databases & Vector DBs](../03-databases.md)*
