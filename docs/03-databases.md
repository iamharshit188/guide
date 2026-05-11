# Module 03 — Databases & Vector DBs

> **Runnable code:** `src/03-databases/`
> ```bash
> cd src/03-databases
> python sql_basics.py          # no extra install — uses sqlite3
> python nosql_patterns.py      # no extra install — pure Python simulation
> python chroma_demo.py         # pip install chromadb
> python pinecone_demo.py       # runs in mock mode (no API key needed)
> python faiss_demo.py          # pip install faiss-cpu
> ```

---

## Prerequisites & Overview

**Prerequisites:** Basic Python, some SQL exposure (know what SELECT/WHERE/JOIN mean conceptually). No vector DB or NoSQL experience required.
**Estimated time:** 8–12 hours (5 scripts; SQL and FAISS sections are the deepest)

### Why This Module Matters
AI/ML systems live inside data pipelines. You need SQL for feature retrieval, joins, and window functions in analytics. You need vector databases for semantic search and RAG. Understanding index internals (B+ tree, HNSW, IVF) is what separates engineers who can tune query performance from those who guess.

### Module Map

| Section | What You'll Learn | Used In |
|---------|------------------|---------|
| SQL (1–2) | Joins, indexes, window functions, query planning | Feature stores, analytics, interview SQL rounds |
| NoSQL (3) | CAP theorem, document/KV/graph trade-offs | Caching, session stores, ML metadata |
| Vector DBs — Math (4) | ANN metrics: cosine, L2, IP | Embedding similarity |
| HNSW / IVF / PQ (5–6) | Index algorithms and their parameters | Choosing and tuning a vector DB |
| ChromaDB / Pinecone / FAISS (7) | Production APIs and patterns | RAG (Module 08) |

### Before You Start
- Know Python dictionaries and lists
- Understand what a database table is (rows, columns, primary keys)
- Know what "cosine similarity" means from Module 01 (dot product / norms)

---

## 1. Relational Databases & SQL

### 1.1 Data Model

**Relations (tables):** Named set of tuples. Schema = column names + types.

**Keys:**
- **Primary key (PK):** Uniquely identifies each row. `NOT NULL`, immutable, indexed automatically.
- **Foreign key (FK):** References PK in another table. Enforces referential integrity.
- **Candidate key:** Minimal set of columns that could serve as PK.
- **Composite key:** PK spanning multiple columns.

**Constraints:** `NOT NULL`, `UNIQUE`, `CHECK`, `DEFAULT`, `FOREIGN KEY`.

### 1.2 Joins

Given tables $R$ and $S$ with a join condition $\theta$:

| Join Type | Definition | Rows returned |
|-----------|-----------|--------------|
| `INNER JOIN` | $R \bowtie_\theta S$ | Matching rows only |
| `LEFT JOIN` | $R \text{ ⟕}_\theta S$ | All of R + matched S (NULL if no match) |
| `RIGHT JOIN` | $R \text{ ⟖}_\theta S$ | All of S + matched R |
| `FULL OUTER` | $R \text{ ⟗}_\theta S$ | All rows from both, NULLs for mismatches |
| `CROSS JOIN` | $R \times S$ | Cartesian product — $|R| \times |S|$ rows |
| `SELF JOIN` | $R \bowtie_\theta R$ | Table joined with itself (alias required) |

**Join algorithms (query planner chooses):**
- **Nested Loop Join:** $O(|R| \cdot |S|)$ — small tables or indexed inner
- **Hash Join:** $O(|R| + |S|)$ — large unsorted tables, no useful index
- **Sort-Merge Join:** $O(|R|\log|R| + |S|\log|S|)$ — when both inputs sortable

### 1.3 Indexes

An index is a separate data structure (B+ tree by default in most RDBMS) that maps column values → row locations, enabling $O(\log N)$ lookups instead of $O(N)$ full scans.

**B+ tree index:**
- Internal nodes store keys as routing guides
- Leaf nodes store (key, row_pointer) pairs in sorted order
- Supports: equality `=`, range `<`, `>`, `BETWEEN`, `ORDER BY`, `GROUP BY`
- **Height:** $\lceil \log_{B}(N) \rceil$ where $B$ = branching factor (~100-1000)

**Hash index:** $O(1)$ equality lookup. Does NOT support range queries. Used in hash joins.

**Composite index on `(a, b, c)`:**
- Useful for queries on: `a`, `(a, b)`, `(a, b, c)` — left-prefix rule
- NOT useful for queries on just `b` or `c`
- Column order matters: put equality-predicate columns first, range-predicate last

**When indexes hurt:**
- High write tables (index must be updated on every INSERT/UPDATE/DELETE)
- Very small tables (full scan is faster — index overhead not worth it)
- Low-selectivity columns (e.g., boolean flags)

**Index-only scan:** If query only reads indexed columns, planner can answer entirely from index without touching the table.

### 1.4 Query Execution & EXPLAIN

SQL query execution pipeline:
1. **Parse** → AST
2. **Bind** → resolve table/column names
3. **Optimize** → enumerate join orders, choose access paths, cost model
4. **Execute** → volcano/iterator model (each operator pulls rows from child)

Read `EXPLAIN QUERY PLAN` output:
- `SCAN TABLE` → full table scan ($O(N)$) — add index?
- `SEARCH TABLE USING INDEX` → index scan ($O(\log N)$)
- `SEARCH TABLE USING INTEGER PRIMARY KEY` → rowid lookup ($O(\log N)$)
- `USE TEMP B-TREE FOR ORDER BY` → sort needed — index on ORDER BY column?

**Cost model:** Optimizer estimates rows × page I/Os. Statistics (histogram of column values) must be up-to-date (`ANALYZE` in SQLite/PostgreSQL).

### 1.5 Window Functions

```sql
SELECT
    user_id,
    purchase_amount,
    SUM(purchase_amount) OVER (PARTITION BY user_id ORDER BY date
                               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total,
    RANK() OVER (PARTITION BY category ORDER BY purchase_amount DESC) AS rank_in_category,
    LAG(purchase_amount, 1) OVER (PARTITION BY user_id ORDER BY date) AS prev_purchase
FROM orders;
```

Execute after `WHERE` and `GROUP BY`, before `ORDER BY` and `LIMIT`. Cannot be used in `WHERE`.

### 1.6 CTEs & Subqueries

**CTE (Common Table Expression):** Named temporary result set, improves readability, enables recursion.

```sql
WITH monthly_sales AS (
    SELECT DATE_TRUNC('month', order_date) AS month,
           SUM(amount) AS revenue
    FROM orders GROUP BY 1
),
growth AS (
    SELECT month, revenue,
           LAG(revenue) OVER (ORDER BY month) AS prev_revenue
    FROM monthly_sales
)
SELECT month,
       revenue,
       ROUND((revenue - prev_revenue) / prev_revenue * 100, 2) AS growth_pct
FROM growth;
```

**Recursive CTE:** Traversing hierarchical data (org charts, BOM):
```sql
WITH RECURSIVE tree AS (
    SELECT id, name, parent_id, 0 AS depth FROM employees WHERE parent_id IS NULL
    UNION ALL
    SELECT e.id, e.name, e.parent_id, t.depth + 1
    FROM employees e JOIN tree t ON e.parent_id = t.id
)
SELECT * FROM tree ORDER BY depth;
```

> **Run:** `python src/03-databases/sql_basics.py`

---

## 2. NoSQL Databases

### 2.1 CAP Theorem

**Consistency (C):** All nodes see the same data at the same time.
**Availability (A):** Every request receives a (non-error) response.
**Partition Tolerance (P):** System continues operating despite network partitions.

**CAP theorem:** A distributed system can guarantee at most **two** of {C, A, P} simultaneously.

Since network partitions are a reality in distributed systems, the real tradeoff is:
- **CP systems** (sacrifice availability during partition): HBase, ZooKeeper, MongoDB (with strong consistency)
- **AP systems** (sacrifice consistency, stay available): Cassandra, DynamoDB, CouchDB

**PACELC extension:** Even without partitions, tradeoff exists between latency (L) and consistency (C).

### 2.2 NoSQL Types

| Type | Data Model | Query | Examples | ML Use |
|------|-----------|-------|---------|--------|
| Document | JSON/BSON documents | Rich queries on nested fields | MongoDB, Firestore | Model configs, feature stores |
| Key-Value | Flat key → blob | GET/SET/DELETE by key | Redis, DynamoDB | Caching, session state |
| Column-family | Rows + dynamic columns | Row-key + column range | Cassandra, HBase | Time-series, IoT |
| Graph | Nodes + edges | Traversal queries (Cypher) | Neo4j, TigerGraph | Knowledge graphs, GNN data |
| Time-series | Timestamped measurements | Range + downsample | InfluxDB, TimescaleDB | Model monitoring, metrics |

### 2.3 MongoDB Document Model

```json
{
  "_id": ObjectId("..."),
  "embedding_id": "emb_001",
  "text": "Machine learning is great",
  "metadata": {
    "source": "arxiv",
    "tags": ["ml", "ai"],
    "created_at": ISODate("2024-01-01")
  },
  "vector": [0.1, 0.2, ...]
}
```

**Aggregation pipeline** — each stage transforms the document stream:
```
$match → $group → $project → $sort → $limit → $lookup (join) → $unwind
```

> **Run:** `python src/03-databases/nosql_patterns.py`

---

## 3. Embeddings & Similarity Search

### 3.1 Why Vector DBs Exist

Dense embeddings (from transformers, word2vec, etc.) represent semantic meaning as points in $\mathbb{R}^d$ (typically $d = 384, 768, 1536$).

**Naive similarity search:** Compare query vector against every stored vector — $O(N \cdot d)$. At $N=10^9$, $d=768$: ~$10^{12}$ FLOPS per query → infeasible.

**Approximate Nearest Neighbor (ANN):** Trade exact answers for speed. Goal: find $k$ vectors with highest similarity in $O(\log N)$ or $O(1)$ time with high recall.

### 3.2 Similarity Metrics

**Cosine similarity** (angle-based, magnitude-invariant):
$$\text{sim}(\mathbf{q}, \mathbf{v}) = \frac{\mathbf{q} \cdot \mathbf{v}}{\|\mathbf{q}\|_2 \|\mathbf{v}\|_2}$$

**L2 distance** (Euclidean):
$$d(\mathbf{q}, \mathbf{v}) = \|\mathbf{q} - \mathbf{v}\|_2 = \sqrt{\sum_i (q_i - v_i)^2}$$

**Dot product** (inner product — for normalized vectors, equivalent to cosine):
$$\text{score}(\mathbf{q}, \mathbf{v}) = \mathbf{q} \cdot \mathbf{v} = \sum_i q_i v_i$$

**Relationship:** For L2-normalized vectors ($\|\mathbf{v}\|_2 = 1$):
$$\|\mathbf{q} - \mathbf{v}\|^2 = 2 - 2(\mathbf{q} \cdot \mathbf{v}) \quad \Rightarrow \quad \text{cosine} \equiv \text{dot product} \equiv -\frac{1}{2}\text{L2}^2$$

---

## 4. HNSW — Hierarchical Navigable Small World

### 4.1 Algorithm

HNSW builds a multi-layer graph:
- **Layer 0:** All nodes, dense connections (short-range)
- **Layer 1, 2, ...:** Exponentially fewer nodes, long-range connections (highway)

**Insert node $q$ at layer $\ell$:**
1. Assign max layer $l_q \sim -\ln(\text{Uniform}(0,1)) \cdot \frac{1}{\ln M}$
2. Greedily traverse from entry point top → layer $l_q+1$ to find entry point
3. At each layer $\leq l_q$: find $ef_{construction}$ nearest neighbors, add bidirectional edges (max $M$ per node)

**Query for $k$-NN:**
1. Start at entry point in top layer
2. Greedy descent: at each layer, move to neighbor closest to query
3. At layer 0: beam search with $ef$ candidates, return top $k$

**Complexity:**
- Build: $O(N \cdot \log N \cdot M \cdot ef_{construction})$
- Query: $O(\log N)$ expected hops at each layer

**Parameters:**
| Param | Effect | Default |
|-------|--------|---------|
| `M` | Max connections per node. Higher → better recall, more memory | 16 |
| `ef_construction` | Beam width during build. Higher → better graph quality, slower build | 200 |
| `ef_search` | Beam width during query. Tune recall/speed tradeoff at runtime | 50 |

### 4.2 IVF — Inverted File Index

**Idea:** Cluster vectors into $n_{list}$ Voronoi cells (K-Means). Store inverted lists of vectors per cluster.

**Query:**
1. Find $n_{probe}$ nearest cluster centroids to query
2. Exhaustively search vectors in those $n_{probe}$ lists
3. Return global top-$k$

**Tradeoff:** $n_{probe}$ controls recall vs speed. Higher $n_{probe}$ → higher recall, slower.

**Complexity:** $O(\sqrt{N})$ expected with $n_{probe}=1$; $O(N)$ when $n_{probe} = n_{list}$.

### 4.3 Product Quantization (PQ)

Compress $d$-dim vectors to fixed bytes. Split vector into $M$ sub-vectors of size $d/M$. Quantize each sub-vector to 1 byte (256 centroids per sub-space).

**Storage:** $d$-dim float32 vector (4d bytes) → $M$ bytes. Compression ratio = $4d/M$.

**ADC (Asymmetric Distance Computation):** Precompute distances from query sub-vectors to all codebook entries → fast approximate distances without decoding.

**IVFPQ:** IVF coarse quantization + PQ fine quantization — standard for billion-scale search.

---

## 5. ChromaDB

### 5.1 Architecture

ChromaDB is an open-source embedding database. Storage backends: DuckDB+Parquet (local), ClickHouse (server). Default ANN: HNSW via `hnswlib`.

**Core concepts:**
- **Collection:** Named namespace for vectors + metadata + documents
- **Embedding function:** Callable that maps text → vectors (pluggable)
- **Metadata:** Arbitrary JSON dict per document (filterable)
- **Distance functions:** `l2`, `cosine`, `ip` (inner product)

### 5.2 Key Operations

```python
import chromadb

client = chromadb.Client()                    # in-memory
# client = chromadb.PersistentClient("./db")  # on disk

collection = client.create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"}         # distance metric
)

# Add — IDs must be unique strings
collection.add(
    ids=["doc1", "doc2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    documents=["text of doc1", "text of doc2"],
    metadatas=[{"source": "arxiv", "year": 2024}, {"source": "web"}]
)

# Query
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],
    n_results=5,
    where={"source": {"$eq": "arxiv"}},           # metadata filter
    where_document={"$contains": "neural"},       # document content filter
    include=["documents", "metadatas", "distances"]
)

# Metadata filter operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
# Combine: {"$and": [{"year": {"$gte": 2023}}, {"source": {"$eq": "arxiv"}}]}
```

> **Run:** `python src/03-databases/chroma_demo.py`

---

## 6. Pinecone

### 6.1 Architecture

Fully managed vector database. Serverless and pod-based deployment. Supports hybrid search (sparse + dense).

**Key concepts:**
- **Index:** Named collection of vectors with fixed dimension and metric
- **Namespace:** Logical partition within an index (zero-copy isolation)
- **Pod type:** `s1` (storage optimized), `p1` (performance), `p2` (latency)

### 6.2 Key Operations

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-key")
pc.create_index(
    name="my-index",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
index = pc.Index("my-index")

# Upsert (insert or update)
index.upsert(vectors=[
    {"id": "vec1", "values": [0.1, 0.2, ...], "metadata": {"category": "ml"}},
    {"id": "vec2", "values": [0.3, 0.4, ...], "metadata": {"category": "nlp"}},
], namespace="prod")

# Query
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=5,
    namespace="prod",
    filter={"category": {"$eq": "ml"}},
    include_metadata=True,
    include_values=False
)

# Hybrid search (sparse + dense)
results = index.query(
    vector=[0.1, 0.2, ...],                            # dense
    sparse_vector={"indices": [10, 50], "values": [0.8, 0.5]},  # BM25
    top_k=5,
    alpha=0.7                                           # 0=sparse, 1=dense
)
```

> **Run:** `python src/03-databases/pinecone_demo.py` (mock mode — no API key needed)

---

## 7. FAISS

### 7.1 Index Types

| Index | Algorithm | Build | Query | Memory | Use case |
|-------|-----------|-------|-------|--------|----------|
| `IndexFlatL2` | Exact brute-force | $O(N)$ | $O(N \cdot d)$ | $4Nd$ bytes | Small datasets, ground-truth |
| `IndexFlatIP` | Exact brute-force (inner product) | $O(N)$ | $O(N \cdot d)$ | $4Nd$ bytes | Normalized vectors |
| `IndexIVFFlat` | IVF + flat lists | $O(NK)$ train | $O(\sqrt{N})$ | $4Nd$ bytes | Medium datasets |
| `IndexHNSWFlat` | HNSW graph | $O(N \log N)$ | $O(\log N)$ | $4Nd + graph$ | Low-latency ANN |
| `IndexPQ` | Product quantization | $O(NKM)$ | $O(N/M)$ | $NM$ bytes | Memory-limited |
| `IndexIVFPQ` | IVF + PQ | Slowest | Fastest | Smallest | Billion-scale |

### 7.2 GPU Support

```python
import faiss
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
```

### 7.3 Recall vs Speed Tradeoff

```
        Recall
          ↑
    1.0 ─ ●──● FlatL2 (exact)
    0.9 ─    ●──● HNSW (ef=200)
    0.8 ─       ●──● HNSW (ef=50)
    0.7 ─          ●──● IVFFlat (nprobe=32)
          └────────────────────→ QPS (queries/sec)
```

> **Run:** `python src/03-databases/faiss_demo.py`

---

## 8. Choosing a Vector DB

| Factor | ChromaDB | Pinecone | FAISS | Weaviate | Qdrant |
|--------|----------|----------|-------|---------|--------|
| Hosting | Local/self-hosted | Fully managed | Library | Self-hosted | Self-hosted |
| Scale | Millions | Billions | Depends | Millions | Millions |
| Hybrid search | No | Yes | No (manual) | Yes | Yes |
| Metadata filtering | Yes | Yes | Manual | Yes | Yes |
| Best for | Local RAG dev | Production RAG | Research/custom | Production | Production |

---

## 9. Redis — Caching Layer for ML Systems

Redis is an in-memory key-value store used as a caching layer between ML services and backends. Common patterns: embedding cache (avoid re-encoding the same query), prediction cache (memoize expensive model calls), session store for RAG conversation history.

### 9.1 Key Patterns

**Embedding cache:** Hash the query string → store (embedding vector, TTL) as Redis key. Subsequent identical queries skip the model call.

```python
import hashlib, json, numpy as np

def get_or_compute_embedding(query: str, model, redis_client, ttl: int = 3600):
    key = "emb:" + hashlib.sha256(query.encode()).hexdigest()[:16]
    cached = redis_client.get(key)
    if cached:
        return np.frombuffer(cached, dtype=np.float32)
    emb = model.encode(query).astype(np.float32)
    redis_client.setex(key, ttl, emb.tobytes())
    return emb
```

**LRU cache with max memory:** Redis evicts least-recently-used keys when `maxmemory` is reached.

```
# redis.conf (or via CONFIG SET)
maxmemory 2gb
maxmemory-policy allkeys-lru
```

### 9.2 Pipeline Batching

Every `redis.get()` is a round-trip (~0.1ms on localhost, ~1ms on network). For bulk ops, pipeline batches multiple commands into one round-trip:

```python
pipe = redis_client.pipeline()
for key in keys:
    pipe.get(key)
results = pipe.execute()   # one round-trip for N commands
```

### 9.3 Connection Pooling

```python
import redis
pool = redis.ConnectionPool(host="localhost", port=6379, db=0,
                            max_connections=20, decode_responses=False)
client = redis.Redis(connection_pool=pool)
```

Never create a new connection per request — connection pools amortize TCP handshake overhead.

### 9.4 Interview Trade-offs

| Property | Detail |
|----------|--------|
| Persistence | RDB snapshots (periodic) + AOF (append-only log); both optional |
| Data structures | Strings, Lists, Sets, Sorted Sets, Hashes, HyperLogLog, Streams |
| Cache invalidation | TTL-based; or explicit `DEL`/`UNLINK`; or keyspace notifications |
| Consistency | Single-threaded command execution → no race conditions on atomic ops |
| Cluster scaling | Hash slots (16384); shards 0-5460, 5461-10922, 10923-16383 |
| vs. Memcached | Redis has persistence, richer data types, pub/sub, Lua scripting |

---

## Interview Reference — Databases & Vector DBs

### Q: What is a B+ tree index and why does SQL use it?

B+ trees keep all data in leaf nodes (connected as a linked list) and use internal nodes only for routing. This enables both equality lookup ($O(\log n)$) and range scans ($O(\log n + k)$ for $k$ results — just traverse the leaf linked list). B+ tree depth stays $\leq 4$ for billions of rows with branching factor ~200.

### Q: What does EXPLAIN tell you and which metrics matter?

`EXPLAIN ANALYZE` shows actual vs. estimated row counts, execution time, and access methods per node. Key signals: **Seq Scan** on a large table (missing index), **Nested Loop** with large outer set (O(n²) risk), large row count estimation errors (stale statistics → run `ANALYZE`). Cost units are arbitrary — compare relative values, not absolute.

### Q: What is the CAP theorem?

A distributed system can guarantee at most two of: **Consistency** (every read sees the latest write), **Availability** (every request gets a response), **Partition tolerance** (system works despite network partition). Since partitions are unavoidable in distributed networks, the real choice is CP (prefer consistency, may refuse requests) vs. AP (prefer availability, may return stale data). Cassandra = AP; HBase = CP; DynamoDB = tunable.

### Q: How does HNSW handle approximate nearest neighbor search?

HNSW builds a multi-layer graph. Layer 0 has all nodes with short-range edges; higher layers have fewer nodes with long-range edges. Search: enter at highest layer, greedily traverse toward query, descend to lower layers, repeat. This approximates a skip list in high dimensions. Time: $O(\log n)$ per query; Space: $O(n \log n)$. Approximate because greedy search may not find the true global optimum.

### Q: What is the difference between dense and sparse retrieval?

Dense: embed query and documents into $\mathbb{R}^d$; compute cosine similarity. Captures semantic meaning; fails on rare terms (out-of-vocab). Sparse: BM25 or TF-IDF; exact term matching; fast with inverted index; fails on synonyms. Hybrid = weighted sum of both scores — best of both worlds. Most production RAG systems use hybrid search.

### Q: What is product quantization in FAISS?

PQ splits the $d$-dimensional vector into $M$ sub-vectors of $d/M$ dimensions each. Each sub-vector is quantized to one of $K=256$ centroids. Result: each vector stored as $M$ bytes instead of $d \times 4$ bytes → $4d/M$ compression. Distance computed as sum of pre-computed sub-distance table lookups → fast. IVF-PQ = coarse clustering (IVF) + fine quantization (PQ).

### Q: What is a TTL and how does cache invalidation work in Redis?

TTL (time-to-live): Redis automatically deletes a key after `N` seconds. Set via `SETEX key ttl value` or `EXPIRE key ttl`. Cache invalidation strategies: (1) TTL only — simplest, accepts staleness up to TTL. (2) Write-through — update cache on every DB write. (3) Cache-aside — read cache first; on miss, load from DB and populate cache. (4) Write-behind — write to cache, async flush to DB.

---

## Cheat Sheet — Databases & Vector DBs

| Concept | Key Fact |
|---------|----------|
| B+ tree | $O(\log n)$ lookup + range scan; leaf linked list |
| Composite index | Leftmost prefix rule: index (a,b,c) helps queries on a, (a,b), (a,b,c) but not b alone |
| EXPLAIN ANALYZE | Seq Scan = no index used; Hash Join = large unsorted sets |
| CAP theorem | Pick 2: Consistency / Availability / Partition tolerance |
| Eventual consistency | Writes propagate over time; reads may see stale data |
| Cosine similarity | $\cos(\mathbf{a},\mathbf{b}) = \mathbf{a}\cdot\mathbf{b}$ for unit vectors |
| HNSW | Multi-layer graph; $O(\log n)$ approx-ANN; tuned by `ef_construction`, `M` |
| IVF | K-means clusters; search only `nprobe` nearest clusters |
| PQ compression | Split $d$-dim into $M$ sub-vectors × 256 centroids = $M$ bytes/vector |
| ChromaDB | Local dev RAG; simple Python API; default HNSW |
| Pinecone | Managed; sparse-dense hybrid; namespaces for multi-tenancy |
| FAISS IndexFlatL2 | Exact brute-force; reference baseline |
| Redis LRU | `maxmemory-policy allkeys-lru`; evicts least-recently-used on OOM |
| Redis pipeline | Batch N commands → 1 round-trip |
| TTL | `SETEX key seconds value`; auto-delete after expiry |

---

## Resources

### SQL
- **Use The Index, Luke** (`use-the-index-luke.com`): The best free resource for SQL index internals — B+ tree mechanics, composite indexes, query planning. Language-agnostic.
- **Mode SQL Tutorial** (`mode.com/sql-tutorial`): Free, browser-based SQL practice with real datasets. Good for window functions.
- **SQLite documentation** (`sqlite.org/docs.html`): The engine used in `sql_basics.py`. Lightweight, zero-install, covers all standard SQL.

### NoSQL & Distributed Systems
- **Designing Data-Intensive Applications** — Martin Kleppmann: Best book on CAP theorem, replication, consistency models. Chapter 2 covers data models; Chapter 5-9 cover distributed systems.
- **MongoDB University** (`learn.mongodb.com`): Free courses on the aggregation pipeline.

### Vector Search — Papers
- **Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs** — Malkov & Yashunin (2018): HNSW paper. `arxiv.org/abs/1603.09320`
- **Billion-Scale Similarity Search with GPUs** — Johnson, Douze, Jégou (2017): FAISS paper. `arxiv.org/abs/1702.08734`
- **Product Quantization for Nearest Neighbor Search** — Jégou et al. (2011): Foundational PQ paper.

### Tooling
- FAISS wiki (`github.com/facebookresearch/faiss/wiki`): Index types, guidelines, benchmarks.
- ChromaDB docs (`docs.trychroma.com`): API reference for the library used in Module 08.
- Pinecone learning center (`docs.pinecone.io/guides/get-started/overview`): Architecture overview and filtering guide.

---

*Next: [Module 04 — Backend with Flask](04-backend.md)*
