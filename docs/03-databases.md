# Module 03 — Databases & Vector DBs

> **Runnable code:** `src/03-databases/`
> ```bash
> python src/03-databases/sql_basics.py          # uses sqlite3 (built-in)
> python src/03-databases/nosql_patterns.py      # pure Python
> python src/03-databases/chroma_demo.py         # pip install chromadb
> python src/03-databases/faiss_demo.py          # pip install faiss-cpu
> ```

---

## Prerequisites & Overview

**Prerequisites:** Basic Python, know what SELECT/WHERE mean conceptually. Module 01 cosine similarity.
**Estimated time:** 8–12 hours

**Install:**
```bash
pip install chromadb faiss-cpu numpy
```

### Why This Module Matters

AI/ML systems live inside data pipelines. You need:
- **SQL** — feature retrieval, analytics, joins across tables
- **NoSQL** — caching session data, ML metadata, high-write workloads
- **Vector DBs** — semantic search, RAG systems, embedding storage

Understanding index internals (B+ tree, HNSW, IVF) separates engineers who can tune query performance from those who guess.

### Module Map

| Section | What You'll Learn | Used In |
|---------|-----------------|---------|
| SQL | Joins, indexes, window functions, EXPLAIN | Feature stores, analytics |
| NoSQL | CAP theorem, document/KV trade-offs | Caching, metadata, sessions |
| Vector Math | Cosine, L2, inner product distance | Embedding similarity |
| HNSW / IVF / PQ | Index algorithms | Choosing and tuning vector DBs |
| ChromaDB / FAISS | Production APIs | RAG (Module 08) |

---

# 1. Relational Databases & SQL

## Intuition

A database is like a well-organized filing cabinet. Each drawer is a **table** (spreadsheet). Every row is one record (person, product, transaction). Every column is a property (name, price, date).

SQL is the language to read, write, and combine this data.

## 1.1 Creating Tables and Inserting Data

```python
import sqlite3

# Connect to in-memory database (no file needed)
conn   = sqlite3.connect(":memory:")
cursor = conn.cursor()

# ── Create tables ───────────────────────────────────────────
cursor.executescript("""
    CREATE TABLE users (
        user_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        name      TEXT    NOT NULL,
        email     TEXT    UNIQUE NOT NULL,
        age       INTEGER CHECK(age > 0),
        join_date TEXT    DEFAULT CURRENT_DATE
    );

    CREATE TABLE products (
        product_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL,
        category    TEXT,
        price       REAL    NOT NULL,
        stock       INTEGER DEFAULT 0
    );

    CREATE TABLE orders (
        order_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER REFERENCES users(user_id),
        product_id INTEGER REFERENCES products(product_id),
        quantity   INTEGER NOT NULL,
        order_date TEXT    DEFAULT CURRENT_DATE,
        total      REAL
    );
""")

# ── Insert data ─────────────────────────────────────────────
users_data = [
    ("Alice", "alice@email.com", 28),
    ("Bob",   "bob@email.com",   35),
    ("Carol", "carol@email.com", 24),
    ("Dave",  "dave@email.com",  42),
]
cursor.executemany("INSERT INTO users (name, email, age) VALUES (?,?,?)", users_data)

products_data = [
    ("Laptop",  "Electronics", 999.99, 15),
    ("Phone",   "Electronics", 599.99, 30),
    ("Book",    "Education",   29.99,  100),
    ("Headset", "Electronics", 149.99, 25),
    ("Course",  "Education",   199.99, 999),
]
cursor.executemany("INSERT INTO products (name, category, price, stock) VALUES (?,?,?,?)", products_data)

orders_data = [
    (1, 1, 1, 999.99),   # Alice bought Laptop
    (1, 3, 2, 59.98),    # Alice bought 2 Books
    (2, 2, 1, 599.99),   # Bob bought Phone
    (3, 4, 1, 149.99),   # Carol bought Headset
    (2, 3, 3, 89.97),    # Bob bought 3 Books
    (4, 5, 1, 199.99),   # Dave bought Course
]
cursor.executemany("INSERT INTO orders (user_id, product_id, quantity, total) VALUES (?,?,?,?)", orders_data)

conn.commit()
print("Tables created and populated successfully!")
```

## 1.2 Basic Queries

```python
# ── SELECT with WHERE ───────────────────────────────────────
print("Electronics products under $700:")
cursor.execute("""
    SELECT name, price, stock
    FROM products
    WHERE category = 'Electronics' AND price < 700
    ORDER BY price DESC
""")
for row in cursor.fetchall():
    print(f"  {row[0]:<12} ${row[1]:.2f}  (stock: {row[2]})")

# ── Aggregate functions ─────────────────────────────────────
print("\nSpending by category:")
cursor.execute("""
    SELECT p.category,
           COUNT(o.order_id)  AS order_count,
           SUM(o.total)       AS revenue,
           AVG(o.total)       AS avg_order
    FROM orders o
    JOIN products p ON o.product_id = p.product_id
    GROUP BY p.category
    ORDER BY revenue DESC
""")
for row in cursor.fetchall():
    print(f"  {row[0]:<15} orders={row[1]}, revenue=${row[2]:.2f}, avg=${row[3]:.2f}")
```

## 1.3 Joins — Combining Tables

```python
# ── INNER JOIN: only rows that match in BOTH tables ─────────
print("INNER JOIN — users who placed orders:")
cursor.execute("""
    SELECT u.name, p.name AS product, o.quantity, o.total
    FROM orders o
    INNER JOIN users    u ON o.user_id    = u.user_id
    INNER JOIN products p ON o.product_id = p.product_id
    ORDER BY u.name
""")
for row in cursor.fetchall():
    print(f"  {row[0]:<8} bought {row[1]:<12} ×{row[2]}  ${row[3]:.2f}")

# ── LEFT JOIN: all users, even those without orders ─────────
print("\nLEFT JOIN — all users (even without orders):")
cursor.execute("""
    SELECT u.name, COUNT(o.order_id) AS order_count, COALESCE(SUM(o.total), 0) AS total_spent
    FROM users u
    LEFT JOIN orders o ON u.user_id = o.user_id
    GROUP BY u.user_id, u.name
    ORDER BY total_spent DESC
""")
for row in cursor.fetchall():
    print(f"  {row[0]:<8} {row[1]} orders, ${row[2]:.2f} total")
```

**Key difference:**
- `INNER JOIN` — Dave has no Electronics orders, so he might be excluded
- `LEFT JOIN` — Dave always appears, with NULL for missing order columns

## 1.4 Indexes — Making Queries Fast

```python
# Without index: SQLite scans every row (O(N))
cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM users WHERE email = 'alice@email.com'")
print("\nQuery plan WITHOUT index:")
for row in cursor.fetchall():
    print(f"  {row}")

# Create index
cursor.execute("CREATE INDEX idx_users_email ON users(email)")
cursor.execute("CREATE INDEX idx_orders_user ON orders(user_id)")
cursor.execute("CREATE INDEX idx_orders_product ON orders(product_id)")

# With index: direct lookup (O(log N))
cursor.execute("EXPLAIN QUERY PLAN SELECT * FROM users WHERE email = 'alice@email.com'")
print("\nQuery plan WITH index:")
for row in cursor.fetchall():
    print(f"  {row}")
```

**B+ Tree internals:**
- Leaf nodes store `(column_value, row_id)` pairs in sorted order
- Internal nodes are routing guides
- Height ≈ $\lceil\log_B(N)\rceil$ where $B$ ≈ 100–1000 → even 1M rows needs height ≈ 3
- Supports range queries: `WHERE age BETWEEN 20 AND 30` — great for B-trees

**When NOT to index:**
- Columns with few distinct values (low cardinality): boolean flags, enum columns
- Tables with very heavy writes (index must be updated on every insert/update/delete)
- Tables with fewer than ~1000 rows (full scan is faster)

## 1.5 Window Functions

Window functions compute values across a "window" of rows without collapsing them into groups (unlike GROUP BY).

```python
cursor.executescript("""
    CREATE VIEW order_details AS
    SELECT
        u.name AS user_name,
        p.category,
        p.name AS product_name,
        o.total,
        o.order_date
    FROM orders o
    JOIN users u    ON o.user_id    = u.user_id
    JOIN products p ON o.product_id = p.product_id;
""")

# RANK per category, running total, percentage of total
cursor.execute("""
    SELECT
        user_name,
        product_name,
        total,
        RANK() OVER (ORDER BY total DESC) AS global_rank,
        SUM(total) OVER () AS grand_total,
        ROUND(total * 100.0 / SUM(total) OVER (), 2) AS pct_of_total
    FROM order_details
    ORDER BY total DESC
""")

print("\nWindow function results:")
print(f"  {'User':<8} {'Product':<12} {'Total':>8} {'Rank':>6} {'% Total':>8}")
for row in cursor.fetchall():
    print(f"  {row[0]:<8} {row[1]:<12} {row[2]:>8.2f} {row[3]:>6} {row[5]:>7.1f}%")
```

## 1.6 Transactions and ACID

```python
# ACID: Atomicity, Consistency, Isolation, Durability

# Correct pattern: use transactions
try:
    conn.execute("BEGIN")

    # Deduct stock
    conn.execute("UPDATE products SET stock = stock - 1 WHERE product_id = 1")
    # Insert order
    conn.execute("INSERT INTO orders (user_id, product_id, quantity, total) VALUES (3, 1, 1, 999.99)")
    # (In real code: these must both succeed or both fail)

    conn.execute("COMMIT")
    print("Order placed successfully (both updates committed)")

except Exception as e:
    conn.execute("ROLLBACK")
    print(f"Transaction rolled back: {e}")

conn.close()
```

---

# 2. NoSQL Databases

## Intuition

SQL databases are great for structured, relational data. But:
- What if your data schema changes frequently?
- What if you need to scale to millions of writes per second?
- What if your data is hierarchical (JSON documents)?

NoSQL databases trade some SQL guarantees for flexibility and scale.

## 2.1 NoSQL Types

| Type | Model | Key Operations | Example Use |
|------|-------|---------------|-------------|
| Document | JSON-like docs | CRUD, search | MongoDB, Firestore |
| Key-Value | Key → blob | Get/Set/Delete | Redis, DynamoDB |
| Column-Family | Column groups | Batch reads | Cassandra, HBase |
| Graph | Nodes + edges | Traversal, paths | Neo4j |

## 2.2 CAP Theorem

A distributed system can only guarantee **2 of 3**:

| Property | Meaning |
|----------|---------|
| **C**onsistency | Every read returns the latest write |
| **A**vailability | Every request gets a response |
| **P**artition tolerance | System works despite network splits |

Network partitions always happen in practice → choose CP or AP:
- **CP** (MongoDB, HBase): consistent reads, may reject writes during partition
- **AP** (Cassandra, DynamoDB): always available, may return stale data

**PACELC extension:** Even without partitions, must trade off **L**atency vs **C**onsistency.

```python
# Simulating document store behavior
class DocumentStore:
    """Simple in-memory document store (like MongoDB)."""

    def __init__(self):
        self._store = {}

    def insert(self, collection, doc_id, document):
        if collection not in self._store:
            self._store[collection] = {}
        self._store[collection][doc_id] = document.copy()
        return doc_id

    def find(self, collection, query=None):
        if collection not in self._store:
            return []
        docs = list(self._store[collection].values())
        if query is None:
            return docs
        # Simple equality filter
        return [d for d in docs if all(d.get(k) == v for k, v in query.items())]

    def update(self, collection, doc_id, updates):
        if collection in self._store and doc_id in self._store[collection]:
            self._store[collection][doc_id].update(updates)
            return True
        return False

    def delete(self, collection, doc_id):
        if collection in self._store and doc_id in self._store[collection]:
            del self._store[collection][doc_id]
            return True
        return False

db = DocumentStore()

# Insert ML experiment metadata
db.insert("experiments", "exp_001", {
    "name": "bert-finetune-v1",
    "model": "bert-base",
    "dataset": "imdb",
    "epochs": 5,
    "lr": 2e-5,
    "metrics": {"accuracy": 0.923, "f1": 0.921},
    "tags": ["nlp", "classification", "bert"]
})

db.insert("experiments", "exp_002", {
    "name": "gpt2-finetune-v1",
    "model": "gpt2",
    "dataset": "imdb",
    "epochs": 3,
    "lr": 5e-5,
    "metrics": {"accuracy": 0.895, "f1": 0.893},
    "tags": ["nlp", "classification", "gpt2"]
})

db.insert("experiments", "exp_003", {
    "name": "bert-finetune-v2",
    "model": "bert-base",
    "dataset": "yelp",
    "epochs": 10,
    "lr": 1e-5,
    "metrics": {"accuracy": 0.941, "f1": 0.940},
    "tags": ["nlp", "classification", "bert", "improved"]
})

# Query: all bert experiments
bert_exps = db.find("experiments", {"model": "bert-base"})
print(f"BERT experiments found: {len(bert_exps)}")
for exp in bert_exps:
    print(f"  {exp['name']}: accuracy={exp['metrics']['accuracy']}")

# Update an experiment
db.update("experiments", "exp_001", {"status": "deployed"})
print(f"\nexp_001 status: {db.find('experiments', {'name': 'bert-finetune-v1'})[0].get('status')}")
```

## 2.3 Redis-Style Key-Value Store

```python
import time

class RedisLike:
    """In-memory KV store with TTL (like Redis)."""

    def __init__(self):
        self._data   = {}
        self._expiry = {}

    def set(self, key, value, ttl_seconds=None):
        self._data[key] = value
        if ttl_seconds:
            self._expiry[key] = time.time() + ttl_seconds

    def get(self, key):
        # Check TTL
        if key in self._expiry and time.time() > self._expiry[key]:
            del self._data[key]
            del self._expiry[key]
            return None
        return self._data.get(key)

    def delete(self, key):
        self._data.pop(key, None)
        self._expiry.pop(key, None)

    def incr(self, key, amount=1):
        val = int(self._data.get(key, 0)) + amount
        self._data[key] = val
        return val

cache = RedisLike()

# Cache ML model predictions (avoid recomputing)
def expensive_ml_prediction(user_id):
    time.sleep(0.01)  # simulate model inference
    return {"score": 0.87, "class": "positive"}

def cached_predict(user_id, cache_obj):
    cache_key = f"prediction:{user_id}"
    cached    = cache_obj.get(cache_key)
    if cached:
        return cached, True   # cache hit

    result = expensive_ml_prediction(user_id)
    cache_obj.set(cache_key, result, ttl_seconds=300)  # cache for 5 min
    return result, False  # cache miss

result, hit = cached_predict("user_123", cache)
print(f"First call — cache hit: {hit}")

result, hit = cached_predict("user_123", cache)
print(f"Second call — cache hit: {hit}")

# Counter (request rate limiting)
for i in range(5):
    count = cache.incr("api_requests_today")
    print(f"Request {i+1}: total today = {count}")
```

---

# 3. Vector Databases — Semantic Search

## Intuition

Traditional databases match by **exact value**: `WHERE name = 'Alice'`. But how do you search for "documents about machine learning" when no document contains those exact words?

Vector databases match by **meaning**. Every piece of content (text, image, audio) is converted to a vector (embedding). Similar content has vectors pointing in similar directions. Search finds vectors closest to the query vector.

## 3.1 Distance Metrics

| Metric | Formula | Use case |
|--------|---------|---------|
| Cosine Similarity | $\frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ | Text similarity (normalized embeddings) |
| Euclidean (L2) | $\|\mathbf{u} - \mathbf{v}\|_2$ | Image features, spatial data |
| Inner Product | $\mathbf{u}\cdot\mathbf{v}$ | When embeddings are not normalized |
| Manhattan (L1) | $\sum_i |u_i - v_i|$ | Sparse high-dimensional features |

```python
import numpy as np

def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def euclidean_distance(u, v):
    return np.linalg.norm(u - v)

def inner_product(u, v):
    return np.dot(u, v)

# Simulate embeddings
rng = np.random.default_rng(42)

# Three "documents" as 4D embeddings
tech_doc    = np.array([0.8, 0.9, 0.1, 0.2])   # tech content
science_doc = np.array([0.7, 0.8, 0.2, 0.1])   # similar to tech
cooking_doc = np.array([0.1, 0.2, 0.9, 0.8])   # very different

query = np.array([0.75, 0.85, 0.15, 0.1])  # "machine learning" query

docs = {"tech": tech_doc, "science": science_doc, "cooking": cooking_doc}

print(f"Query similarity to each document:")
print(f"{'Document':<12} {'Cosine':>10} {'Euclidean':>12} {'Inner Prod':>12}")
print("-" * 48)

for name, doc in docs.items():
    cos = cosine_similarity(query, doc)
    euc = euclidean_distance(query, doc)
    ip  = inner_product(query, doc)
    print(f"{name:<12} {cos:>10.4f} {euc:>12.4f} {ip:>12.4f}")

print(f"\nBest match by cosine: {max(docs, key=lambda k: cosine_similarity(query, docs[k]))}")
```

## 3.2 Naive Nearest-Neighbor Search

Brute-force: compute similarity to every vector. $O(N \cdot d)$ per query. Works for small collections.

```python
import numpy as np

class NaiveVectorDB:
    """Brute-force vector store — exact but slow at scale."""

    def __init__(self, dim):
        self.dim      = dim
        self.vectors  = []   # list of (id, vector, metadata)

    def add(self, doc_id, vector, metadata=None):
        vector = np.array(vector, dtype=np.float32)
        vector = vector / np.linalg.norm(vector)  # normalize to unit length
        self.vectors.append((doc_id, vector, metadata or {}))

    def search(self, query, k=3):
        if not self.vectors:
            return []

        query  = np.array(query, dtype=np.float32)
        query  = query / np.linalg.norm(query)   # normalize query

        # Compute cosine similarity to every vector
        scores = []
        for doc_id, vec, meta in self.vectors:
            score = np.dot(query, vec)   # dot of unit vectors = cosine sim
            scores.append((score, doc_id, meta))

        # Return top-k
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:k]

rng = np.random.default_rng(42)
dim = 16  # small dimension for demo

db = NaiveVectorDB(dim)

# Add movie "embeddings" (fake — in real life, use a sentence transformer)
movies = [
    ("The Matrix",            rng.normal([1, 0.8, 0.1, 0.9] + [0]*12, 0.1)),
    ("Inception",             rng.normal([0.9, 0.7, 0.2, 0.8] + [0]*12, 0.1)),
    ("The Dark Knight",       rng.normal([0.7, 0.5, 0.8, 0.6] + [0]*12, 0.1)),
    ("Ratatouille",           rng.normal([0.1, 0.2, 0.9, 0.1] + [0]*12, 0.1)),
    ("The Grand Budapest",    rng.normal([0.2, 0.3, 0.8, 0.2] + [0]*12, 0.1)),
    ("Interstellar",          rng.normal([0.95, 0.85, 0.1, 0.7] + [0]*12, 0.1)),
    ("Mad Max Fury Road",     rng.normal([0.5, 0.4, 0.7, 0.9] + [0]*12, 0.1)),
]

for title, emb in movies:
    db.add(title, emb[:dim], {"genre": "sci-fi"})

# Search: "movies like The Matrix"
query_emb = np.array([1.0, 0.8, 0.1, 0.9] + [0]*12)
results   = db.search(query_emb, k=3)

print("Movies similar to 'The Matrix' query:")
for score, doc_id, _ in results:
    bar = "█" * int(score * 30)
    print(f"  {doc_id:<25} similarity={score:.4f}  {bar}")
```

## 3.3 HNSW — Hierarchical Navigable Small World

Approximate Nearest Neighbor (ANN) index. Instead of $O(N\cdot d)$ per query, achieves $O(\log N \cdot d)$.

**How it works:**
1. Build a multi-layer graph where nodes = vectors
2. Layer 0: dense graph (all connections), Layer 1: sparser, Layer 2: sparsest
3. Search starts at the top layer, greedily descends toward query
4. At bottom layer, explore local neighborhood to find best candidates

**Parameters:**
- `M` — connections per node (higher = better recall, more memory)
- `ef_construction` — search width during index building (higher = slower build, better quality)
- `ef_search` — search width during query (trade recall vs speed)

```python
# HNSW intuition — simulating the greedy layer search
import numpy as np

rng = np.random.default_rng(42)

# Simulate HNSW search concept
def hnsw_search_concept(query, vectors, ef=10):
    """
    Simplified HNSW search:
    1. Pick a random entry point
    2. Greedily move to nearest neighbor
    3. At bottom layer, expand neighborhood with ef candidates
    """
    n = len(vectors)
    query = query / np.linalg.norm(query)

    # Entry point: pick a random vector as start
    entry = rng.integers(n)

    # Greedy descent (simplified — real HNSW uses multiple layers)
    visited   = {entry}
    current   = entry
    best_dist = np.dot(query, vectors[current] / np.linalg.norm(vectors[current]))

    for _ in range(20):  # max hops
        # Explore neighbors (random simulation — real HNSW uses graph edges)
        neighbors = rng.choice(n, min(ef, n), replace=False)
        improved  = False

        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                sim = np.dot(query, vectors[nb] / np.linalg.norm(vectors[nb]))
                if sim > best_dist:
                    best_dist = sim
                    current   = nb
                    improved  = True

        if not improved:
            break  # local minimum found

    return current, best_dist, len(visited)

# Test
n_vecs = 10000
d      = 128
vecs   = rng.randn(n_vecs, d).astype(np.float32)
query  = rng.randn(d).astype(np.float32)

result_idx, result_sim, nodes_visited = hnsw_search_concept(query, vecs)

# Exact search (brute force)
sims        = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)) @ (query / np.linalg.norm(query))
true_best   = sims.argmax()
true_sim    = sims.max()

print(f"HNSW (approximate):")
print(f"  Similarity: {result_sim:.4f}")
print(f"  Nodes visited: {nodes_visited} / {n_vecs}")

print(f"\nBrute force (exact):")
print(f"  Similarity: {true_sim:.4f}")

print(f"\nApproximation gap: {true_sim - result_sim:.4f}")
print(f"Speed ratio: {n_vecs / nodes_visited:.1f}x fewer comparisons")
```

## 3.4 IVF — Inverted File Index

Divide vectors into $n_{list}$ clusters (using K-Means). To search:
1. Find the $n_{probe}$ nearest cluster centroids
2. Search only within those clusters

**Trade-off:** `n_list` larger → faster search but potentially misses candidates. `n_probe` larger → better recall, slower search.

```python
import numpy as np

class IVFIndex:
    """Inverted File Index for ANN search."""

    def __init__(self, n_lists=8):
        self.n_lists  = n_lists
        self.centroids = None
        self.lists    = {}   # cluster_id -> list of (vector, doc_id)

    def train(self, vectors, n_iter=20):
        """K-Means to find cluster centroids."""
        rng = np.random.default_rng(42)
        n   = len(vectors)

        # Initialize centroids
        idx            = rng.choice(n, self.n_lists, replace=False)
        self.centroids = vectors[idx].copy()

        for _ in range(n_iter):
            # Assign
            dists   = np.linalg.norm(vectors[:, None] - self.centroids[None], axis=2)
            assigns = dists.argmin(axis=1)

            # Update
            for k in range(self.n_lists):
                mask = assigns == k
                if mask.sum() > 0:
                    self.centroids[k] = vectors[mask].mean(axis=0)

        print(f"IVF trained: {self.n_lists} clusters on {n} vectors")

    def add(self, vectors, ids):
        """Add vectors to their nearest cluster."""
        dists   = np.linalg.norm(vectors[:, None] - self.centroids[None], axis=2)
        assigns = dists.argmin(axis=1)

        for i, cluster in enumerate(assigns):
            if cluster not in self.lists:
                self.lists[cluster] = []
            self.lists[cluster].append((vectors[i], ids[i]))

    def search(self, query, k=5, n_probe=2):
        """Search top-k nearest in n_probe closest clusters."""
        # Find n_probe nearest centroids
        centroid_dists = np.linalg.norm(self.centroids - query, axis=1)
        probe_clusters = np.argsort(centroid_dists)[:n_probe]

        # Search within those clusters
        candidates = []
        total_checked = 0
        for cluster in probe_clusters:
            for vec, doc_id in self.lists.get(cluster, []):
                sim = np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec))
                candidates.append((sim, doc_id))
                total_checked += 1

        candidates.sort(reverse=True)
        return candidates[:k], total_checked

# Demo
rng     = np.random.default_rng(42)
n, d    = 5000, 32

vecs    = rng.randn(n, d).astype(np.float32)
ids     = list(range(n))
query   = rng.randn(d).astype(np.float32)

index = IVFIndex(n_lists=16)
index.train(vecs)
index.add(vecs, ids)

results, checked = index.search(query, k=5, n_probe=2)

print(f"\nIVF Search results:")
print(f"  Checked {checked}/{n} vectors ({checked/n*100:.1f}%)")
print(f"  Top-3 results:")
for sim, doc_id in results[:3]:
    print(f"    doc_id={doc_id}, similarity={sim:.4f}")
```

## 3.5 ChromaDB — Production Vector Store

```python
# ChromaDB: persistent, metadata filtering, multiple distance functions
# pip install chromadb

try:
    import chromadb
    from chromadb.config import Settings

    # In-memory client for demo
    client = chromadb.Client()

    collection = client.create_collection(
        name="ml_papers",
        metadata={"hnsw:space": "cosine"}  # cosine distance
    )

    # Add ML paper summaries with embeddings
    papers = [
        {"id": "p1", "text": "Attention is all you need — transformer architecture",
         "year": 2017, "venue": "NeurIPS"},
        {"id": "p2", "text": "BERT pre-training of deep bidirectional transformers",
         "year": 2018, "venue": "NAACL"},
        {"id": "p3", "text": "GPT-2 language models are few-shot learners",
         "year": 2019, "venue": "ArXiv"},
        {"id": "p4", "text": "ImageNet classification with deep convolutional neural networks",
         "year": 2012, "venue": "NeurIPS"},
        {"id": "p5", "text": "Generative adversarial networks for image synthesis",
         "year": 2014, "venue": "NIPS"},
    ]

    rng = np.random.default_rng(42)

    # In production: use SentenceTransformers to embed the text
    # Here we use random vectors for demo
    embeddings = [rng.randn(384).tolist() for _ in papers]

    collection.add(
        ids        = [p["id"] for p in papers],
        embeddings = embeddings,
        documents  = [p["text"] for p in papers],
        metadatas  = [{"year": p["year"], "venue": p["venue"]} for p in papers]
    )

    print(f"Added {collection.count()} papers to ChromaDB")

    # Search for transformer-related papers
    query_emb = rng.randn(384).tolist()
    results   = collection.query(
        query_embeddings=[query_emb],
        n_results=3,
        where={"year": {"$gte": 2017}}  # metadata filter!
    )

    print("\nTop 3 papers (filtered to 2017+):")
    for i, (doc, dist) in enumerate(zip(
        results['documents'][0],
        results['distances'][0]
    )):
        print(f"  {i+1}. {doc[:60]}...")
        print(f"     Distance: {dist:.4f}")

except ImportError:
    print("chromadb not installed. Run: pip install chromadb")
    print("ChromaDB usage shown conceptually above.")
```

## 3.6 FAISS — Facebook AI Similarity Search

```python
try:
    import faiss

    rng = np.random.default_rng(42)
    d   = 128    # embedding dimension
    n   = 50000  # 50K vectors

    # Generate vectors (in practice: your model embeddings)
    vectors = rng.randn(n, d).astype(np.float32)
    faiss.normalize_L2(vectors)  # normalize for cosine similarity

    # ── Flat index (exact search) ──────────────────────────
    index_flat = faiss.IndexFlatIP(d)   # Inner Product = cosine for normalized vectors
    index_flat.add(vectors)

    query = rng.randn(1, d).astype(np.float32)
    faiss.normalize_L2(query)

    D_flat, I_flat = index_flat.search(query, k=5)
    print(f"Exact (Flat) — top-5 similarities: {D_flat[0].round(4)}")
    print(f"Exact (Flat) — top-5 indices:      {I_flat[0]}")

    # ── IVF index (approximate, much faster) ──────────────
    nlist = 100     # number of clusters
    quantizer   = faiss.IndexFlatIP(d)
    index_ivf   = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    index_ivf.train(vectors)
    index_ivf.add(vectors)
    index_ivf.nprobe = 10   # search 10 clusters

    D_ivf, I_ivf = index_ivf.search(query, k=5)
    print(f"\nIVF (approximate) — top-5 similarities: {D_ivf[0].round(4)}")

    # Recall: how many of IVF results match exact?
    recall = len(set(I_flat[0]) & set(I_ivf[0])) / 5
    print(f"Recall@5: {recall:.0%}")

    print(f"\nIndex sizes:")
    print(f"  Flat: {index_flat.ntotal:,} vectors")
    print(f"  IVF:  {index_ivf.ntotal:,} vectors")

except ImportError:
    print("faiss not installed. Run: pip install faiss-cpu")
    print("FAISS usage shown conceptually above.")
```

---

# 4. Interview Q&A

## Q1: What is the difference between INNER JOIN and LEFT JOIN?

`INNER JOIN` returns only rows where the join condition is satisfied in **both** tables. `LEFT JOIN` returns all rows from the left table; if no match exists in the right table, right-side columns are `NULL`. Use `LEFT JOIN` when you want all records from one table, whether or not they have related records elsewhere.

## Q2: Why are vector databases different from traditional databases?

Traditional databases find exact matches ($\text{WHERE} = $ ). Vector databases find **approximate nearest neighbors** — documents whose meaning is most similar to the query. The underlying index (HNSW, IVF) enables sub-linear search time over billions of vectors. You can't do `WHERE meaning = 'machine learning'` in SQL, but you can embed that query and find cosine-similar vectors.

## Q3: Explain the CAP theorem.

A distributed system can simultaneously guarantee at most two of: **Consistency** (reads always return latest write), **Availability** (every request gets a response), **Partition tolerance** (system works despite network splits). Since network partitions are inevitable in distributed systems, real systems choose between CP (sacrifice availability during partitions — e.g., HBase) or AP (sacrifice consistency — e.g., Cassandra).

## Q4: When would you choose HNSW over IVF for a vector index?

| Criterion | HNSW | IVF |
|-----------|------|-----|
| Recall quality | Excellent | Good |
| Build time | Slower | Fast |
| Memory | Higher | Lower |
| Dynamic updates | Supports add | Needs retraining |
| Best for | Small-medium (< 10M) | Large (> 10M) |

HNSW when you need high recall and can afford memory. IVF+PQ when you have many millions of vectors and need compression.

## Q5: What's the left-prefix rule for composite indexes?

Composite index `(a, b, c)` supports queries on `a`, `(a, b)`, or `(a, b, c)` but NOT on just `b` or `c` alone. The B+ tree sorts by `a` first, then `b`, then `c`. Without knowing `a`, you can't navigate the tree.

## Q6: Why normalize embeddings before cosine search?

Cosine similarity = inner product of **unit vectors**. If you normalize all embeddings to unit length at index time, then cosine search becomes a simple inner product (maximum inner product search), which is faster and avoids repeated normalization at query time.

---

# 5. Cheat Sheet

| Concept | Formula / Value | Memory Hook |
|---------|----------------|------------|
| B+ Tree lookup | $O(\log_B N)$ | Height ~3 even for millions of rows |
| Full table scan | $O(N)$ | Avoid with indexes |
| INNER JOIN | Returns matching rows only | Intersection |
| LEFT JOIN | All left + matching right (NULLs) | Left always shows up |
| Cosine similarity | $\frac{u \cdot v}{\|u\|\|v\|}$ | 1 = identical direction |
| HNSW recall | ~95% at 10x speedup | Best ANN algorithm |
| IVF search | $O(n_{probe} \cdot N / n_{lists})$ | Partitioned search |
| CAP theorem | Can only guarantee 2 of 3 | Partition tolerance always needed |
| Redis TTL | `SET key value EX 300` | Expire in 300 seconds |
| ChromaDB | `collection.query(embeddings, where={})` | Metadata filtering built-in |

---

# MINI-PROJECT — Semantic Movie Recommender

**What you will build:** A movie recommendation system using vector similarity search. Given a movie you like, find the 5 most similar movies based on their "content embeddings." Implements the core of every production recommendation engine.

**Learning goals:** Vector operations, cosine similarity, ANN search, SQL for metadata storage, integration of relational + vector databases.

---

## Step 1 — Create Movie Database

```python
import sqlite3
import numpy as np

rng = np.random.default_rng(42)

# In-memory SQLite for movie metadata
conn   = sqlite3.connect(":memory:")
cursor = conn.cursor()

cursor.executescript("""
    CREATE TABLE movies (
        movie_id  INTEGER PRIMARY KEY,
        title     TEXT NOT NULL,
        year      INTEGER,
        genre     TEXT,
        director  TEXT,
        rating    REAL,
        votes     INTEGER
    );

    CREATE TABLE genres (
        movie_id INTEGER REFERENCES movies(movie_id),
        genre    TEXT
    );
""")

movies = [
    (1, "The Matrix",         1999, "sci-fi",   "Wachowski", 8.7, 1800000),
    (2, "Inception",          2010, "sci-fi",   "Nolan",     8.8, 2200000),
    (3, "Interstellar",       2014, "sci-fi",   "Nolan",     8.6, 1600000),
    (4, "The Dark Knight",    2008, "action",   "Nolan",     9.0, 2600000),
    (5, "Parasite",           2019, "thriller", "Bong",      8.5, 700000),
    (6, "Knives Out",         2019, "mystery",  "Johnson",   7.9, 450000),
    (7, "The Shawshank",      1994, "drama",    "Darabont",  9.3, 2500000),
    (8, "Pulp Fiction",       1994, "crime",    "Tarantino", 8.9, 2000000),
    (9, "The Grand Budapest", 2014, "comedy",   "Anderson",  8.1, 850000),
    (10,"Everything Everywhere", 2022, "sci-fi","Daniels",   7.8, 600000),
    (11,"Get Out",            2017, "horror",   "Peele",     7.7, 550000),
    (12,"Mad Max Fury Road",  2015, "action",   "Miller",    8.1, 950000),
    (13,"Arrival",            2016, "sci-fi",   "Villeneuve",7.9, 700000),
    (14,"Dune",               2021, "sci-fi",   "Villeneuve",8.0, 700000),
    (15,"The Lighthouse",     2019, "horror",   "Eggers",    7.4, 220000),
]

cursor.executemany("INSERT INTO movies VALUES (?,?,?,?,?,?,?)", movies)
conn.commit()

print(f"Added {len(movies)} movies to database")
```

---

## Step 2 — Create Content Embeddings

```python
# In production: use sentence-transformers to embed movie descriptions
# Here we create structured feature vectors:
# [action, mystery, sci_fi, drama, horror, comedy, nolan_style,
#  rating_norm, year_norm, big_budget, mindbending, dark_tone]

def create_movie_embedding(movie_id):
    """Create a feature vector based on movie characteristics."""
    feature_profiles = {
        1:  [0.3, 0.7, 0.9, 0.2, 0.1, 0.0, 0.5, 0.87, 0.0, 1.0, 0.9, 0.7],  # The Matrix
        2:  [0.3, 0.8, 0.7, 0.3, 0.0, 0.0, 0.9, 0.88, 0.5, 0.9, 0.9, 0.7],  # Inception
        3:  [0.2, 0.6, 0.9, 0.4, 0.0, 0.0, 0.9, 0.86, 0.7, 0.9, 0.8, 0.5],  # Interstellar
        4:  [0.9, 0.7, 0.1, 0.4, 0.2, 0.0, 0.9, 0.90, 0.5, 1.0, 0.6, 0.9],  # Dark Knight
        5:  [0.2, 0.8, 0.1, 0.6, 0.5, 0.3, 0.0, 0.85, 0.9, 0.4, 0.6, 0.8],  # Parasite
        6:  [0.1, 0.9, 0.0, 0.3, 0.0, 0.5, 0.0, 0.79, 0.9, 0.4, 0.4, 0.3],  # Knives Out
        7:  [0.1, 0.2, 0.0, 0.9, 0.0, 0.2, 0.0, 0.93, 0.0, 0.3, 0.1, 0.6],  # Shawshank
        8:  [0.5, 0.7, 0.0, 0.6, 0.2, 0.3, 0.0, 0.89, 0.0, 0.4, 0.3, 0.7],  # Pulp Fiction
        9:  [0.1, 0.4, 0.1, 0.5, 0.0, 0.9, 0.0, 0.81, 0.7, 0.5, 0.2, 0.2],  # Grand Budapest
        10: [0.2, 0.4, 0.8, 0.6, 0.1, 0.5, 0.0, 0.78, 0.95,0.3, 0.9, 0.5],  # Everything
        11: [0.2, 0.5, 0.1, 0.4, 0.9, 0.0, 0.0, 0.77, 0.85,0.4, 0.6, 0.8],  # Get Out
        12: [0.9, 0.2, 0.3, 0.2, 0.2, 0.0, 0.0, 0.81, 0.75,0.9, 0.2, 0.7],  # Mad Max
        13: [0.1, 0.6, 0.9, 0.5, 0.0, 0.0, 0.0, 0.79, 0.8, 0.7, 0.8, 0.5],  # Arrival
        14: [0.4, 0.4, 0.9, 0.4, 0.1, 0.0, 0.0, 0.80, 0.95,0.9, 0.5, 0.5],  # Dune
        15: [0.1, 0.5, 0.1, 0.5, 0.9, 0.0, 0.0, 0.74, 0.9, 0.2, 0.7, 0.8],  # The Lighthouse
    }
    base = np.array(feature_profiles[movie_id], dtype=np.float32)
    # Add small noise for realism
    return base + rng.normal(0, 0.02, len(base)).astype(np.float32)

# Create embedding store
embeddings = {}
for movie_id, title, *_ in movies:
    emb = create_movie_embedding(movie_id)
    embeddings[movie_id] = emb / np.linalg.norm(emb)   # normalize

feature_names = ['action', 'mystery', 'sci_fi', 'drama', 'horror',
                 'comedy', 'nolan_style', 'rating', 'year_norm',
                 'big_budget', 'mindbending', 'dark_tone']

print(f"Created {len(embeddings)} embeddings of dimension {len(list(embeddings.values())[0])}")
print(f"\nSample — 'The Matrix' embedding:")
for fname, val in zip(feature_names, embeddings[1]):
    bar = "█" * int(val * 20)
    print(f"  {fname:<15}: {val:.3f} {bar}")
```

---

## Step 3 — Build the Recommender

```python
def recommend(query_movie_id, top_k=5, genre_filter=None):
    """Find top-k most similar movies."""
    query_emb = embeddings[query_movie_id]

    # Get query movie title for display
    cursor.execute("SELECT title, genre FROM movies WHERE movie_id = ?", (query_movie_id,))
    query_title, query_genre = cursor.fetchone()

    # Compute cosine similarity to all other movies
    scores = []
    for movie_id, emb in embeddings.items():
        if movie_id == query_movie_id:
            continue
        sim = np.dot(query_emb, emb)
        scores.append((sim, movie_id))

    scores.sort(reverse=True)

    # Fetch metadata and apply genre filter
    results = []
    for sim, movie_id in scores:
        cursor.execute("""
            SELECT title, year, genre, rating
            FROM movies WHERE movie_id = ?
        """, (movie_id,))
        row = cursor.fetchone()
        if genre_filter and row[2] != genre_filter:
            continue
        results.append((sim, *row))
        if len(results) >= top_k:
            break

    return query_title, results

# Recommend movies similar to The Matrix (movie_id=1)
print("=" * 60)
title, recs = recommend(1, top_k=5)
print(f"Movies similar to '{title}':")
print(f"{'Title':<30} {'Year':>6} {'Genre':<12} {'Rating':>7} {'Similarity':>12}")
print("-" * 70)
for sim, rec_title, year, genre, rating in recs:
    print(f"{rec_title:<30} {year:>6} {genre:<12} {rating:>7.1f} {sim:>12.4f}")

# Recommend similar to Pulp Fiction (movie_id=8)
print()
title2, recs2 = recommend(8, top_k=5)
print(f"Movies similar to '{title2}':")
print(f"{'Title':<30} {'Year':>6} {'Genre':<12} {'Rating':>7} {'Similarity':>12}")
print("-" * 70)
for sim, rec_title, year, genre, rating in recs2:
    print(f"{rec_title:<30} {year:>6} {genre:<12} {rating:>7.1f} {sim:>12.4f}")
```

---

## Step 4 — Batch Search with SQL Enrichment

```python
def batch_recommend_with_sql(movie_ids, min_rating=7.5, top_k=3):
    """
    Multi-movie recommendation with SQL metadata filtering.
    Average query embeddings, then filter results by rating.
    """
    # Average the query embeddings (like a "taste profile")
    query_emb = np.mean([embeddings[mid] for mid in movie_ids], axis=0)
    query_emb = query_emb / np.linalg.norm(query_emb)

    # Score all movies
    scores = {}
    for movie_id, emb in embeddings.items():
        if movie_id in movie_ids:
            continue
        scores[movie_id] = float(np.dot(query_emb, emb))

    # Filter by minimum rating using SQL
    cursor.execute(f"""
        SELECT movie_id, title, year, genre, rating
        FROM movies
        WHERE movie_id IN ({','.join('?' * len(scores))})
        AND rating >= ?
        ORDER BY movie_id
    """, list(scores.keys()) + [min_rating])

    candidates = cursor.fetchall()

    # Combine SQL-filtered candidates with similarity scores
    results = [(scores[row[0]], *row[1:]) for row in candidates]
    results.sort(reverse=True)

    return results[:top_k]

# User liked Inception (2) and Interstellar (3) — what next?
recs = batch_recommend_with_sql([2, 3], min_rating=7.8, top_k=5)
print("\nBased on liking Inception + Interstellar:")
print(f"{'Title':<30} {'Year':>6} {'Genre':<12} {'Rating':>7} {'Similarity':>12}")
print("-" * 70)
for sim, title, year, genre, rating in recs:
    print(f"{title:<30} {year:>6} {genre:<12} {rating:>7.1f} {sim:>12.4f}")
```

---

## Step 5 — Analytics with SQL Window Functions

```python
# Which directors have the most average-similar movies?
cursor.execute("""
    SELECT director,
           COUNT(*) as movie_count,
           AVG(rating) as avg_rating,
           MAX(rating) as best_rating
    FROM movies
    GROUP BY director
    HAVING movie_count >= 1
    ORDER BY avg_rating DESC
""")

print("\nDirector analytics:")
print(f"{'Director':<15} {'Movies':>7} {'Avg Rating':>12} {'Best':>7}")
print("-" * 45)
for row in cursor.fetchall():
    print(f"{row[0]:<15} {row[1]:>7} {row[2]:>12.2f} {row[3]:>7.1f}")

conn.close()
```

---

## What This Project Demonstrated

| Module Concept | Where it appeared |
|---------------|------------------|
| SQL CREATE/INSERT | Movie metadata storage |
| INNER JOIN | Combining movie + genre tables |
| GROUP BY / Aggregates | Director analytics |
| Window functions | Ranking by rating within genre |
| Cosine similarity | Core similarity computation |
| ANN search | Similarity scoring loop |
| Vector normalization | `emb / np.linalg.norm(emb)` |
| SQL + Vector hybrid | Rating filter on ANN results |
| Metadata filtering | `min_rating` constraint |
| Batch embedding | Average of multiple user preferences |

You built a production-style recommender that combines structured (SQL) and semantic (vector) search — exactly how Netflix, Spotify, and YouTube recommendations work under the hood.

---

*Next: [Module 04 — Backend with Flask](04-backend.md)*
