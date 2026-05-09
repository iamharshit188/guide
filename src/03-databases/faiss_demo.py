"""
FAISS deep dive — covers all major index types with benchmarks.
pip install faiss-cpu   (or faiss-gpu for CUDA)

Topics:
  - FlatL2 / FlatIP (exact brute-force baseline)
  - IVFFlat (inverted file, approximate)
  - HNSWFlat (graph-based ANN)
  - IndexPQ (product quantization — compression)
  - IndexIVFPQ (billion-scale workhorse)
  - Recall vs QPS tradeoffs
  - ID mapping (IndexIDMap)
  - Serialization
"""

import numpy as np
import time
import struct
import tempfile
import os

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("faiss-cpu not installed. Run: pip install faiss-cpu")
    print("Showing expected output structure.\n")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Helpers ───────────────────────────────────────────────────

def make_dataset(n, d, seed=42, normalize=True):
    """Generate n random d-dim vectors."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / (norms + 1e-10)
    return X


def compute_recall(true_ids, approx_ids):
    """
    Recall@k: fraction of true k-NN found in approximate result.
    true_ids/approx_ids: (n_queries, k) arrays.
    """
    n, k = true_ids.shape
    recall = 0.0
    for i in range(n):
        true_set = set(true_ids[i].tolist())
        found = sum(1 for j in approx_ids[i] if j in true_set)
        recall += found / k
    return recall / n


def benchmark_index(index, xb, xq, k=10, name="Index"):
    """Query benchmark: returns (QPS, mean_latency_ms, results)."""
    n_queries = len(xq)
    # Warmup
    index.search(xq[:min(10, n_queries)], k)
    # Timed run
    t0 = time.perf_counter()
    D, I = index.search(xq, k)
    elapsed = time.perf_counter() - t0
    qps = n_queries / elapsed
    lat_ms = elapsed * 1000 / n_queries
    return qps, lat_ms, D, I


def print_bench(name, qps, lat_ms, recall=None, memory_mb=None):
    recall_str = f"Recall@10={recall:.4f}" if recall is not None else ""
    mem_str    = f"Mem≈{memory_mb:.1f}MB"  if memory_mb  is not None else ""
    print(f"  {name:30s}  QPS={qps:8.0f}  Lat={lat_ms:.3f}ms  {recall_str}  {mem_str}")


# ── Main ──────────────────────────────────────────────────────

def main():
    if not FAISS_AVAILABLE:
        return

    N, D, N_QUERY, K = 50_000, 128, 200, 10
    print(f"Dataset: N={N:,} vectors, D={D}, queries={N_QUERY}, k={K}")

    xb = make_dataset(N, D, seed=0, normalize=True)
    xq = make_dataset(N_QUERY, D, seed=1, normalize=True)

    section("1. INDEXFLATL2 — EXACT BRUTE-FORCE BASELINE")
    index_flat = faiss.IndexFlatL2(D)
    index_flat.add(xb)

    D_flat, I_flat = index_flat.search(xq, K)
    qps, lat, D_flat, I_flat = benchmark_index(index_flat, xb, xq, K, "FlatL2")
    mem_flat = (N * D * 4) / 1e6  # 4 bytes per float32

    print(f"  Vectors indexed: {index_flat.ntotal:,}")
    print(f"  Memory (data only): {mem_flat:.1f} MB")
    print_bench("IndexFlatL2 (exact)", qps, lat, recall=1.0, memory_mb=mem_flat)
    print(f"  Top-3 neighbors of query[0]: {I_flat[0, :3]}")
    print(f"  Distances:                   {D_flat[0, :3].round(4)}")

    section("2. INDEXFLATIP — INNER PRODUCT (MIPS)")
    # For normalized vectors: IP ≡ cosine similarity
    index_ip = faiss.IndexFlatIP(D)
    index_ip.add(xb)
    D_ip, I_ip = index_ip.search(xq, K)
    qps_ip, lat_ip, _, _ = benchmark_index(index_ip, xb, xq, K)
    print_bench("IndexFlatIP (cosine)", qps_ip, lat_ip, recall=1.0, memory_mb=mem_flat)
    print(f"  Scores (cosine similarities): {D_ip[0, :3].round(4)}")
    # FlatL2 and FlatIP give same ordering for normalized vectors
    order_match = np.array_equal(I_flat[0], I_ip[0])
    print(f"  FlatL2 and FlatIP same order (normalized vecs): {order_match}")

    section("3. IVFFLAT — INVERTED FILE INDEX")
    # n_list: number of Voronoi cells.
    # Rule of thumb: n_list ≈ sqrt(N) for N < 1M, 4*sqrt(N) for N > 1M
    n_list = int(np.sqrt(N))
    quantizer = faiss.IndexFlatL2(D)
    index_ivf = faiss.IndexIVFFlat(quantizer, D, n_list, faiss.METRIC_L2)

    # IVF must be trained (K-Means over xb to find cluster centroids)
    t0 = time.perf_counter()
    index_ivf.train(xb)
    train_time = (time.perf_counter() - t0) * 1000
    index_ivf.add(xb)
    print(f"  n_list={n_list}, train time={train_time:.1f}ms, indexed={index_ivf.ntotal:,}")

    print(f"\n  nprobe sweep (recall vs speed tradeoff):")
    print(f"  {'nprobe':>8}  {'QPS':>10}  {'Lat(ms)':>9}  {'Recall@10':>10}")
    print("  " + "-" * 45)
    for nprobe in [1, 4, 16, 32, 64, n_list]:
        index_ivf.nprobe = nprobe
        qps_ivf, lat_ivf, _, I_ivf = benchmark_index(index_ivf, xb, xq, K)
        recall_ivf = compute_recall(I_flat, I_ivf)
        print(f"  {nprobe:>8}  {qps_ivf:>10.0f}  {lat_ivf:>9.3f}  {recall_ivf:>10.4f}")

    section("4. HNSWFLAT — GRAPH-BASED ANN")
    # M: connections per node (16-64 typical)
    # efConstruction: beam width during build (200 typical)
    M = 16
    index_hnsw = faiss.IndexHNSWFlat(D, M)
    index_hnsw.hnsw.efConstruction = 200

    t0 = time.perf_counter()
    index_hnsw.add(xb)
    build_time = (time.perf_counter() - t0) * 1000

    # Memory: data + graph (~M * 2 links per vector, 4 bytes each)
    mem_hnsw = (N * D * 4 + N * M * 2 * 4) / 1e6
    print(f"  M={M}, efConstruction=200, build={build_time:.0f}ms, mem≈{mem_hnsw:.1f}MB")

    print(f"\n  ef_search sweep:")
    print(f"  {'efSearch':>9}  {'QPS':>10}  {'Lat(ms)':>9}  {'Recall@10':>10}")
    print("  " + "-" * 45)
    for ef in [10, 32, 64, 128, 200, 400]:
        index_hnsw.hnsw.efSearch = ef
        qps_hnsw, lat_hnsw, _, I_hnsw = benchmark_index(index_hnsw, xb, xq, K)
        recall_hnsw = compute_recall(I_flat, I_hnsw)
        print(f"  {ef:>9}  {qps_hnsw:>10.0f}  {lat_hnsw:>9.3f}  {recall_hnsw:>10.4f}")

    section("5. PRODUCT QUANTIZATION (IndexPQ)")
    # M sub-quantizers, nbits bits per sub-quantizer
    # Compression: 4*D bytes → M*(nbits/8) bytes
    M_pq, nbits = 8, 8   # 8 sub-quantizers, 1 byte each → 8 bytes/vector (vs 512 for D=128 float32)
    index_pq = faiss.IndexPQ(D, M_pq, nbits)

    t0 = time.perf_counter()
    index_pq.train(xb)
    train_pq = (time.perf_counter() - t0) * 1000
    index_pq.add(xb)

    mem_float = N * D * 4 / 1e6
    mem_pq = N * M_pq * (nbits / 8) / 1e6
    compression = mem_float / mem_pq

    qps_pq, lat_pq, _, I_pq = benchmark_index(index_pq, xb, xq, K)
    recall_pq = compute_recall(I_flat, I_pq)

    print(f"  M={M_pq} sub-quantizers, nbits={nbits}")
    print(f"  Original: {mem_float:.1f}MB → PQ: {mem_pq:.1f}MB ({compression:.0f}x compression)")
    print(f"  Train time: {train_pq:.0f}ms")
    print_bench("IndexPQ", qps_pq, lat_pq, recall=recall_pq, memory_mb=mem_pq)

    section("6. IVFPQ — BILLION-SCALE WORKHORSE")
    n_list_pq = 256
    M_ivfpq, nbits_ivfpq = 8, 8
    quantizer2 = faiss.IndexFlatL2(D)
    index_ivfpq = faiss.IndexIVFPQ(quantizer2, D, n_list_pq, M_ivfpq, nbits_ivfpq)

    t0 = time.perf_counter()
    index_ivfpq.train(xb)
    train_ivfpq = (time.perf_counter() - t0) * 1000
    index_ivfpq.add(xb)

    mem_ivfpq = (N * M_ivfpq + n_list_pq * D * 4) / 1e6

    print(f"  n_list={n_list_pq}, M={M_ivfpq}, nbits={nbits_ivfpq}")
    print(f"  Train time: {train_ivfpq:.0f}ms")

    print(f"\n  nprobe sweep (IVFPQ):")
    print(f"  {'nprobe':>8}  {'QPS':>10}  {'Lat(ms)':>9}  {'Recall@10':>10}")
    print("  " + "-" * 45)
    for nprobe in [1, 4, 16, 32, 64]:
        index_ivfpq.nprobe = nprobe
        qps_ivfpq, lat_ivfpq, _, I_ivfpq = benchmark_index(index_ivfpq, xb, xq, K)
        recall_ivfpq = compute_recall(I_flat, I_ivfpq)
        print(f"  {nprobe:>8}  {qps_ivfpq:>10.0f}  {lat_ivfpq:>9.3f}  {recall_ivfpq:>10.4f}")

    section("7. INDEXIDMAP — CUSTOM EXTERNAL IDs")
    # FAISS uses 0-indexed int64 IDs internally.
    # IndexIDMap maps arbitrary int64 IDs to internal indices.
    inner = faiss.IndexFlatL2(D)
    index_idmap = faiss.IndexIDMap(inner)

    # Use doc IDs 1000, 1001, ... instead of 0, 1, ...
    custom_ids = np.arange(1000, 1000 + N, dtype=np.int64)
    index_idmap.add_with_ids(xb, custom_ids)

    _, I_id = index_idmap.search(xq[:3], 3)
    print(f"Query returns custom IDs:")
    for i, row in enumerate(I_id):
        print(f"  query[{i}]: top-3 IDs = {row.tolist()}")

    section("8. SERIALIZATION — SAVE & LOAD")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "hnsw_index.faiss")

        faiss.write_index(index_hnsw, path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  Written HNSW index: {path}")
        print(f"  File size: {size_mb:.1f} MB")

        loaded = faiss.read_index(path)
        loaded.hnsw.efSearch = 64
        _, I_loaded = loaded.search(xq[:5], 3)
        _, I_orig = index_hnsw.search(xq[:5], 3)
        match = np.array_equal(I_loaded, I_orig)
        print(f"  Loaded index results match original: {match}")

    section("9. FULL COMPARISON TABLE")
    print(f"{'Index':20s}  {'QPS':>8}  {'Recall':>8}  {'Mem(MB)':>9}  {'Trainable':>10}")
    print("-" * 65)

    # Gather results
    index_hnsw.hnsw.efSearch = 64
    qps_h, lat_h, _, I_h = benchmark_index(index_hnsw, xb, xq, K)
    recall_h = compute_recall(I_flat, I_h)

    index_ivf.nprobe = 16
    qps_i16, lat_i16, _, I_i16 = benchmark_index(index_ivf, xb, xq, K)
    recall_i16 = compute_recall(I_flat, I_i16)

    index_ivfpq.nprobe = 16
    qps_ip16, lat_ip16, _, I_ip16 = benchmark_index(index_ivfpq, xb, xq, K)
    recall_ip16 = compute_recall(I_flat, I_ip16)

    rows = [
        ("FlatL2 (exact)",    qps,      1.0,         mem_flat,  "No"),
        ("IVFFlat np=16",     qps_i16,  recall_i16,  mem_flat,  "Yes (KMeans)"),
        ("HNSWFlat ef=64",    qps_h,    recall_h,    mem_hnsw,  "No"),
        ("IndexPQ",           qps_pq,   recall_pq,   mem_pq,    "Yes (PQ)"),
        ("IVFPQ np=16",       qps_ip16, recall_ip16, mem_ivfpq, "Yes (both)"),
    ]
    for name, qps_r, recall_r, mem_r, train_r in rows:
        print(f"{name:20s}  {qps_r:>8.0f}  {recall_r:>8.4f}  {mem_r:>9.1f}  {train_r:>10}")

    section("10. SELECTION GUIDE")
    print("""
  Choose your FAISS index:

  N < 10K    → FlatL2/FlatIP  (exact, no overhead)
  N < 1M     → HNSWFlat       (best recall/speed, ~2x memory)
  N < 10M    → IVFFlat        (good recall with nprobe tuning)
  N > 10M    → IVFPQ          (compressed, scalable, tune nprobe)
  Memory critical → IndexPQ   (maximum compression)

  Metric choice:
  • Cosine similarity → normalize vectors + use IndexFlatIP
  • L2 distance → IndexFlatL2 (unnormalized vectors)
  • Dot product → IndexFlatIP (raw scores, e.g., recommendation)

  Production checklist:
  ✓ Normalize vectors if using cosine/IP
  ✓ Train on representative sample (50K-500K) before adding all data
  ✓ Benchmark recall@10 vs QPS on your actual query distribution
  ✓ Serialize index with faiss.write_index() for persistence
  ✓ For multi-GPU: faiss.index_cpu_to_all_gpus(index)
    """)


if __name__ == "__main__":
    main()
