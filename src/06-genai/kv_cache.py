"""
KV cache simulation and memory analysis — pure NumPy.
Covers: autoregressive generation with/without cache, FLOP and memory math,
        MQA/GQA cache reduction, PagedAttention intuition.
pip install numpy
"""

import numpy as np
import math
import time

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Tiny Transformer layer (single MHA + FFN) for timing demo
# ---------------------------------------------------------------------------

class TinyAttentionLayer:
    """
    One self-attention layer with optional KV cache.
    d_model, n_heads, d_k = d_model // n_heads
    """

    def __init__(self, d_model, n_heads, rng):
        self.d_model = d_model
        self.h = n_heads
        self.d_k = d_model // n_heads
        scale = 1.0 / math.sqrt(d_model)
        self.WQ = rng.uniform(-scale, scale, (d_model, d_model))
        self.WK = rng.uniform(-scale, scale, (d_model, d_model))
        self.WV = rng.uniform(-scale, scale, (d_model, d_model))
        self.WO = rng.uniform(-scale, scale, (d_model, d_model))

    def _split_heads(self, X):
        """(seq, d_model) → (h, seq, d_k)"""
        seq = X.shape[0]
        return X.reshape(seq, self.h, self.d_k).transpose(1, 0, 2)

    def _merge_heads(self, X):
        """(h, seq, d_k) → (seq, d_model)"""
        h, seq, dk = X.shape
        return X.transpose(1, 0, 2).reshape(seq, h * dk)

    def forward_no_cache(self, X):
        """
        X: (seq, d_model) — full sequence every step.
        Returns: (seq, d_model)
        """
        Q = self._split_heads(X @ self.WQ)   # (h, seq, d_k)
        K = self._split_heads(X @ self.WK)
        V = self._split_heads(X @ self.WV)

        scores = Q @ K.swapaxes(-2, -1) / math.sqrt(self.d_k)
        W = softmax(scores, axis=-1)
        out = self._merge_heads(W @ V)        # (seq, d_model)
        return out @ self.WO

    def forward_with_cache(self, x_new, k_cache, v_cache):
        """
        x_new   : (1, d_model) — only the new token
        k_cache : (h, t, d_k) — accumulated keys so far
        v_cache : (h, t, d_v)
        Returns: (1, d_model), updated k_cache, v_cache
        """
        # Compute Q, K, V for the new token only
        q = self._split_heads(x_new @ self.WQ)   # (h, 1, d_k)
        k_new = self._split_heads(x_new @ self.WK)
        v_new = self._split_heads(x_new @ self.WV)

        # Append to cache
        k_cache = np.concatenate([k_cache, k_new], axis=1)  # (h, t+1, d_k)
        v_cache = np.concatenate([v_cache, v_new], axis=1)

        # Attend over full cache
        scores = q @ k_cache.swapaxes(-2, -1) / math.sqrt(self.d_k)  # (h, 1, t+1)
        W = softmax(scores, axis=-1)
        out = self._merge_heads(W @ v_cache)   # (1, d_model)
        return (out @ self.WO), k_cache, v_cache


# ---------------------------------------------------------------------------
# Memory footprint math
# ---------------------------------------------------------------------------

def kv_cache_bytes(batch, seq_len, n_layers, n_heads, d_k,
                   n_kv_heads=None, dtype_bytes=2):
    """
    KV cache size in bytes.
    n_kv_heads: for MQA/GQA (None = MHA = n_heads).
    """
    kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
    # 2 = K and V; 2 = bytes per float16
    return 2 * batch * seq_len * n_layers * kv_heads * d_k * dtype_bytes


def flops_generation_no_cache(T, d_model, n_heads, n_layers):
    """Total FLOPs to generate T tokens without cache (recomputes all K,V each step)."""
    total = 0
    for t in range(1, T + 1):
        # QK^T: (h, t, d_k) @ (h, d_k, t) = h * t * t * d_k
        d_k = d_model // n_heads
        total += n_layers * n_heads * t * t * d_k   # attention
        total += n_layers * t * d_model * 3          # Q, K, V projections
    return total


def flops_generation_with_cache(T, d_model, n_heads, n_layers):
    """Total FLOPs to generate T tokens WITH cache."""
    total = 0
    d_k = d_model // n_heads
    for t in range(1, T + 1):
        # Only compute Q for 1 token; K,V for 1 token; attend over t cached
        total += n_layers * n_heads * 1 * t * d_k   # attention (1 query over t keys)
        total += n_layers * 1 * d_model * 3          # Q, K, V projection (1 token)
    return total


def main():
    rng = np.random.default_rng(42)
    d_model, n_heads = 32, 4
    n_layers = 2

    # -----------------------------------------------------------------------
    section("AUTOREGRESSIVE GENERATION — WITHOUT CACHE")
    # -----------------------------------------------------------------------
    layer = TinyAttentionLayer(d_model, n_heads, rng)
    gen_len = 12
    prompt_len = 4
    # Start with prompt
    tokens = rng.standard_normal((prompt_len, d_model))
    generated = list(tokens)

    print(f"\n  Generating {gen_len} tokens (d_model={d_model}, no cache).")
    print(f"  Each step recomputes attention over ALL previous tokens.")
    t0 = time.perf_counter()
    for step in range(gen_len):
        X = np.array(generated)                  # (t, d_model)
        out = layer.forward_no_cache(X)          # (t, d_model)
        next_tok = out[-1:] + rng.standard_normal((1, d_model)) * 0.01
        generated.append(next_tok[0])
        if step < 3 or step == gen_len - 1:
            print(f"    step {step+1:2d}  input_len={X.shape[0]:3d}  "
                  f"output norm={np.linalg.norm(out[-1]):.4f}")
    t_no_cache = time.perf_counter() - t0
    print(f"  Time (no cache): {t_no_cache*1000:.2f} ms")

    # -----------------------------------------------------------------------
    section("AUTOREGRESSIVE GENERATION — WITH KV CACHE")
    # -----------------------------------------------------------------------
    tokens = rng.standard_normal((prompt_len, d_model))
    generated_c = list(tokens)
    d_k = d_model // n_heads

    # Prefill: compute cache for the prompt
    k_cache = np.zeros((n_heads, 0, d_k))
    v_cache = np.zeros((n_heads, 0, d_k))
    for i in range(prompt_len):
        x = tokens[i:i+1]
        _, k_cache, v_cache = layer.forward_with_cache(x, k_cache, v_cache)

    t0 = time.perf_counter()
    for step in range(gen_len):
        x_new = np.array(generated_c[-1:])[np.newaxis, 0:1]   # (1, d_model)
        out, k_cache, v_cache = layer.forward_with_cache(
            generated_c[-1].reshape(1, -1), k_cache, v_cache)
        next_tok = out + rng.standard_normal((1, d_model)) * 0.01
        generated_c.append(next_tok[0])
        if step < 3 or step == gen_len - 1:
            print(f"    step {step+1:2d}  cache_len={k_cache.shape[1]:3d}  "
                  f"output norm={np.linalg.norm(out):.4f}")
    t_cache = time.perf_counter() - t0
    print(f"  Time (with cache): {t_cache*1000:.2f} ms")

    if t_no_cache > 0 and t_cache > 0:
        print(f"\n  Speedup (for this tiny model/length): {t_no_cache/t_cache:.2f}×")
        print("  (Speedup is much larger for long sequences and real model sizes.)")

    # -----------------------------------------------------------------------
    section("FLOP COMPARISON: WITH vs WITHOUT CACHE")
    # -----------------------------------------------------------------------
    print(f"\n  n_layers={n_layers}, d_model={d_model}, n_heads={n_heads}")
    print(f"  {'T (tokens)':>12} {'No cache (MFLOPs)':>20} {'With cache (MFLOPs)':>22} {'Speedup':>10}")
    for T in [16, 64, 256, 1024, 4096]:
        f_no = flops_generation_no_cache(T, d_model, n_heads, n_layers) / 1e6
        f_ca = flops_generation_with_cache(T, d_model, n_heads, n_layers) / 1e6
        print(f"  {T:>12} {f_no:>20.1f} {f_ca:>22.1f} {f_no/f_ca:>10.1f}×")

    # -----------------------------------------------------------------------
    section("KV CACHE MEMORY FOOTPRINT")
    # -----------------------------------------------------------------------
    print("\n  KV cache formula:")
    print("  bytes = 2 × B × T × L × h_kv × d_k × 2   (K+V, float16)")
    print()

    # Realistic model sizes
    models = [
        ("GPT-2 (117M)", 12, 12, 64),
        ("LLaMA-7B",     32, 32, 128),
        ("LLaMA-70B",    80, 64, 128),
        ("GPT-3 (175B)", 96, 96, 128),
    ]
    print(f"  {'Model':<18} {'L':>4} {'h':>4} {'d_k':>5} "
          f"{'T=512 (MB)':>12} {'T=2048 (MB)':>14} {'T=8192 (MB)':>14}")
    for name, L, h, dk in models:
        b = 1
        s512  = kv_cache_bytes(b, 512,  L, h, dk) / 1e6
        s2048 = kv_cache_bytes(b, 2048, L, h, dk) / 1e6
        s8192 = kv_cache_bytes(b, 8192, L, h, dk) / 1e6
        print(f"  {name:<18} {L:>4} {h:>4} {dk:>5} "
              f"{s512:>12.1f} {s2048:>14.1f} {s8192:>14.1f}")

    # -----------------------------------------------------------------------
    section("MQA / GQA CACHE REDUCTION")
    # -----------------------------------------------------------------------
    print("\n  Reducing n_kv_heads shrinks KV cache proportionally.")
    L, h, dk, T, B = 32, 32, 128, 2048, 1
    print(f"\n  Model: L={L}, h_q={h}, d_k={dk}, T={T}, B={B}")
    print(f"  {'Variant':<20} {'n_kv_heads':>12} {'Cache (MB)':>12} {'Reduction':>12}")
    for variant, n_kv in [("MHA", h), ("GQA (g=8)", 8), ("GQA (g=4)", 4),
                          ("GQA (g=2)", 2), ("MQA", 1)]:
        mb = kv_cache_bytes(B, T, L, h, dk, n_kv_heads=n_kv) / 1e6
        red = kv_cache_bytes(B, T, L, h, dk) / 1e6
        print(f"  {variant:<20} {n_kv:>12} {mb:>12.1f} {red/mb:>11.1f}×")

    # -----------------------------------------------------------------------
    section("PAGEDATTENTION INTUITION")
    # -----------------------------------------------------------------------
    print("""
  Problem: KV cache has variable length per request.
  Naive allocation: allocate T_max × L × h × d_k per slot → massive waste.

  PagedAttention (vLLM):
    - Divide KV cache into fixed-size "pages" (e.g., 16 tokens each)
    - Maintain a block table: logical sequence → physical page addresses
    - Allocate pages on demand; free on sequence completion
    - Non-contiguous physical layout — no memory fragmentation

  Analogy: OS virtual memory paging for process address spaces.

  Benefits:
    - Near-zero memory waste (< 4% fragmentation vs 60-80% naive)
    - Enables prefix sharing (system prompt cached once across requests)
    - Higher throughput batching (more concurrent sequences fit in GPU RAM)
    """)

    # -----------------------------------------------------------------------
    section("GENERATION LOOP PSEUDOCODE")
    # -----------------------------------------------------------------------
    print("""
  Without cache:
    for t in range(T):
        X = tokens[0:t+1]                    # grows each step: O(t)
        K, V = compute_kv(X)                 # recomputed: O(t) work
        q_t = compute_q(X[-1])
        out_t = attention(q_t, K, V)
        tokens.append(sample(out_t))

  With cache:
    k_cache, v_cache = [], []
    for t in range(T):
        x_t = tokens[t]                      # only new token: O(1) input
        q_t, k_t, v_t = project(x_t)        # O(1) projection
        k_cache.append(k_t)                  # O(1) append
        v_cache.append(v_t)
        out_t = attention(q_t, k_cache, v_cache)  # O(t) attend
        tokens.append(sample(out_t))

  Total FLOPs:
    No cache: O(T² × d)      (T steps, each O(t×d) average)
    Cache:    O(T × d)       (T steps, each O(d) projection + O(t) attention)
  But cache attend is O(t) × d_k per step → O(T² d_k / T) = O(T d_k) total? No:
  Total attend with cache = Σ_{t=1}^{T} t × d_k = O(T² d_k / 2)
  Still O(T²) in FLOPs but constant factor advantage over no-cache.

  Main benefit: eliminates redundant K,V PROJECTION computation (O(T d) savings).
    """)

    # -----------------------------------------------------------------------
    section("CACHE GROWTH OVER GENERATION")
    # -----------------------------------------------------------------------
    print(f"\n  KV cache grows linearly with generated tokens.")
    print(f"  Model: L=32, h=32, d_k=128, float16 (2 bytes), batch=1")
    L2, h2, dk2 = 32, 32, 128
    print(f"\n  {'Step':>6} {'Cache size (MB)':>18} {'% of 80GB GPU RAM':>22}")
    for t in [1, 64, 256, 512, 1024, 2048, 4096, 8192]:
        mb = kv_cache_bytes(1, t, L2, h2, dk2) / 1e6
        pct = mb / 80000 * 100
        bar = "█" * int(pct * 2)
        print(f"  {t:>6} {mb:>18.1f} {pct:>16.3f}%  {bar}")


if __name__ == "__main__":
    main()
