"""
Scaled dot-product attention from scratch — pure NumPy.
Covers: similarity scores, scaling, softmax, causal masking, cross-attention,
        complexity analysis, numerical stability.
pip install numpy
"""

import numpy as np
import math

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Numerically stable softmax (row-wise)
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    """Subtract max before exp to avoid overflow."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Scaled dot-product attention
# ---------------------------------------------------------------------------

def attention(Q, K, V, mask=None):
    """
    Q : (batch, n_q, d_k)
    K : (batch, n_k, d_k)
    V : (batch, n_k, d_v)
    mask : (batch, n_q, n_k) or (n_q, n_k)  — True means MASK OUT (set to -inf)
    Returns: (output (batch, n_q, d_v), weights (batch, n_q, n_k))
    """
    d_k = Q.shape[-1]
    # (batch, n_q, n_k)
    scores = Q @ K.swapaxes(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------

def causal_mask(seq_len):
    """True at positions (i, j) where j > i (future positions)."""
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------

def print_matrix(name, M, decimals=3):
    print(f"\n  {name}  shape={M.shape}")
    if M.ndim == 2:
        for row in M:
            print("   ", " ".join(f"{v:+.{decimals}f}" for v in row))
    elif M.ndim == 3:
        for b, batch_m in enumerate(M):
            print(f"  [batch {b}]")
            for row in batch_m:
                print("   ", " ".join(f"{v:+.{decimals}f}" for v in row))


def main():
    rng = np.random.default_rng(42)

    # -----------------------------------------------------------------------
    section("SCALED DOT-PRODUCT ATTENTION — STEP BY STEP")
    # -----------------------------------------------------------------------
    n_q, n_k, d_k, d_v = 3, 4, 6, 8
    Q = rng.standard_normal((n_q, d_k))
    K = rng.standard_normal((n_k, d_k))
    V = rng.standard_normal((n_k, d_v))

    # Step 1: raw similarity scores
    S_raw = Q @ K.T                          # (n_q, n_k)
    print(f"\n  Inputs: Q({n_q},{d_k}), K({n_k},{d_k}), V({n_k},{d_v})")
    print(f"\n  Step 1 — raw scores Q @ K^T  shape={S_raw.shape}")
    print(f"    variance (before scaling) : {S_raw.var():.4f}   (expected ≈ d_k = {d_k})")

    # Step 2: scale
    S_scaled = S_raw / math.sqrt(d_k)
    print(f"\n  Step 2 — scaled by 1/sqrt(d_k={d_k})")
    print(f"    variance (after  scaling) : {S_scaled.var():.4f}   (expected ≈ 1.0)")

    # Step 3: softmax
    W = softmax(S_scaled, axis=-1)
    print(f"\n  Step 3 — row-wise softmax  shape={W.shape}")
    print(f"    rows sum to 1: {np.allclose(W.sum(axis=-1), 1.0)}")
    print(f"    weight matrix (attention pattern):")
    for row in W:
        print("   ", " ".join(f"{v:.3f}" for v in row))

    # Step 4: weighted sum
    out = W @ V
    print(f"\n  Step 4 — output = W @ V  shape={out.shape}")

    # -----------------------------------------------------------------------
    section("WHY SCALE BY 1/sqrt(d_k)?")
    # -----------------------------------------------------------------------
    print("\n  Var[q_i * k_i] = 1 (if both ~ N(0,1))")
    print("  Var[q^T k] = sum of d_k independent terms = d_k")
    print("  Std grows as sqrt(d_k) — large scores → flat or spiky softmax")
    for dk in [1, 4, 16, 64, 256]:
        q = rng.standard_normal((1000, dk))
        k = rng.standard_normal((1000, dk))
        raw = (q * k).sum(axis=1)
        scaled = raw / math.sqrt(dk)
        w_raw = softmax(raw.reshape(1, -1))[0]
        w_scaled = softmax(scaled.reshape(1, -1))[0]
        entropy_raw    = -float((w_raw * np.log(w_raw + 1e-9)).sum())
        entropy_scaled = -float((w_scaled * np.log(w_scaled + 1e-9)).sum())
        print(f"    d_k={dk:4d}  raw_var={raw.var():.1f}  "
              f"entropy_unscaled={entropy_raw:.3f}  entropy_scaled={entropy_scaled:.3f}")
    print("  → Scaled softmax maintains higher entropy (attends to more positions).")

    # -----------------------------------------------------------------------
    section("BATCHED ATTENTION")
    # -----------------------------------------------------------------------
    batch, n, d_k2, d_v2 = 2, 5, 8, 8
    Q_b = rng.standard_normal((batch, n, d_k2))
    K_b = rng.standard_normal((batch, n, d_k2))
    V_b = rng.standard_normal((batch, n, d_v2))
    out_b, W_b = attention(Q_b, K_b, V_b)
    print(f"\n  Batched self-attention:")
    print(f"    Q shape: {Q_b.shape}  K shape: {K_b.shape}  V shape: {V_b.shape}")
    print(f"    Output : {out_b.shape}  Weights: {W_b.shape}")
    print(f"    All attention rows sum to 1: {np.allclose(W_b.sum(axis=-1), 1.0)}")

    # -----------------------------------------------------------------------
    section("CAUSAL (MASKED) ATTENTION")
    # -----------------------------------------------------------------------
    seq_len = 5
    mask = causal_mask(seq_len)
    print(f"\n  Causal mask (True=masked out) for seq_len={seq_len}:")
    for row in mask:
        print("   ", " ".join(str(int(v)) for v in row))

    Q_c = rng.standard_normal((1, seq_len, d_k2))
    K_c = rng.standard_normal((1, seq_len, d_k2))
    V_c = rng.standard_normal((1, seq_len, d_v2))
    out_c, W_c = attention(Q_c, K_c, V_c, mask=mask)
    print(f"\n  Attention weights with causal mask:")
    for row in W_c[0]:
        vals = " ".join(f"{v:.3f}" for v in row)
        print(f"    [{vals}]")
    print("  → Upper triangle is zero (future positions not attended to).")

    # -----------------------------------------------------------------------
    section("CROSS-ATTENTION (ENCODER-DECODER)")
    # -----------------------------------------------------------------------
    n_dec, n_enc = 3, 6   # decoder query length, encoder key/value length
    Q_dec = rng.standard_normal((1, n_dec, d_k2))
    K_enc = rng.standard_normal((1, n_enc, d_k2))
    V_enc = rng.standard_normal((1, n_enc, d_v2))
    out_cross, W_cross = attention(Q_dec, K_enc, V_enc)
    print(f"\n  Cross-attention:")
    print(f"    Q (decoder): {Q_dec.shape}  K,V (encoder): {K_enc.shape}, {V_enc.shape}")
    print(f"    Attention weights: {W_cross.shape}  (n_dec × n_enc)")
    print(f"    Output: {out_cross.shape}")
    print("  Each decoder position attends over all encoder positions.")

    # -----------------------------------------------------------------------
    section("COMPLEXITY ANALYSIS")
    # -----------------------------------------------------------------------
    print("\n  Complexity of Attention(Q, K, V):")
    print("  ┌────────────────────────────────────────────────────┐")
    print("  │ Operation          │ Time          │ Memory        │")
    print("  ├────────────────────────────────────────────────────┤")
    print("  │ QK^T               │ O(n² d_k)     │ O(n²)         │")
    print("  │ softmax            │ O(n²)         │ O(n²)         │")
    print("  │ AV                 │ O(n² d_v)     │ O(n d_v)      │")
    print("  ├────────────────────────────────────────────────────┤")
    print("  │ Total              │ O(n² d)       │ O(n²)         │")
    print("  └────────────────────────────────────────────────────┘")
    print("  n = sequence length; bottleneck is quadratic in n.")

    for n_seq in [128, 512, 2048, 8192]:
        d = 64
        flops = n_seq * n_seq * d  # QK^T dominant
        attn_mem_mb = (n_seq * n_seq * 4) / 1e6  # float32
        print(f"    n={n_seq:5d}  FLOPs ≈ {flops/1e6:8.1f}M  "
              f"attention matrix ≈ {attn_mem_mb:.2f} MB")

    # -----------------------------------------------------------------------
    section("NUMERICAL STABILITY: STABLE SOFTMAX")
    # -----------------------------------------------------------------------
    print("\n  Naïve softmax overflows for large inputs:")
    x_large = np.array([1000.0, 1001.0, 1002.0])
    naive_exp = np.exp(x_large)
    print(f"    exp([1000, 1001, 1002]) = {naive_exp}")

    print("\n  Stable softmax subtracts max first:")
    stable = softmax(x_large)
    print(f"    stable softmax([1000, 1001, 1002]) = {stable.round(4)}")
    print(f"    sum = {stable.sum():.6f}  (= 1.0)")

    # -----------------------------------------------------------------------
    section("ATTENTION AS INFORMATION RETRIEVAL")
    # -----------------------------------------------------------------------
    print("""
  Analogy to database lookup:
    Query Q  ←→  search query
    Key   K  ←→  database keys (indexed)
    Value V  ←→  database values

  Hard retrieval:   fetch value where key exactly matches query
  Soft retrieval:   weighted sum of ALL values, weights ∝ query-key similarity

  The softmax weight determines "how much" each value contributes.
  High-temperature softmax (before scaling) → uniform retrieval (loses specificity).
  Low-temperature (after scaling) → sharp selection (like hard retrieval).
    """)


if __name__ == "__main__":
    main()
