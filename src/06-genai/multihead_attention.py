"""
Multi-head attention from scratch — pure NumPy.
Covers: projection matrices W^Q/K/V/O, head splitting, head concatenation,
        MQA, GQA, parameter count analysis, head pattern diversity.
pip install numpy
"""

import numpy as np
import math

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K: (..., seq, d_k)
    V:    (..., seq, d_v)
    Returns: (..., seq, d_v)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    w = softmax(scores, axis=-1)
    return w @ V, w


class MultiHeadAttention:
    """
    Standard MHA: each head has its own W^Q, W^K, W^V.

    d_model : model dimension
    n_heads : number of attention heads
    d_k     : key/query dimension per head  (default: d_model // n_heads)
    d_v     : value dimension per head      (default: d_model // n_heads)
    """

    def __init__(self, d_model, n_heads, rng, d_k=None, d_v=None):
        self.d_model = d_model
        self.h = n_heads
        self.d_k = d_k or d_model // n_heads
        self.d_v = d_v or d_model // n_heads
        scale = 1.0 / math.sqrt(d_model)

        # Projection matrices
        # W^Q, W^K, W^V for all heads stacked: (d_model, h * d_k)
        self.WQ = rng.uniform(-scale, scale, (d_model, n_heads * self.d_k))
        self.WK = rng.uniform(-scale, scale, (d_model, n_heads * self.d_k))
        self.WV = rng.uniform(-scale, scale, (d_model, n_heads * self.d_v))
        # Output projection: (h * d_v, d_model)
        self.WO = rng.uniform(-scale, scale, (n_heads * self.d_v, d_model))

    def param_count(self):
        return (self.WQ.size + self.WK.size + self.WV.size + self.WO.size)

    def forward(self, X, mask=None):
        """
        X    : (batch, seq, d_model)
        mask : (seq, seq) or (batch, seq, seq)
        Returns: (batch, seq, d_model)
        """
        batch, seq, _ = X.shape

        # Project to Q, K, V for all heads simultaneously
        Q = X @ self.WQ   # (batch, seq, h * d_k)
        K = X @ self.WK
        V = X @ self.WV   # (batch, seq, h * d_v)

        # Split into individual heads: (batch, h, seq, d_k)
        Q = Q.reshape(batch, seq, self.h, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq, self.h, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq, self.h, self.d_v).transpose(0, 2, 1, 3)

        # Attention per head
        head_out, weights = scaled_dot_product_attention(Q, K, V, mask)
        # head_out: (batch, h, seq, d_v)

        # Concatenate heads: (batch, seq, h * d_v)
        head_out = head_out.transpose(0, 2, 1, 3).reshape(batch, seq, self.h * self.d_v)

        # Output projection
        output = head_out @ self.WO  # (batch, seq, d_model)
        return output, weights


class MultiQueryAttention:
    """
    MQA: all heads share a single W^K, W^V pair.
    Reduces KV cache size by factor n_heads.
    """

    def __init__(self, d_model, n_heads, rng, d_k=None, d_v=None):
        self.d_model = d_model
        self.h = n_heads
        self.d_k = d_k or d_model // n_heads
        self.d_v = d_v or d_model // n_heads
        scale = 1.0 / math.sqrt(d_model)

        self.WQ = rng.uniform(-scale, scale, (d_model, n_heads * self.d_k))
        # Single K, V projection (not per-head)
        self.WK = rng.uniform(-scale, scale, (d_model, self.d_k))
        self.WV = rng.uniform(-scale, scale, (d_model, self.d_v))
        self.WO = rng.uniform(-scale, scale, (n_heads * self.d_v, d_model))

    def param_count(self):
        return self.WQ.size + self.WK.size + self.WV.size + self.WO.size

    def forward(self, X, mask=None):
        batch, seq, _ = X.shape
        Q = X @ self.WQ   # (batch, seq, h * d_k)
        K = X @ self.WK   # (batch, seq, d_k)    — shared
        V = X @ self.WV   # (batch, seq, d_v)    — shared

        Q = Q.reshape(batch, seq, self.h, self.d_k).transpose(0, 2, 1, 3)
        # Broadcast K, V across heads
        K = K[:, np.newaxis, :, :]           # (batch, 1, seq, d_k)
        V = V[:, np.newaxis, :, :]           # (batch, 1, seq, d_v)

        head_out, weights = scaled_dot_product_attention(Q, K, V, mask)
        head_out = head_out.transpose(0, 2, 1, 3).reshape(batch, seq, self.h * self.d_v)
        output = head_out @ self.WO
        return output, weights


class GroupedQueryAttention:
    """
    GQA: n_kv_heads groups, each sharing W^K, W^V across (n_heads // n_kv_heads) heads.
    Interpolates between MHA (n_kv_heads=n_heads) and MQA (n_kv_heads=1).
    Used in LLaMA 2/3, Mistral.
    """

    def __init__(self, d_model, n_heads, n_kv_heads, rng, d_k=None, d_v=None):
        assert n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.h = n_heads
        self.g = n_kv_heads
        self.queries_per_kv = n_heads // n_kv_heads
        self.d_k = d_k or d_model // n_heads
        self.d_v = d_v or d_model // n_heads
        scale = 1.0 / math.sqrt(d_model)

        self.WQ = rng.uniform(-scale, scale, (d_model, n_heads * self.d_k))
        self.WK = rng.uniform(-scale, scale, (d_model, n_kv_heads * self.d_k))
        self.WV = rng.uniform(-scale, scale, (d_model, n_kv_heads * self.d_v))
        self.WO = rng.uniform(-scale, scale, (n_heads * self.d_v, d_model))

    def param_count(self):
        return self.WQ.size + self.WK.size + self.WV.size + self.WO.size

    def forward(self, X, mask=None):
        batch, seq, _ = X.shape
        Q = X @ self.WQ   # (batch, seq, h * d_k)
        K = X @ self.WK   # (batch, seq, g * d_k)
        V = X @ self.WV   # (batch, seq, g * d_v)

        Q = Q.reshape(batch, seq, self.h, self.d_k).transpose(0, 2, 1, 3)
        # (batch, g, seq, d_k) → repeat to match h heads
        K = K.reshape(batch, seq, self.g, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq, self.g, self.d_v).transpose(0, 2, 1, 3)
        # Expand: each KV group covers (h//g) query heads
        K = np.repeat(K, self.queries_per_kv, axis=1)
        V = np.repeat(V, self.queries_per_kv, axis=1)

        head_out, weights = scaled_dot_product_attention(Q, K, V, mask)
        head_out = head_out.transpose(0, 2, 1, 3).reshape(batch, seq, self.h * self.d_v)
        output = head_out @ self.WO
        return output, weights


def main():
    rng = np.random.default_rng(42)
    batch, seq, d_model, n_heads = 2, 8, 64, 4

    # -----------------------------------------------------------------------
    section("MULTI-HEAD ATTENTION — FORWARD PASS")
    # -----------------------------------------------------------------------
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, rng=rng)
    X = rng.standard_normal((batch, seq, d_model))

    out, weights = mha.forward(X)
    print(f"\n  Input   : {X.shape}    (batch, seq, d_model)")
    print(f"  Output  : {out.shape}    (batch, seq, d_model)")
    print(f"  Weights : {weights.shape}  (batch, n_heads, seq, seq)")
    print(f"\n  All output values finite: {np.all(np.isfinite(out))}")
    print(f"  All weight rows sum to 1: {np.allclose(weights.sum(axis=-1), 1.0)}")

    # -----------------------------------------------------------------------
    section("PROJECTION MATRICES — SHAPES AND PURPOSE")
    # -----------------------------------------------------------------------
    d_k = d_model // n_heads
    print(f"\n  d_model={d_model}, n_heads={n_heads}, d_k=d_v={d_k}")
    print(f"\n  {'Matrix':<6} {'Shape':<22} {'Purpose'}")
    print(f"  {'W^Q':<6} ({d_model} × {n_heads * d_k}){'':<10} projects input to Q space for all heads")
    print(f"  {'W^K':<6} ({d_model} × {n_heads * d_k}){'':<10} projects input to K space for all heads")
    print(f"  {'W^V':<6} ({d_model} × {n_heads * d_k}){'':<10} projects input to V space for all heads")
    print(f"  {'W^O':<6} ({n_heads * d_k} × {d_model}){'':<10} projects concatenated heads back to d_model")
    print(f"\n  After projection, split into {n_heads} heads of dim d_k={d_k}")
    print(f"  Each head runs attention in a {d_k}-dim subspace independently")

    # -----------------------------------------------------------------------
    section("PARAMETER COUNT COMPARISON: MHA vs MQA vs GQA")
    # -----------------------------------------------------------------------
    mqa = MultiQueryAttention(d_model=d_model, n_heads=n_heads, rng=rng)
    gqa_2 = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=2, rng=rng)
    gqa_1 = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=1, rng=rng)

    print(f"\n  d_model={d_model}, n_heads={n_heads}")
    print(f"  {'Method':<22} {'Params':>8}  {'KV cache (per token)':>22}  {'Reduction'}")
    mha_kv_params = mha.WK.size + mha.WV.size
    for name, m, n_kv in [("MHA (n_kv=n_heads)", mha, n_heads),
                           ("GQA (n_kv=2)",       gqa_2, 2),
                           ("MQA (n_kv=1)",        mqa, 1)]:
        kv_per_tok = 2 * n_kv * d_k  # K + V, n_kv groups, per token
        red = mha_kv_params // max(1, (n_kv * d_k * 2))
        print(f"  {name:<22} {m.param_count():>8}  {kv_per_tok:>22} floats  {n_heads // n_kv}×")

    # -----------------------------------------------------------------------
    section("HEAD PATTERN DIVERSITY")
    # -----------------------------------------------------------------------
    print("\n  Different heads attend to different positions.")
    print(f"  Showing attention weights for token 0 across all {n_heads} heads (batch=0):")
    print(f"  (rows = query positions, shown for query position 0)")
    print()
    for h in range(n_heads):
        row = weights[0, h, 0, :]   # query pos 0, head h
        print(f"    head {h}: " + " ".join(f"{v:.3f}" for v in row))
    print("\n  Heads differ in which keys they upweight.")

    # -----------------------------------------------------------------------
    section("CAUSAL MASK WITH MHA")
    # -----------------------------------------------------------------------
    mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    out_c, w_c = mha.forward(X, mask=mask)
    print(f"\n  With causal mask:")
    print(f"  Output shape: {out_c.shape}")
    upper_sum = w_c[:, :, :, :].sum(axis=-1)   # row sums should still be 1
    print(f"  All rows sum to 1: {np.allclose(w_c.sum(axis=-1), 1.0)}")
    # check no future attention: w_c[b,h,i,j] should be 0 for j > i
    future_leak = (w_c[:, :, np.arange(seq)[:, None] < np.arange(seq)[None, :]]).max()
    print(f"  Max attention to future positions: {future_leak:.2e}  (should be ≈ 0)")

    # -----------------------------------------------------------------------
    section("MQA AND GQA — FORWARD PASS VALIDATION")
    # -----------------------------------------------------------------------
    out_mqa, _ = mqa.forward(X)
    out_gqa, _ = gqa_2.forward(X)
    print(f"\n  All methods produce output shape (batch, seq, d_model):")
    print(f"    MHA : {out.shape}    values finite: {np.all(np.isfinite(out))}")
    print(f"    MQA : {out_mqa.shape}    values finite: {np.all(np.isfinite(out_mqa))}")
    print(f"    GQA : {out_gqa.shape}    values finite: {np.all(np.isfinite(out_gqa))}")
    print("\n  Note: outputs differ (different weights) — same architecture, different params.")

    # -----------------------------------------------------------------------
    section("ATTENTION IN TRANSFORMERS: WHAT EACH TYPE DOES")
    # -----------------------------------------------------------------------
    print("""
  Self-attention (encoder):
    Q = K = V = encoder hidden states
    All positions attend to all others — full bidirectional context

  Causal self-attention (decoder):
    Q = K = V = decoder hidden states + causal mask
    Position i only attends to positions 0..i — autoregressive

  Cross-attention (encoder-decoder):
    Q = decoder hidden states
    K = V = encoder output
    Each decoder position attends over all encoder positions

  In GPT (decoder-only):
    Only causal self-attention blocks — no cross-attention

  In T5, original Transformer:
    Encoder: bidirectional self-attention
    Decoder: causal self-attention + cross-attention to encoder
    """)


if __name__ == "__main__":
    main()
