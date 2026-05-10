"""
Inference-only Transformer encoder — pure NumPy, no autograd.
Covers: full encoder forward pass (embed + PE + MHA + FFN + LN),
        parameter loading from PyTorch, numerical equivalence check.
pip install numpy torch  (torch optional — only needed for equivalence check)
"""

import numpy as np
import math

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# NumPy ops
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    """x: (..., d) — normalise over last dim."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def relu(x):
    return np.maximum(0.0, x)


# ---------------------------------------------------------------------------
# Sinusoidal PE
# ---------------------------------------------------------------------------

def sinusoidal_pe(max_len, d_model):
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[:d_model // 2])
    return pe


# ---------------------------------------------------------------------------
# MHA (NumPy)
# ---------------------------------------------------------------------------

def mha_forward(x, WQ, WK, WV, WO, n_heads, mask=None):
    """
    x  : (batch, seq, d_model)
    W* : (d_model, d_model)  [no bias]
    Returns: (batch, seq, d_model)
    """
    B, T, D = x.shape
    d_k = D // n_heads

    def project_split(inp, W):
        out = inp @ W                              # (B, T, D)
        out = out.reshape(B, T, n_heads, d_k)
        return out.transpose(0, 2, 1, 3)           # (B, h, T, d_k)

    Q = project_split(x, WQ)
    K = project_split(x, WK)
    V = project_split(x, WV)

    scores = Q @ K.swapaxes(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    W = softmax(scores, axis=-1)
    out = W @ V                                    # (B, h, T, d_k)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
    return out @ WO


# ---------------------------------------------------------------------------
# FFN (NumPy)
# ---------------------------------------------------------------------------

def ffn_forward(x, W1, b1, W2, b2):
    """x: (B, T, d_model)"""
    return relu(x @ W1 + b1) @ W2 + b2


# ---------------------------------------------------------------------------
# Encoder Layer (NumPy)
# ---------------------------------------------------------------------------

def encoder_layer_forward(x, weights, layer_idx, n_heads):
    """
    weights dict keys (for layer i):
      encoder.layers.{i}.norm1.{weight,bias}
      encoder.layers.{i}.norm2.{weight,bias}
      encoder.layers.{i}.self_attn.W{Q,K,V,O}.weight
      encoder.layers.{i}.ffn.net.{0,3}.{weight,bias}
    """
    p = f"encoder.layers.{layer_idx}"
    # Pre-LN + self-attention
    x_n = layer_norm(x,
                     weights[f"{p}.norm1.weight"],
                     weights[f"{p}.norm1.bias"])
    attn_out = mha_forward(x_n,
                           weights[f"{p}.self_attn.WQ.weight"].T,
                           weights[f"{p}.self_attn.WK.weight"].T,
                           weights[f"{p}.self_attn.WV.weight"].T,
                           weights[f"{p}.self_attn.WO.weight"].T,
                           n_heads)
    x = x + attn_out
    # Pre-LN + FFN
    x_n = layer_norm(x,
                     weights[f"{p}.norm2.weight"],
                     weights[f"{p}.norm2.bias"])
    ffn_out = ffn_forward(x_n,
                          weights[f"{p}.ffn.net.0.weight"].T,
                          weights[f"{p}.ffn.net.0.bias"],
                          weights[f"{p}.ffn.net.3.weight"].T,
                          weights[f"{p}.ffn.net.3.bias"])
    x = x + ffn_out
    return x


# ---------------------------------------------------------------------------
# Full NumPy Encoder
# ---------------------------------------------------------------------------

class NumPyEncoder:
    def __init__(self, weights: dict, n_layers: int, n_heads: int, d_model: int):
        self.weights = weights
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.pe = sinusoidal_pe(512, d_model)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        token_ids: (batch, seq)  — integer token indices
        Returns  : (batch, seq, d_model)
        """
        # Embedding lookup
        E = self.weights["encoder.embed.weight"]    # (vocab, d_model)
        x = E[token_ids] * math.sqrt(self.d_model)
        # Add PE
        x = x + self.pe[np.newaxis, :x.shape[1], :]
        # Encoder layers
        for i in range(self.n_layers):
            x = encoder_layer_forward(x, self.weights, i, self.n_heads)
        # Final LayerNorm
        x = layer_norm(x,
                       self.weights["encoder.norm.weight"],
                       self.weights["encoder.norm.bias"])
        return x


# ---------------------------------------------------------------------------
# Weight extraction from PyTorch model
# ---------------------------------------------------------------------------

def extract_weights(model) -> dict:
    """Extract named parameters as NumPy arrays."""
    return {name: param.detach().numpy() for name, param in model.named_parameters()}


# ---------------------------------------------------------------------------
# Standalone demo weights (random, reproducible)
# ---------------------------------------------------------------------------

def make_demo_weights(vocab_size, d_model, d_ff, n_layers, n_heads):
    rng = np.random.default_rng(42)
    s = 1.0 / math.sqrt(d_model)
    w = {}
    w["encoder.embed.weight"] = rng.uniform(-s, s, (vocab_size, d_model))
    for i in range(n_layers):
        p = f"encoder.layers.{i}"
        w[f"{p}.norm1.weight"] = np.ones(d_model)
        w[f"{p}.norm1.bias"]   = np.zeros(d_model)
        w[f"{p}.norm2.weight"] = np.ones(d_model)
        w[f"{p}.norm2.bias"]   = np.zeros(d_model)
        for proj in ["WQ", "WK", "WV", "WO"]:
            w[f"{p}.self_attn.{proj}.weight"] = rng.uniform(-s, s, (d_model, d_model))
        w[f"{p}.ffn.net.0.weight"] = rng.uniform(-s, s, (d_ff, d_model))
        w[f"{p}.ffn.net.0.bias"]   = np.zeros(d_ff)
        w[f"{p}.ffn.net.3.weight"] = rng.uniform(-s, s, (d_model, d_ff))
        w[f"{p}.ffn.net.3.bias"]   = np.zeros(d_model)
    w["encoder.norm.weight"] = np.ones(d_model)
    w["encoder.norm.bias"]   = np.zeros(d_model)
    return w


def main():
    VOCAB_SIZE = 100
    D_MODEL = 64
    D_FF = 128
    N_LAYERS = 2
    N_HEADS = 4

    section("NUMPY TRANSFORMER ENCODER — FORWARD PASS")
    weights = make_demo_weights(VOCAB_SIZE, D_MODEL, D_FF, N_LAYERS, N_HEADS)
    encoder = NumPyEncoder(weights, N_LAYERS, N_HEADS, D_MODEL)

    rng = np.random.default_rng(0)
    B, T = 2, 10
    token_ids = rng.integers(4, VOCAB_SIZE, (B, T))
    out = encoder.forward(token_ids)
    print(f"\n  Input token_ids : {token_ids.shape}")
    print(f"  Encoder output  : {out.shape}")
    print(f"  Output finite   : {np.all(np.isfinite(out))}")
    print(f"  Output mean     : {out.mean():.6f}")
    print(f"  Output std      : {out.std():.6f}")

    section("LAYER NORM VERIFICATION")
    x_test = rng.standard_normal((2, 5, D_MODEL))
    g = np.ones(D_MODEL)
    b = np.zeros(D_MODEL)
    x_ln = layer_norm(x_test, g, b)
    print(f"\n  LayerNorm output stats (should be mean≈0, std≈1 per sample):")
    for bi in range(min(2, B)):
        for ti in range(min(3, T)):
            m = x_ln[bi, ti].mean()
            s = x_ln[bi, ti].std()
            print(f"    batch={bi}, pos={ti}:  mean={m:.4f}  std={s:.4f}")

    section("SINUSOIDAL PE PROPERTIES")
    pe = sinusoidal_pe(20, D_MODEL)
    print(f"\n  PE shape: {pe.shape}")
    print(f"  PE range: [{pe.min():.4f}, {pe.max():.4f}]")
    print(f"\n  Dot product between positions (should decrease with distance):")
    ref = pe[0] @ pe[0]
    for k in [0, 1, 2, 5, 10]:
        if k < pe.shape[0]:
            dot = pe[0] @ pe[k]
            print(f"    pos 0 · pos {k:2d}: {dot:.4f}  (relative: {dot/ref:.4f})")

    section("PYTORCH EQUIVALENCE CHECK")
    try:
        import torch
        import torch.nn as nn
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from model import Transformer, Encoder

        torch.manual_seed(42)
        pt_model = Transformer(
            src_vocab=VOCAB_SIZE, tgt_vocab=VOCAB_SIZE,
            d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
            n_layers=N_LAYERS, max_len=128, dropout=0.0,
        )
        pt_model.eval()

        # Extract weights and run NumPy encoder
        pt_weights = extract_weights(pt_model)
        np_encoder = NumPyEncoder(pt_weights, N_LAYERS, N_HEADS, D_MODEL)

        src_pt = torch.from_numpy(token_ids).long()
        with torch.no_grad():
            pt_enc_out = pt_model.encoder(src_pt).numpy()

        np_enc_out = np_encoder.forward(token_ids)

        max_diff = np.abs(pt_enc_out - np_enc_out).max()
        mean_diff = np.abs(pt_enc_out - np_enc_out).mean()
        print(f"\n  PyTorch encoder output shape : {pt_enc_out.shape}")
        print(f"  NumPy   encoder output shape : {np_enc_out.shape}")
        print(f"  Max absolute difference      : {max_diff:.2e}")
        print(f"  Mean absolute difference     : {mean_diff:.2e}")
        match = "PASS" if max_diff < 1e-4 else "DIFF (check PE/LN implementation)"
        print(f"  Equivalence check            : {match}")

    except ImportError as e:
        print(f"\n  PyTorch not available for equivalence check: {e}")
        print("  NumPy encoder ran successfully with random weights above.")

    section("ATTENTION PATTERN INSPECTION")
    # Run one MHA step manually and show attention weights
    x_small = rng.standard_normal((1, 6, D_MODEL))
    WQ = weights["encoder.layers.0.self_attn.WQ.weight"].T
    WK = weights["encoder.layers.0.self_attn.WK.weight"].T
    WV = weights["encoder.layers.0.self_attn.WV.weight"].T
    WO = weights["encoder.layers.0.self_attn.WO.weight"].T
    d_k = D_MODEL // N_HEADS
    Q = (x_small @ WQ).reshape(1, 6, N_HEADS, d_k).transpose(0, 2, 1, 3)
    K = (x_small @ WK).reshape(1, 6, N_HEADS, d_k).transpose(0, 2, 1, 3)
    scores = Q @ K.swapaxes(-2, -1) / math.sqrt(d_k)
    attn_w = softmax(scores, axis=-1)
    print(f"\n  Attention weights [head 0, batch 0]  shape={attn_w.shape}")
    print(f"  (rows = query positions, cols = key positions)")
    for row in attn_w[0, 0]:
        print("   ", " ".join(f"{v:.3f}" for v in row))
    print(f"\n  Row sums (should all be 1.0): {attn_w[0, 0].sum(axis=-1).round(4)}")

    section("EFFICIENCY: NUMPY vs PYTORCH (INFERENCE)")
    import time
    B_bench, T_bench = 4, 32
    ids_bench = rng.integers(4, VOCAB_SIZE, (B_bench, T_bench))
    n_runs = 20

    t0 = time.perf_counter()
    for _ in range(n_runs):
        encoder.forward(ids_bench)
    t_np = (time.perf_counter() - t0) / n_runs * 1000

    print(f"\n  NumPy encoder  ({B_bench}×{T_bench}, {N_LAYERS} layers): {t_np:.2f} ms/call")
    try:
        import torch
        pt_enc = pt_model.encoder if 'pt_model' in dir() else None
        if pt_enc is not None:
            ids_pt = torch.from_numpy(ids_bench).long()
            with torch.no_grad():
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    pt_enc(ids_pt)
                t_pt = (time.perf_counter() - t0) / n_runs * 1000
            print(f"  PyTorch encoder ({B_bench}×{T_bench}, {N_LAYERS} layers): {t_pt:.2f} ms/call")
            print(f"  (NumPy overhead is expected — no fused ops, no BLAS tuning)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
