"""
Positional encoding — pure NumPy.
Covers: sinusoidal PE (Vaswani 2017), learned PE, RoPE intuition,
        frequency spectrum, relative position properties.
pip install numpy
"""

import numpy as np
import math

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Sinusoidal PE
# ---------------------------------------------------------------------------

def sinusoidal_pe(max_len, d_model):
    """
    PE_{(pos, 2i)}   = sin(pos / 10000^(2i / d_model))
    PE_{(pos, 2i+1)} = cos(pos / 10000^(2i / d_model))

    Returns: (max_len, d_model)
    """
    positions = np.arange(max_len)[:, np.newaxis]          # (L, 1)
    dims = np.arange(0, d_model, 2)[np.newaxis, :]         # (1, d_model/2)
    angles = positions / (10000.0 ** (dims / d_model))      # (L, d_model/2)

    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(angles)   # even dims
    pe[:, 1::2] = np.cos(angles[:, :d_model // 2])   # odd dims
    return pe


# ---------------------------------------------------------------------------
# Learned PE
# ---------------------------------------------------------------------------

class LearnedPE:
    """Trainable embedding table E[pos] ∈ R^d_model."""

    def __init__(self, max_len, d_model, rng):
        scale = 1.0 / math.sqrt(d_model)
        self.E = rng.uniform(-scale, scale, (max_len, d_model))

    def __call__(self, positions):
        """positions: array of int indices"""
        return self.E[positions]

    def param_count(self):
        return self.E.size


# ---------------------------------------------------------------------------
# RoPE — Rotary Positional Embedding (intuition + 2D demo)
# ---------------------------------------------------------------------------

def rope_apply_2d(q, k, pos_q, pos_k, theta=10000.0):
    """
    Apply RoPE to 2D query/key vectors (d=2 case for illustration).
    Rotation angle: θ_m = m / theta^(2i/d) for dimension pair i.
    For d=2: single rotation by angle pos / theta.
    """
    angle_q = pos_q / theta
    angle_k = pos_k / theta

    R = lambda angle: np.array([[math.cos(angle), -math.sin(angle)],
                                 [math.sin(angle),  math.cos(angle)]])
    q_rot = R(angle_q) @ q
    k_rot = R(angle_k) @ k
    return q_rot, k_rot


def rope_dot_product(q, k, pos_q, pos_k, theta=10000.0):
    """RoPE property: dot product depends only on relative offset."""
    q_rot, k_rot = rope_apply_2d(q, k, pos_q, pos_k, theta)
    return float(q_rot @ k_rot)


# ---------------------------------------------------------------------------
# Relative position analysis
# ---------------------------------------------------------------------------

def relative_position_dot_products(pe, max_offset=5):
    """
    Check: PE_{pos}^T PE_{pos+k} as a function of k only (not pos).
    If true → sinusoidal PE implicitly encodes relative positions.
    """
    L = pe.shape[0]
    results = {}
    for k in range(max_offset + 1):
        dots = []
        for pos in range(0, min(L - k, 20)):
            dots.append(float(pe[pos] @ pe[pos + k]))
        results[k] = (np.mean(dots), np.std(dots))
    return results


# ---------------------------------------------------------------------------
# PE visualisation (text-based heatmap)
# ---------------------------------------------------------------------------

def ascii_heatmap(M, title, rows=8, cols=16):
    """Print a small heatmap using ASCII block characters."""
    chars = " ░▒▓█"
    m = M[:rows, :cols]
    vmin, vmax = m.min(), m.max()
    print(f"\n  {title}  [{rows}×{cols} excerpt, range [{vmin:.2f}, {vmax:.2f}]]")
    for row in m:
        line = "  "
        for v in row:
            idx = int((v - vmin) / (vmax - vmin + 1e-9) * (len(chars) - 1))
            line += chars[idx] * 2
        print(line)


def main():
    rng = np.random.default_rng(42)
    max_len, d_model = 64, 32

    # -----------------------------------------------------------------------
    section("SINUSOIDAL POSITIONAL ENCODING")
    # -----------------------------------------------------------------------
    pe_sin = sinusoidal_pe(max_len, d_model)
    print(f"\n  PE shape: {pe_sin.shape}   (max_len, d_model)")
    print(f"  Range  : [{pe_sin.min():.4f}, {pe_sin.max():.4f}]  (guaranteed in [-1, 1] by sin/cos)")
    print(f"\n  First 4 positions, first 8 dims:")
    print(f"  {'pos':>4}", " ".join(f"dim{i:02d}" for i in range(8)))
    for pos in range(4):
        print(f"  {pos:>4}", " ".join(f"{pe_sin[pos, d]:+.3f}" for d in range(8)))

    ascii_heatmap(pe_sin, "PE heatmap (pos=row, dim=col)")

    # -----------------------------------------------------------------------
    section("FREQUENCY SPECTRUM")
    # -----------------------------------------------------------------------
    print(f"\n  Wavelength λ_i = 2π × 10000^(2i/d_model)")
    print(f"  {'dim pair i':>12} {'wavelength λ':>14} {'period (tokens)':>16}")
    for i in [0, 1, 2, 4, 8, d_model // 2 - 1]:
        lam = 2 * math.pi * (10000 ** (2 * i / d_model))
        print(f"  {i:>12} {lam:>14.1f} {lam / (2 * math.pi):>16.1f}")
    print(f"\n  Low dims change every ~{2*math.pi:.1f} tokens (high frequency).")
    print(f"  High dims change over ~{2*math.pi * 10000**(1.0):.0f} tokens (low frequency).")
    print(f"  Together they provide unique encoding up to max_len={max_len}.")

    # -----------------------------------------------------------------------
    section("RELATIVE POSITION PROPERTY")
    # -----------------------------------------------------------------------
    rels = relative_position_dot_products(pe_sin, max_offset=6)
    print("\n  PE[pos]^T @ PE[pos+k]  — does it depend only on k?")
    print(f"  {'offset k':>10} {'mean dot':>12} {'std (↓ = position-invariant)':>32}")
    for k, (mean, std) in rels.items():
        bar = "█" * int(std * 100)
        print(f"  {k:>10} {mean:>12.4f} {std:>10.6f}  {bar}")
    print("\n  Low std → PE[pos]^T PE[pos+k] is approximately a function of k only.")
    print("  This gives the model implicit relative position information.")

    # -----------------------------------------------------------------------
    section("LEARNED POSITIONAL ENCODING")
    # -----------------------------------------------------------------------
    learned_pe = LearnedPE(max_len, d_model, rng)
    print(f"\n  Learned PE embedding table shape: {learned_pe.E.shape}")
    print(f"  Parameter count: {learned_pe.param_count()} floats")
    print(f"  Sinusoidal PE parameter count: 0 (deterministic formula)")

    # Compare: cosine similarity between adjacent positions
    print(f"\n  Cosine similarity between adjacent positions (sin vs learned):")
    print(f"  {'pos':>5} {'sin_cos_sim':>14} {'learned_cos_sim':>18}")
    for pos in range(0, min(max_len - 1, 8)):
        def cos_sim(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        s = cos_sim(pe_sin[pos], pe_sin[pos + 1])
        l = cos_sim(learned_pe.E[pos], learned_pe.E[pos + 1])
        print(f"  {pos:>5} {s:>14.4f} {l:>18.4f}")
    print("\n  Sinusoidal: adjacent positions highly similar (smooth).")
    print("  Learned: random init, no structure until trained.")

    # -----------------------------------------------------------------------
    section("SINUSOIDAL vs LEARNED — COMPARISON TABLE")
    # -----------------------------------------------------------------------
    print("""
  ┌──────────────────────────┬──────────────────┬──────────────────┐
  │ Property                 │ Sinusoidal        │ Learned          │
  ├──────────────────────────┼──────────────────┼──────────────────┤
  │ Parameters               │ 0                 │ L_max × d_model  │
  │ Extrapolates > L_max     │ Yes (imperfectly) │ No (OOV)         │
  │ Relative position info   │ Implicit          │ Must be learned  │
  │ Training signal needed   │ No                │ Yes              │
  │ Smoothness               │ Built-in          │ Random init      │
  │ Used in                  │ Transformer 2017  │ BERT, GPT-2      │
  │ Max position seen        │ Unlimited (math)  │ L_max            │
  └──────────────────────────┴──────────────────┴──────────────────┘
    """)

    # -----------------------------------------------------------------------
    section("ROPE — ROTARY POSITIONAL EMBEDDING (INTUITION)")
    # -----------------------------------------------------------------------
    print("\n  RoPE rotates query/key vectors by angle proportional to position.")
    print("  Key property: q_pos @ k_pos' depends only on (q, k, pos - pos').")
    print("\n  2D illustration (d=2, single rotation):")
    q = np.array([1.0, 0.0])
    k = np.array([1.0, 0.0])
    print(f"\n  q = {q}, k = {k}")
    print(f"  {'pos_q':>6} {'pos_k':>6} {'dot':>10} {'pos_q-pos_k':>14}")
    for pos_q, pos_k in [(0,0), (1,0), (2,0), (3,0), (5,2), (10,7)]:
        dot = rope_dot_product(q, k, pos_q, pos_k)
        print(f"  {pos_q:>6} {pos_k:>6} {dot:>10.4f} {pos_q-pos_k:>14}")
    print("\n  Same relative offset (pos_q - pos_k) → same dot product.")
    print("  RoPE directly encodes relative positions in the attention score.")

    print("\n  RoPE formula (per 2D subspace i):")
    print("    Rotate (q_{2i}, q_{2i+1}) by angle pos/theta^(2i/d)")
    print("    θ_i = 1/10000^(2i/d) — same base as sinusoidal PE")
    print("  Advantage: no added positional vector, position encoded in rotation.")
    print("  Used in: LLaMA 1/2/3, GPT-NeoX, Falcon, Mistral.")

    # -----------------------------------------------------------------------
    section("HOW PE IS ADDED TO INPUT EMBEDDINGS")
    # -----------------------------------------------------------------------
    X = rng.standard_normal((1, max_len, d_model))    # (batch, seq, d_model)
    pe_broadcast = pe_sin[np.newaxis, :max_len, :]    # (1, seq, d_model)
    X_pe = X + pe_broadcast
    print(f"\n  X (token embeddings) : {X.shape}")
    print(f"  PE (sinusoidal)      : {pe_broadcast.shape}")
    print(f"  X + PE               : {X_pe.shape}")
    print(f"\n  Addition adds positional info without changing dimensionality.")
    print(f"  During training, the model learns to use the PE signal in attention.")
    print(f"\n  For learned PE:")
    positions = np.arange(max_len)
    pe_learned_seq = learned_pe(positions)       # (seq, d_model)
    print(f"  PE lookup result shape: {pe_learned_seq.shape}")
    X_learned = X[0] + pe_learned_seq
    print(f"  X + learned PE: {X_learned.shape}")


if __name__ == "__main__":
    main()
