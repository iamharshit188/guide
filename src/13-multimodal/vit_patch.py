"""
Vision Transformer (ViT) patch embedding from scratch.
Covers: patch extraction, linear projection, [CLS] token, positional
        encoding (learned + sinusoidal), multi-head self-attention,
        full ViT encoder forward pass — pure NumPy.
"""

import numpy as np

rng = np.random.default_rng(42)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Patch Extraction ──────────────────────────────────────────────
def extract_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Reshape (H, W, C) image into (N, patch_size²·C) patch tokens.

    N = (H/P) × (W/P)   where P = patch_size
    Each patch is flattened to a 1-D vector of length P²·C.

    ViT-B/32 on 224×224: N = (224/32)² = 49 patches
    ViT-B/16 on 224×224: N = (224/16)² = 196 patches
    """
    H, W, C = image.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, f"Image {H}×{W} not divisible by P={P}"

    n_h = H // P
    n_w = W // P
    N   = n_h * n_w

    # Reshape: (H, W, C) → (n_h, P, n_w, P, C) → (N, P*P*C)
    patches = image.reshape(n_h, P, n_w, P, C)
    patches = patches.transpose(0, 2, 1, 3, 4)   # (n_h, n_w, P, P, C)
    patches = patches.reshape(N, P * P * C)

    return patches


def patch_embed(patches: np.ndarray, d_model: int,
                W_proj: np.ndarray | None = None) -> np.ndarray:
    """
    Linear projection: (N, P²C) → (N, d_model)
    W_proj shape: (P²C, d_model)
    This is the 'patch embedding' layer in ViT — equivalent to a
    Conv2d(in=C, out=d_model, kernel=P, stride=P).
    """
    patch_dim = patches.shape[-1]
    if W_proj is None:
        W_proj = rng.standard_normal((patch_dim, d_model)) * (patch_dim ** -0.5)
    return patches @ W_proj


# ── Positional Encoding ───────────────────────────────────────────
def sinusoidal_pe_2d(n_h: int, n_w: int, d_model: int) -> np.ndarray:
    """
    2D sinusoidal positional encoding for patch grid.
    Encodes row and column position independently, concatenated.
    Returns shape (n_h*n_w, d_model).
    """
    half = d_model // 2

    def pe_1d(n_pos: int, dim: int) -> np.ndarray:
        positions = np.arange(n_pos)[:, None]           # (n_pos, 1)
        divs = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
        pe = np.zeros((n_pos, dim))
        pe[:, 0::2] = np.sin(positions * divs)
        pe[:, 1::2] = np.cos(positions * divs[:dim//2])
        return pe

    row_pe = pe_1d(n_h, half)   # (n_h, half)
    col_pe = pe_1d(n_w, half)   # (n_w, half)

    # Broadcast: each patch gets row_pe[row] concat col_pe[col]
    row_pe_tiled = np.repeat(row_pe, n_w, axis=0)    # (n_h*n_w, half)
    col_pe_tiled = np.tile(col_pe, (n_h, 1))          # (n_h*n_w, half)
    return np.concatenate([row_pe_tiled, col_pe_tiled], axis=-1)


def learned_pe(n_positions: int, d_model: int) -> np.ndarray:
    """
    Learned positional embeddings: random init, trained via backprop.
    Shape: (n_positions, d_model). In ViT, n_positions = N + 1 (with [CLS]).
    """
    return rng.standard_normal((n_positions, d_model)) * 0.02


# ── [CLS] Token ───────────────────────────────────────────────────
def prepend_cls(patch_tokens: np.ndarray, cls_token: np.ndarray) -> np.ndarray:
    """
    Prepend [CLS] token to sequence.
    patch_tokens: (N, d_model)
    cls_token:    (1, d_model)
    Returns:      (N+1, d_model)
    """
    return np.concatenate([cls_token, patch_tokens], axis=0)


# ── Multi-Head Self-Attention ─────────────────────────────────────
class MultiHeadSelfAttention:
    """
    Standard MHA for ViT encoder.
    Q, K, V projections + output projection.
    Supports both causal (GPT-style) and full (ViT-style) attention.
    """

    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        scale = (d_model) ** -0.5

        self.W_Q = rng.standard_normal((d_model, d_model)) * scale
        self.W_K = rng.standard_normal((d_model, d_model)) * scale
        self.W_V = rng.standard_normal((d_model, d_model)) * scale
        self.W_O = rng.standard_normal((d_model, d_model)) * scale

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """
        x: (seq_len, d_model)
        Returns: (seq_len, d_model)
        """
        seq_len, d = x.shape
        h = self.n_heads
        dh = self.d_head

        Q = x @ self.W_Q   # (seq_len, d_model)
        K = x @ self.W_K
        V = x @ self.W_V

        # Split heads: (seq_len, d_model) → (h, seq_len, dh)
        Q = Q.reshape(seq_len, h, dh).transpose(1, 0, 2)
        K = K.reshape(seq_len, h, dh).transpose(1, 0, 2)
        V = V.reshape(seq_len, h, dh).transpose(1, 0, 2)

        # Scaled dot-product attention
        scale = dh ** -0.5
        attn = (Q @ K.transpose(0, 2, 1)) * scale   # (h, seq_len, seq_len)

        if mask is not None:
            attn = np.where(mask, attn, -1e9)

        attn = np.exp(attn - attn.max(axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-9)

        out = attn @ V   # (h, seq_len, dh)
        out = out.transpose(1, 0, 2).reshape(seq_len, d)
        return out @ self.W_O


# ── Feed-Forward Network ──────────────────────────────────────────
class FFN:
    """
    Two-layer MLP with GELU activation.
    ViT uses d_ff = 4 * d_model (same as original Transformer).
    """

    def __init__(self, d_model: int, d_ff: int | None = None):
        self.d_ff = d_ff or 4 * d_model
        scale = d_model ** -0.5
        self.W1 = rng.standard_normal((d_model, self.d_ff)) * scale
        self.b1 = np.zeros(self.d_ff)
        self.W2 = rng.standard_normal((self.d_ff, d_model)) * scale
        self.b2 = np.zeros(d_model)

    def gelu(self, x: np.ndarray) -> np.ndarray:
        """Approximate GELU: x · Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2


# ── Layer Norm ────────────────────────────────────────────────────
def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


# ── ViT Encoder Block ─────────────────────────────────────────────
class ViTBlock:
    """
    Pre-norm ViT encoder block:
    x = x + MHA(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
    """

    def __init__(self, d_model: int, n_heads: int):
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ffn  = FFN(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attn.forward(layer_norm(x))
        x = x + self.ffn.forward(layer_norm(x))
        return x


# ── Full ViT Forward Pass ─────────────────────────────────────────
class ViT:
    """
    Vision Transformer (encoder-only, as used in CLIP image encoder).

    Architecture:
      1. Patch + Position Embed → (N+1, d_model)  ([CLS] prepended)
      2. L × ViTBlock
      3. LayerNorm → take [CLS] output → Linear head (optional)
    """

    def __init__(self, image_size: int = 32, patch_size: int = 8,
                 in_channels: int = 3, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 4):
        assert image_size % patch_size == 0
        self.patch_size  = patch_size
        self.n_h = self.n_w = image_size // patch_size
        self.N   = self.n_h * self.n_w
        self.d_model = d_model

        patch_dim = patch_size * patch_size * in_channels
        self.W_patch = rng.standard_normal((patch_dim, d_model)) * (patch_dim ** -0.5)
        self.cls_token = rng.standard_normal((1, d_model)) * 0.02
        self.pos_emb   = learned_pe(self.N + 1, d_model)

        self.blocks = [ViTBlock(d_model, n_heads) for _ in range(n_layers)]

        # Linear head for classification
        self.W_head = rng.standard_normal((d_model, d_model)) * (d_model ** -0.5)

    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        image: (H, W, C)
        Returns: (d_model,) — [CLS] embedding after all blocks
        """
        # 1. Patch extraction + projection
        patches = extract_patches(image, self.patch_size)   # (N, P²C)
        tokens  = patch_embed(patches, self.d_model, self.W_patch)  # (N, d_model)

        # 2. Prepend [CLS], add positional embeddings
        tokens = prepend_cls(tokens, self.cls_token)        # (N+1, d_model)
        tokens = tokens + self.pos_emb                      # (N+1, d_model)

        # 3. Transformer blocks
        for block in self.blocks:
            tokens = block.forward(tokens)

        # 4. Layer norm → [CLS] output
        tokens = layer_norm(tokens)
        cls_out = tokens[0]   # (d_model,) — aggregate representation

        return cls_out


def main():
    section("1. PATCH EXTRACTION")
    print("""
  ViT splits image into non-overlapping P×P patches:
    N = (H/P) × (W/P)   patches total
    Each patch → flattened vector of length P²·C

  Image (H, W, C) → (N, P²C) via reshape + transpose
  Then: linear projection (P²C, d_model) → (N, d_model) patch tokens

  ViT-B/32 on 224×224×3:  N=49,   P²C=3072,  d_model=768
  ViT-B/16 on 224×224×3:  N=196,  P²C=768,   d_model=768
  ViT-L/14 on 224×224×3:  N=256,  P²C=588,   d_model=1024
  ViT-L/14 on 336×336×3:  N=576,  P²C=588,   d_model=1024
""")

    # Demonstrate
    H, W, C, P = 32, 32, 3, 8
    image = rng.standard_normal((H, W, C))
    patches = extract_patches(image, P)
    n_h = n_w = H // P
    print(f"  Image:   ({H}, {W}, {C})")
    print(f"  P={P}: N={n_h}×{n_w}={patches.shape[0]} patches, each dim={patches.shape[1]}")
    print(f"  Patches: {patches.shape}")

    d_model = 64
    tokens = patch_embed(patches, d_model)
    print(f"  After linear proj: {tokens.shape}   → (N, d_model)")

    section("2. [CLS] TOKEN + POSITIONAL ENCODING")
    print("""
  [CLS] is a learnable token prepended to patch sequence.
  After L transformer blocks, [CLS]'s output summarizes the image.
  Used as the image embedding for CLIP / classification head.

  Positional encoding (ViT uses learned PEs):
  - Sinusoidal: deterministic, generalizes to unseen resolutions
  - Learned:    initialized randomly, optimized jointly with model
  - 2D-aware:   encodes (row, col) independently → better for images
""")

    cls_tok = rng.standard_normal((1, d_model)) * 0.02
    seq_with_cls = prepend_cls(tokens, cls_tok)
    print(f"  tokens (N, d):      {tokens.shape}")
    print(f"  after [CLS] prepend:{seq_with_cls.shape}   (N+1, d_model)")

    pe_sin = sinusoidal_pe_2d(n_h, n_w, d_model)
    pe_ler = learned_pe(patches.shape[0] + 1, d_model)
    print(f"  2D sinusoidal PE:   {pe_sin.shape}")
    print(f"  Learned PE (N+1):   {pe_ler.shape}")

    section("3. MULTI-HEAD SELF-ATTENTION")
    mhsa = MultiHeadSelfAttention(d_model=64, n_heads=4)
    seq = seq_with_cls + pe_ler                    # (N+1, 64)
    out = mhsa.forward(seq)
    print(f"  Input:  {seq.shape}  (N+1, d_model)")
    print(f"  Output: {out.shape}  unchanged shape (residual connection adds back)")
    print(f"\n  Each of 4 heads has d_head = {64//4} = 16 dims")
    print(f"  Attention pattern: every patch attends to every other patch")
    print(f"  [CLS] (position 0) accumulates global context from all patches")

    section("4. FULL VIT FORWARD PASS")
    vit = ViT(image_size=32, patch_size=8, in_channels=3,
              d_model=64, n_heads=4, n_layers=3)
    image = rng.standard_normal((32, 32, 3))
    cls_emb = vit.forward(image)
    print(f"  ViT(32px, P=8, d=64, heads=4, layers=3)")
    print(f"  Image:     (32, 32, 3)")
    print(f"  [CLS] emb: {cls_emb.shape}   ← used as image representation")
    print(f"  Norm:      {np.linalg.norm(cls_emb):.4f}  (not unit-normalized yet)")

    # Batch of images
    batch = [rng.standard_normal((32, 32, 3)) for _ in range(5)]
    embs  = np.stack([vit.forward(img) for img in batch])
    print(f"\n  Batch of 5 images → {embs.shape}  (batch, d_model)")

    section("5. VIT SIZE TABLE")
    print("""
  ┌────────────────┬───────┬───────┬───────┬─────────┬──────────┐
  │ Model          │ P     │ d     │ Heads │ Layers  │ Params   │
  ├────────────────┼───────┼───────┼───────┼─────────┼──────────┤
  │ ViT-Ti/16      │  16   │  192  │   3   │   12    │  5.7M    │
  │ ViT-S/16       │  16   │  384  │   6   │   12    │  22M     │
  │ ViT-B/32       │  32   │  768  │  12   │   12    │  86M     │
  │ ViT-B/16       │  16   │  768  │  12   │   12    │  86M     │
  │ ViT-L/14       │  14   │ 1024  │  16   │   24    │  307M    │
  │ ViT-H/14       │  14   │ 1280  │  16   │   32    │  632M    │
  └────────────────┴───────┴───────┴───────┴─────────┴──────────┘

  ViT-B = Base, ViT-L = Large, ViT-H = Huge
  /P = patch size (32, 16, or 14 pixels)
  CLIP uses ViT-B/32 (fast) → ViT-L/14@336 (best quality)
""")

    section("6. ATTENTION PATTERN ANALYSIS")
    # Compute raw attention weights for visualization
    seq_len = seq_with_cls.shape[0]
    Q = seq_with_cls @ mhsa.W_Q
    K = seq_with_cls @ mhsa.W_K
    scale = (d_model // 4) ** -0.5
    attn_raw = Q @ K.T * scale           # (seq_len, seq_len)
    attn_raw = np.exp(attn_raw - attn_raw.max(axis=-1, keepdims=True))
    attn_map = attn_raw / (attn_raw.sum(axis=-1, keepdims=True) + 1e-9)

    print(f"  Attention map shape: {attn_map.shape}  (seq_len × seq_len)")
    print(f"  Row 0 = [CLS] attention to all patches:")
    cls_attn = attn_map[0]
    print(f"    mean={cls_attn.mean():.4f}  max={cls_attn.max():.4f}  "
          f"argmax={cls_attn.argmax()} (patch {cls_attn.argmax()-1})")
    print(f"\n  ViT attention heads learn to focus on:")
    print(f"    - Object contours and textures (low-level heads)")
    print(f"    - Semantic regions (high-level heads in later layers)")
    print(f"    - [CLS] attends globally; patch tokens attend locally")


if __name__ == "__main__":
    main()
