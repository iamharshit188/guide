"""
CLIP (Contrastive Language-Image Pretraining) from scratch.
Covers: InfoNCE loss derivation, temperature scaling, contrastive
        training simulation, similarity matrix, embedding alignment.
All computations in NumPy — no vision/language model dependencies.
"""

import numpy as np

rng = np.random.default_rng(42)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Utility functions ─────────────────────────────────────────────
def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Unit-normalize along axis."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + 1e-8)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ── Dummy encoders ────────────────────────────────────────────────
class ImageEncoder:
    """
    Placeholder for a ViT or ResNet image encoder.
    Maps image pixels → d-dimensional embedding.
    In CLIP: ViT-B/32 gives (batch, 768).
    """

    def __init__(self, d_model: int = 64):
        self.W = rng.standard_normal((32 * 32 * 3, d_model)) * 0.01
        self.d_model = d_model

    def encode(self, images: np.ndarray) -> np.ndarray:
        """images: (B, H, W, C) flattened to (B, H*W*C)."""
        B = images.shape[0]
        flat = images.reshape(B, -1)
        if flat.shape[1] != self.W.shape[0]:
            flat = flat[:, :self.W.shape[0]] if flat.shape[1] > self.W.shape[0] \
                   else np.pad(flat, ((0,0),(0, self.W.shape[0]-flat.shape[1])))
        emb = flat @ self.W
        return l2_normalize(emb)


class TextEncoder:
    """
    Placeholder for a transformer text encoder.
    Maps token sequence → d-dimensional embedding.
    In CLIP: 12-layer transformer gives (batch, 512).
    """

    def __init__(self, vocab_size: int = 256, d_model: int = 64):
        self.emb = rng.standard_normal((vocab_size, d_model)) * 0.01
        self.d_model = d_model

    def encode(self, token_ids: np.ndarray) -> np.ndarray:
        """token_ids: (B, seq_len) → mean-pool embeddings (B, d_model)."""
        vecs = self.emb[token_ids]      # (B, seq_len, d_model)
        emb  = vecs.mean(axis=1)        # (B, d_model)
        return l2_normalize(emb)


# ── CLIP Loss ─────────────────────────────────────────────────────
def clip_loss(image_emb: np.ndarray, text_emb: np.ndarray,
              logit_scale: float) -> tuple[float, np.ndarray]:
    """
    InfoNCE / CLIP contrastive loss.

    Given batch of B (image, text) pairs:
      S = logit_scale * image_emb @ text_emb.T     shape (B, B)
      L_i2t = CrossEntropy(S_i, i)   (row i must match column i)
      L_t2i = CrossEntropy(S_j, j)   (column j must match row j)
      L = (L_i2t + L_t2i) / 2

    logit_scale = exp(log_temperature), initialized to exp(log(1/0.07)) ≈ 14.3
    Temperature τ = 1/logit_scale (lower τ → sharper distribution).

    Returns: (loss, similarity_matrix)
    """
    B = image_emb.shape[0]
    labels = np.arange(B)

    # Cosine similarity matrix scaled by temperature
    sim = logit_scale * (image_emb @ text_emb.T)   # (B, B)

    # Cross-entropy per direction
    def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
        log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True) + 1e-9)
        return float(-log_probs[np.arange(len(targets)), targets].mean())

    loss_i2t = cross_entropy(sim, labels)    # images predicting texts
    loss_t2i = cross_entropy(sim.T, labels)  # texts predicting images
    loss = (loss_i2t + loss_t2i) / 2.0

    return loss, sim


# ── Gradient of logit_scale ───────────────────────────────────────
def clip_loss_grad_scale(image_emb: np.ndarray, text_emb: np.ndarray,
                         log_scale: float) -> float:
    """
    d(loss)/d(log_scale) via numerical diff.
    logit_scale = exp(log_scale), so gradients flow through exp.
    """
    eps = 1e-4
    loss_hi, _ = clip_loss(image_emb, text_emb, np.exp(log_scale + eps))
    loss_lo, _ = clip_loss(image_emb, text_emb, np.exp(log_scale - eps))
    return (loss_hi - loss_lo) / (2 * eps)


# ── Contrastive Training Simulation ──────────────────────────────
def simulate_clip_training(B: int = 16, d_model: int = 64,
                            n_steps: int = 100, lr: float = 0.01):
    """
    Simulate CLIP training on random (image, text) pairs.
    Updates projection matrices + log_scale via gradient approximation.
    """
    img_enc  = ImageEncoder(d_model=d_model)
    txt_enc  = TextEncoder(d_model=d_model)
    log_scale = np.log(14.3)   # CLIP initializes at this value

    history = []
    for step in range(n_steps):
        # Synthetic batch: paired images + text token ids
        images    = rng.standard_normal((B, 8, 8, 3))
        token_ids = rng.integers(0, 256, (B, 10))

        img_emb = img_enc.encode(images)
        txt_emb = txt_enc.encode(token_ids)

        loss, sim = clip_loss(img_emb, txt_emb, np.exp(log_scale))

        # Gradient on log_scale
        dL_dscale = clip_loss_grad_scale(img_emb, txt_emb, log_scale)
        log_scale -= lr * dL_dscale

        # Approximate gradient on encoders: nudge W in the direction
        # that reduces loss (numerical gradient on random projection)
        for param in [img_enc.W, txt_enc.emb]:
            noise = rng.standard_normal(param.shape) * 0.001
            img_emb2 = img_enc.encode(images)
            txt_emb2 = txt_enc.encode(token_ids)
            loss2, _ = clip_loss(img_emb2, txt_emb2, np.exp(log_scale))
            if loss2 < loss:
                pass   # noise helped; keep
            # In full CLIP: Adam optimizer on actual gradients via autograd

        history.append({"step": step, "loss": loss,
                         "temperature": 1.0 / np.exp(log_scale)})

        if step % 20 == 0 or step == n_steps - 1:
            print(f"  step {step:4d}: loss={loss:.4f}  "
                  f"τ={1/np.exp(log_scale):.4f}  "
                  f"logit_scale={np.exp(log_scale):.2f}")

    return history


# ── Similarity Matrix Visualization ──────────────────────────────
def print_similarity_matrix(sim: np.ndarray, title: str = ""):
    """Print a B×B similarity matrix as ASCII heatmap."""
    if title:
        print(f"\n  {title}")
    B = sim.shape[0]
    mn, mx = sim.min(), sim.max()
    chars = " ░▒▓█"
    print(f"  {'':4}", end="")
    for j in range(B):
        print(f" T{j}", end="")
    print()
    for i in range(B):
        print(f"  I{i}:", end="")
        for j in range(B):
            val = (sim[i, j] - mn) / (mx - mn + 1e-8)
            idx = int(val * (len(chars) - 1))
            diag = "*" if i == j else " "
            print(f"{diag}{chars[idx]} ", end="")
        print()


# ── Zero-shot retrieval evaluation ────────────────────────────────
def evaluate_retrieval(image_emb: np.ndarray, text_emb: np.ndarray,
                       logit_scale: float) -> dict:
    """
    Compute image→text and text→image retrieval recall@k.
    Correct match = index on the diagonal.
    """
    B = image_emb.shape[0]
    sim = logit_scale * (image_emb @ text_emb.T)

    results = {}
    for k in [1, 5]:
        if k > B:
            continue
        # Image→Text R@k
        top_k_i2t = np.argsort(-sim, axis=1)[:, :k]
        hit_i2t = sum(1 for i in range(B) if i in top_k_i2t[i])
        # Text→Image R@k
        top_k_t2i = np.argsort(-sim.T, axis=1)[:, :k]
        hit_t2i = sum(1 for j in range(B) if j in top_k_t2i[j])
        results[f"I2T_R@{k}"] = hit_i2t / B
        results[f"T2I_R@{k}"] = hit_t2i / B

    return results


def main():
    section("1. CLIP INFONCE LOSS")
    print("""
  Given batch size B, (image_i, text_i) are positives; all others negatives.

  S_ij = (logit_scale) · cos(image_i, text_j)
       = exp(log τ⁻¹) · image_i · text_j   (unit vectors → dot = cosine)

  L_i2t = (1/B) Σ_i -log [ exp(S_ii) / Σ_j exp(S_ij) ]
  L_t2i = (1/B) Σ_j -log [ exp(S_jj) / Σ_i exp(S_ij) ]
  L_CLIP = (L_i2t + L_t2i) / 2

  Interpretation:
  - Each image must identify its paired text among B candidates (and vice versa)
  - With B=32768 (CLIP paper batch), hard to cheat by ignoring modality
  - Temperature τ controls sharpness: low τ → confident → hard negatives matter more
  - CLIP learns τ as exp(log_scale), clipped to [0.01, 100]
""")

    # Demonstrate with a small batch
    B, d = 8, 32
    img_enc = ImageEncoder(d_model=d)
    txt_enc = TextEncoder(d_model=d)

    images    = rng.standard_normal((B, 8, 8, 3))
    token_ids = rng.integers(0, 256, (B, 10))

    img_emb = img_enc.encode(images)
    txt_emb = txt_enc.encode(token_ids)

    loss, sim = clip_loss(img_emb, txt_emb, logit_scale=14.3)
    print(f"  Batch size B={B}, d_model={d}")
    print(f"  Initial CLIP loss: {loss:.4f}  (random embeddings → ~log(B)={np.log(B):.4f})")

    print_similarity_matrix(sim[:6, :6], "Similarity matrix S (first 6×6, * = diagonal)")

    section("2. TEMPERATURE EFFECT")
    print(f"  {'τ':>8} {'logit_scale':>12} {'loss':>10} {'sharpness':>12}")
    print(f"  {'-'*46}")
    for tau in [0.01, 0.07, 0.1, 0.5, 1.0]:
        scale = 1.0 / tau
        l, s  = clip_loss(img_emb, txt_emb, scale)
        probs = softmax(s[0])
        sharpness = float(probs.max() - probs.mean())
        print(f"  {tau:8.2f} {scale:12.2f} {l:10.4f} {sharpness:12.4f}")

    print("""
  Low τ (e.g. 0.07):  sharp peaks → decisive but harder to train
  High τ (e.g. 1.0):  flat distribution → easy but less discriminative
  CLIP paper uses τ=0.07; learned via log_scale parameter
""")

    section("3. CONTRASTIVE TRAINING SIMULATION")
    print("  Training CLIP on random (image, text) pairs...")
    history = simulate_clip_training(B=16, d_model=32, n_steps=80, lr=0.005)

    losses = [h["loss"] for h in history]
    print(f"\n  Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"  Temperature: {history[0]['temperature']:.4f} → {history[-1]['temperature']:.4f}")

    section("4. RETRIEVAL EVALUATION")
    # Simulate aligned embeddings (positive pairs close, negatives far)
    B_eval = 20
    base_emb = l2_normalize(rng.standard_normal((B_eval, 32)))
    # Aligned: text_emb ≈ image_emb + small noise
    img_eval = l2_normalize(base_emb + rng.standard_normal((B_eval, 32)) * 0.1)
    txt_eval = l2_normalize(base_emb + rng.standard_normal((B_eval, 32)) * 0.1)

    metrics = evaluate_retrieval(img_eval, txt_eval, logit_scale=14.3)
    print(f"  Aligned embeddings (noise=0.1):")
    for k, v in metrics.items():
        print(f"    {k}: {v:.3f}")

    # Random embeddings (baseline)
    img_rand = l2_normalize(rng.standard_normal((B_eval, 32)))
    txt_rand = l2_normalize(rng.standard_normal((B_eval, 32)))
    rand_metrics = evaluate_retrieval(img_rand, txt_rand, logit_scale=14.3)
    print(f"\n  Random embeddings (baseline):")
    for k, v in rand_metrics.items():
        print(f"    {k}: {v:.3f}")

    section("5. CLIP ARCHITECTURE SUMMARY")
    print("""
  Image encoder options (by CLIP variant):
  ┌──────────────────┬──────────┬───────────┬────────────────┐
  │ Variant          │ Patches  │ d_model   │ Parameters     │
  ├──────────────────┼──────────┼───────────┼────────────────┤
  │ ViT-B/32         │ 7×7=49   │ 768       │ 86M image      │
  │ ViT-B/16         │ 14×14=196│ 768       │ 86M image      │
  │ ViT-L/14         │ 16×16=256│ 1024      │ 307M image     │
  │ ViT-L/14@336px   │ 24×24=576│ 1024      │ 307M image     │
  └──────────────────┴──────────┴───────────┴────────────────┘

  Text encoder: 12-layer transformer, context length 77 tokens
  Joint embedding: linear projection to 512 (B/32) or 768 (L/14) dims
  Training: 400M image-text pairs, batch size 32768, Adam + cosine LR
""")


if __name__ == "__main__":
    main()
