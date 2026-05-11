"""
Zero-shot image classification with CLIP from scratch.
Covers: prompt engineering, text template ensemble, cosine similarity
        scoring, top-k classification, prompt sensitivity analysis.
All computations in NumPy — no model downloads required.
"""

import numpy as np

rng = np.random.default_rng(42)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    x = x / temperature
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-9)


# ── Dummy CLIP encoders ───────────────────────────────────────────
# In real CLIP: image_encoder = ViT, text_encoder = Transformer
# Here we simulate aligned embeddings to demonstrate the zero-shot pipeline.

class SimulatedCLIP:
    """
    Simulated CLIP model with aligned image + text embedding spaces.
    Class c has a 'true' embedding; images are noisy versions of class embeddings.
    """

    def __init__(self, d_model: int = 64, n_classes: int = 10):
        self.d_model   = d_model
        self.n_classes = n_classes
        # Each class has a canonical direction in embedding space
        self.class_dirs = l2_normalize(
            rng.standard_normal((n_classes, d_model))
        )
        self.logit_scale = 14.3   # CLIP default ≈ exp(log(1/0.07))

    def encode_image(self, class_idx: int, noise: float = 0.3) -> np.ndarray:
        """Simulate image embedding: true class direction + Gaussian noise."""
        emb = self.class_dirs[class_idx] + rng.standard_normal(self.d_model) * noise
        return l2_normalize(emb)

    def encode_text(self, class_idx: int, prompt_noise: float = 0.1) -> np.ndarray:
        """Simulate text embedding: true class direction + small noise."""
        emb = self.class_dirs[class_idx] + rng.standard_normal(self.d_model) * prompt_noise
        return l2_normalize(emb)

    def encode_text_batch(self, class_indices: list[int],
                          noise: float = 0.1) -> np.ndarray:
        """Encode multiple text prompts → (n_prompts, d_model)."""
        return l2_normalize(
            np.stack([self.encode_text(c, noise) for c in class_indices])
        )


# ── Zero-Shot Classification Pipeline ────────────────────────────
CLASSES = [
    "cat", "dog", "bird", "car", "airplane",
    "ship", "horse", "deer", "frog", "truck"
]

PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a picture of a {}.",
    "an image of a {}.",
    "a photo of the {}.",
    "a {} in the image.",
    "this is a {}.",
    "a real photo of a {}.",
    "a clear photo of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
]


def zero_shot_predict(image_emb: np.ndarray, text_embs: np.ndarray,
                      logit_scale: float = 14.3,
                      temperature: float = 1.0) -> tuple[int, np.ndarray]:
    """
    CLIP zero-shot classification:

    ŷ = argmax_c cos(v, t_c)
      = argmax_c  (v · t_c)   (unit vectors)

    Equivalently: softmax of scaled cosine similarities gives
    probability distribution over classes.

    image_emb:  (d_model,)
    text_embs:  (n_classes, d_model)  — one embedding per class
    Returns: predicted class index + probability vector
    """
    # Cosine similarities (dot product of unit vectors)
    sims  = logit_scale * (text_embs @ image_emb)  # (n_classes,)
    probs = softmax(sims, temperature=temperature)
    return int(np.argmax(probs)), probs


def ensemble_text_embedding(clip: SimulatedCLIP, class_idx: int,
                             n_templates: int = 10) -> np.ndarray:
    """
    Prompt ensemble: encode class with multiple templates, then average.
    Reduces sensitivity to specific wording.

    ensemble_emb = L2_normalize( mean_k( encode_text(template_k.format(class)) ) )
    """
    # Each template gets slightly different noise → simulates different prompts
    embs = np.stack([
        clip.encode_text(class_idx, prompt_noise=0.05)
        for _ in range(n_templates)
    ])
    mean_emb = embs.mean(axis=0)
    return l2_normalize(mean_emb)


def build_classifier(clip: SimulatedCLIP, classes: list[str],
                     use_ensemble: bool = True,
                     n_templates: int = 10) -> np.ndarray:
    """
    Build the zero-shot classifier matrix: (n_classes, d_model).
    This is the offline step: encode all class prompts once.
    """
    embs = []
    for i in range(len(classes)):
        if use_ensemble:
            embs.append(ensemble_text_embedding(clip, i, n_templates))
        else:
            embs.append(clip.encode_text(i, prompt_noise=0.0))
    return np.stack(embs)   # (n_classes, d_model)


# ── Accuracy Evaluation ───────────────────────────────────────────
def evaluate_zero_shot(clip: SimulatedCLIP, text_embs: np.ndarray,
                       n_images: int = 200, noise: float = 0.3) -> dict:
    """
    Sample images from each class, classify, compute top-1 and top-5 accuracy.
    """
    n_classes = len(CLASSES)
    correct_top1 = 0
    correct_top5 = 0

    for true_cls in range(n_classes):
        for _ in range(n_images // n_classes):
            img_emb = clip.encode_image(true_cls, noise=noise)
            pred, probs = zero_shot_predict(img_emb, text_embs)
            top5 = np.argsort(-probs)[:5]
            if pred == true_cls:
                correct_top1 += 1
            if true_cls in top5:
                correct_top5 += 1

    return {
        "top1": correct_top1 / n_images,
        "top5": correct_top5 / n_images,
        "n_images": n_images,
    }


# ── Confusion Matrix ──────────────────────────────────────────────
def confusion_matrix(clip: SimulatedCLIP, text_embs: np.ndarray,
                     n_per_class: int = 20) -> np.ndarray:
    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for true_cls in range(n):
        for _ in range(n_per_class):
            img_emb = clip.encode_image(true_cls, noise=0.3)
            pred, _ = zero_shot_predict(img_emb, text_embs)
            cm[true_cls, pred] += 1
    return cm


def print_confusion(cm: np.ndarray, classes: list[str]):
    n = len(classes)
    col_width = 8
    header = f"{'':10}" + "".join(f"{c[:6]:>{col_width}}" for c in classes)
    print(f"  {header}")
    for i, row_name in enumerate(classes):
        row = f"  {row_name:10}" + "".join(
            f"{'*' + str(cm[i,j]) if i == j else str(cm[i,j]):>{col_width}}"
            for j in range(n)
        )
        print(row)


def main():
    section("1. CLIP ZERO-SHOT CLASSIFICATION")
    print("""
  Zero-shot classification without fine-tuning:

  Offline (once):
    For each class c, encode: "a photo of a {c}." → t_c  ∈ ℝ^d

  Online (per image):
    Encode image → v ∈ ℝ^d
    ŷ = argmax_c cos(v, t_c) = argmax_c (v · t_c)   (both unit vectors)

  Probability:
    p(y=c | x) = exp(τ⁻¹ · cos(v, t_c)) / Σ_c' exp(τ⁻¹ · cos(v, t_c'))

  This works because CLIP's contrastive training aligns image and text
  embeddings so that matching pairs have high cosine similarity.
""")

    clip = SimulatedCLIP(d_model=64, n_classes=len(CLASSES))

    # Single image classification example
    true_class = 0   # "cat"
    img_emb    = clip.encode_image(true_class, noise=0.2)
    text_embs  = build_classifier(clip, CLASSES, use_ensemble=False)

    pred, probs = zero_shot_predict(img_emb, text_embs)
    print(f"  True class: {CLASSES[true_class]}")
    print(f"  Predicted:  {CLASSES[pred]}  (correct={pred == true_class})")
    print(f"\n  Top-5 predictions:")
    top5 = np.argsort(-probs)[:5]
    for rank, idx in enumerate(top5, 1):
        marker = " ← TRUE" if idx == true_class else ""
        print(f"    {rank}. {CLASSES[idx]:10s}  p={probs[idx]:.4f}{marker}")

    section("2. PROMPT ENGINEERING")
    print("""
  The choice of text template significantly affects accuracy.

  Templates for ImageNet zero-shot (OpenAI CLIP paper):
    - "a photo of a {label}."           (most common)
    - "a photo of many {label}."        (for countable objects)
    - "a bad photo of a {label}."       (helps with degraded images)
    - "art of a {label}."               (for artwork)
    - "a drawing of a {label}."
    - "a photo of the large {label}."
    - "a photo of the small {label}."

  Key insight: 80 templates + ensemble → +3.5% top-1 accuracy on ImageNet
  compared to a single prompt.
""")

    # Compare single vs. multiple prompt templates
    print("  Prompt sensitivity analysis (noise=0.3, 100 images per class):")
    print(f"  {'Prompt Strategy':30} {'Top-1':>8} {'Top-5':>8}")
    print(f"  {'-'*50}")

    for label, use_ens, n_templates in [
        ("Single prompt (no noise)", False, 1),
        ("Ensemble (5 templates)", True, 5),
        ("Ensemble (10 templates)", True, 10),
    ]:
        embs = build_classifier(clip, CLASSES,
                                use_ensemble=use_ens, n_templates=n_templates)
        metrics = evaluate_zero_shot(clip, embs, n_images=100, noise=0.3)
        print(f"  {label:30} {metrics['top1']:8.3f} {metrics['top5']:8.3f}")

    section("3. ENSEMBLE TEXT EMBEDDINGS")
    print("""
  Prompt ensemble algorithm:
    1. For each class c, apply k prompt templates: t^(1)_c, ..., t^(k)_c
    2. Encode each: e^(i)_c = TextEncoder(t^(i)_c)
    3. Average: ē_c = (1/k) Σ_i e^(i)_c
    4. Normalize: t_c = ē_c / ||ē_c||

  Why averaging works:
    - Different templates capture different aspects of the concept
    - Averaging reduces template-specific noise
    - Normalized mean ≈ most central direction in embedding space
""")

    class_idx = 0   # "cat"
    single_emb   = clip.encode_text(class_idx, prompt_noise=0.0)
    ensemble_emb = ensemble_text_embedding(clip, class_idx, n_templates=10)

    print(f"  Class: {CLASSES[class_idx]}")
    print(f"  Single embedding norm: {np.linalg.norm(single_emb):.4f}")
    print(f"  Ensemble emb norm:     {np.linalg.norm(ensemble_emb):.4f}")
    print(f"  Cosine(single, ensemble): "
          f"{float(single_emb @ ensemble_emb):.4f}")

    # Show how ensemble reduces per-template variance
    single_embs = np.stack([clip.encode_text(class_idx, 0.1) for _ in range(10)])
    variance_single  = float(np.var(single_embs, axis=0).mean())
    variance_ensemble = float(np.var(ensemble_emb))
    print(f"\n  Variance across 10 single embeddings: {variance_single:.6f}")
    print(f"  Variance of ensemble embedding:       {variance_ensemble:.6f}")
    print(f"  Variance reduction: {variance_single / (variance_ensemble + 1e-9):.1f}×")

    section("4. TEMPERATURE SENSITIVITY")
    print("  Effect of temperature on classification confidence:")
    print(f"  {'τ':>6} {'Max prob':>10} {'Entropy':>10} {'Top-1 acc':>12}")
    print(f"  {'-'*42}")

    text_embs_ens = build_classifier(clip, CLASSES, use_ensemble=True)
    img_emb = clip.encode_image(0, noise=0.3)
    for tau in [0.1, 0.5, 1.0, 2.0, 5.0]:
        _, probs = zero_shot_predict(img_emb, text_embs_ens,
                                     logit_scale=14.3, temperature=tau)
        ent = float(-np.sum(probs * np.log(probs + 1e-9)))
        print(f"  {tau:6.1f} {probs.max():10.4f} {ent:10.4f}", end="")

        # Quick accuracy check
        correct = sum(
            1 for c in range(len(CLASSES))
            for _ in range(10)
            if zero_shot_predict(clip.encode_image(c, 0.3), text_embs_ens,
                                 temperature=tau)[0] == c
        ) / (len(CLASSES) * 10)
        print(f"  {correct:12.3f}")

    section("5. CONFUSION MATRIX")
    text_embs_ens = build_classifier(clip, CLASSES, use_ensemble=True)
    cm = confusion_matrix(clip, text_embs_ens, n_per_class=20)
    diag_acc = cm.diagonal().sum() / cm.sum()
    print(f"  Overall accuracy: {diag_acc:.3f}\n")
    print_confusion(cm, [c[:6] for c in CLASSES])
    print("\n  * = correct prediction (diagonal)")

    section("6. ZERO-SHOT vs. LINEAR PROBE vs. FEW-SHOT")
    print("""
  ┌──────────────────┬────────────────────────────────────────────────┐
  │ Method           │ Description                                    │
  ├──────────────────┼────────────────────────────────────────────────┤
  │ Zero-shot        │ No labeled data; prompt engineering only       │
  │                  │ ImageNet: ~76% top-1 (ViT-L/14)               │
  ├──────────────────┼────────────────────────────────────────────────┤
  │ Linear probe     │ Freeze CLIP features; train linear head        │
  │                  │ 1 epoch on ImageNet: ~85% top-1               │
  ├──────────────────┼────────────────────────────────────────────────┤
  │ Few-shot (k=1)   │ 1 image per class → kNN or prototype          │
  │                  │ Between zero-shot and linear probe             │
  ├──────────────────┼────────────────────────────────────────────────┤
  │ Full fine-tune   │ Update all weights on target dataset           │
  │                  │ Best accuracy but loses zero-shot capability   │
  └──────────────────┴────────────────────────────────────────────────┘

  CLIP zero-shot matches supervised ResNet-50 on ImageNet without
  any ImageNet labels — demonstrating emergent generalization.

  Zero-shot generalizes better OOD (out-of-distribution);
  fine-tuned models overfit the training distribution.
""")

    section("7. CLIP APPLICATIONS BEYOND CLASSIFICATION")
    print("""
  ┌────────────────────────┬────────────────────────────────────────────┐
  │ Application            │ How CLIP is used                           │
  ├────────────────────────┼────────────────────────────────────────────┤
  │ Image search           │ Encode query text → retrieve nearest imgs  │
  │ Text-to-image ranking  │ Score candidate images by text similarity  │
  │ Image captioning       │ CLIP score as reward for caption quality   │
  │ Stable Diffusion       │ CLIP text encoder guides denoising process │
  │ DALL-E 2               │ CLIP image embedding → prior → decoder     │
  │ Video retrieval        │ Mean-pool frame embeddings, query with text│
  │ Open-vocab detection   │ CLIP per-region → classify arbitrary class │
  │ Medical imaging        │ BioMedCLIP fine-tuned on radiology reports │
  └────────────────────────┴────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
