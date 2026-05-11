# Module 13 — Multimodal Models

> **Run the code:**
> ```bash
> cd src/13-multimodal
> python3.14 clip_scratch.py
> python3.14 vit_patch.py
> python3.14 captioning.py
> python3.14 zero_shot.py
> ```

---

## Prerequisites & Overview

**Time estimate:** 7–9 hours

| Prerequisite | From |
|-------------|------|
| Transformer attention + MHA | Modules 06–07 |
| Contrastive learning intuition | Module 02 (metric learning) |
| Patch-based image processing | Module 01 (matrix ops) |

**Before you start:**
- [ ] Know what a dot-product similarity is and how it relates to cosine similarity
- [ ] Understand softmax and how temperature controls distribution sharpness
- [ ] Know what a linear projection does (matrix multiply)

**Module map:**

| Section | Core formula |
|---------|-------------|
| CLIP | InfoNCE: $\mathcal{L} = -\frac{1}{N}\sum_i \log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}$ |
| ViT patches | Patch flatten → Linear → positional encoding |
| Cross-modal attention | $\text{Attn}(Q_{\text{text}}, K_{\text{image}}, V_{\text{image}})$ |
| Image captioning | Vision encoder → Transformer decoder (cross-attention) |
| Zero-shot classification | $\hat{y} = \arg\max_c \cos(\mathbf{v}, \mathbf{t}_c)$ |

---

## CLIP — Contrastive Language-Image Pre-training

### Architecture

CLIP (Radford et al., 2021) jointly trains an image encoder and a text encoder to align their embedding spaces. The only training signal is: images and their paired captions should be close; mismatched pairs should be far.

```
Image → Image Encoder (ViT or ResNet) → image_embedding ∈ ℝ^d
Text  → Text Encoder  (Transformer)   → text_embedding  ∈ ℝ^d
Both embeddings L2-normalized to unit sphere.
```

Similarity matrix for a batch of $N$ (image, text) pairs:

$$S_{ij} = \frac{\mathbf{v}_i \cdot \mathbf{t}_j}{\tau} \qquad \mathbf{v}_i, \mathbf{t}_j \in \mathbb{R}^d,\; \|\mathbf{v}_i\|=\|\mathbf{t}_j\|=1$$

where $\tau > 0$ is a learnable temperature parameter (initialized to $\tau = 0.07$).

### InfoNCE Loss

CLIP optimizes a symmetric InfoNCE (Noise-Contrastive Estimation) loss:

$$\mathcal{L}_{\text{image}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{S_{ii}}}{\sum_{j=1}^{N}e^{S_{ij}}}$$

$$\mathcal{L}_{\text{text}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{S_{ii}}}{\sum_{j=1}^{N}e^{S_{ji}}}$$

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}(\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}})$$

Each is a cross-entropy loss where the target is the diagonal of $S$ — matching pairs should dominate each row/column.

**Intuition:** For each image, the text of its own caption should be most similar (row-wise softmax). For each text, its image should be most similar (column-wise softmax). The diagonal of $S$ contains all the "correct" pair scores.

### Why InfoNCE Works

InfoNCE is a lower bound on the mutual information $I(\mathbf{v}; \mathbf{t})$:

$$\mathcal{L}_{\text{InfoNCE}} \leq \log N - I(\mathbf{v}; \mathbf{t})$$

Minimizing InfoNCE loss maximizes a lower bound on mutual information, pushing image and text embeddings of the same concept together.

### Temperature Parameter

$\tau$ controls embedding space "temperature":

| $\tau$ | Effect |
|--------|--------|
| $\tau \to 0$ | Softmax concentrates on highest-similarity pair; very hard negatives |
| $\tau = 0.07$ | CLIP default — sharp but stable distribution |
| $\tau \to \infty$ | Uniform distribution over all pairs; no discriminative signal |

In CLIP, $\log(1/\tau)$ is a learnable scalar initialized to $\log(1/0.07) \approx 2.66$.

### Python Implementation

```python
import numpy as np

def clip_loss(image_emb: np.ndarray, text_emb: np.ndarray, tau: float = 0.07) -> float:
    """
    image_emb: (N, D) L2-normalized image embeddings
    text_emb:  (N, D) L2-normalized text embeddings
    Returns: scalar CLIP InfoNCE loss
    """
    N = image_emb.shape[0]
    # Similarity matrix (already normalized, so = cosine similarity)
    S = image_emb @ text_emb.T / tau   # (N, N)

    # Cross-entropy: targets are the diagonal (matching pairs)
    targets = np.arange(N)

    def cross_entropy(logits, targets):
        m = logits.max(axis=1, keepdims=True)
        log_softmax = logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
        return -log_softmax[np.arange(len(targets)), targets].mean()

    L_image = cross_entropy(S,   targets)    # row-wise: image finds its text
    L_text  = cross_entropy(S.T, targets)    # col-wise: text finds its image
    return float((L_image + L_text) / 2)
```

---

## Vision Transformer (ViT) — Patch Embedding

### From Pixels to Patches

ViT (Dosovitskiy et al., 2020) treats an image as a sequence of fixed-size patches, just like a Transformer treats text as a sequence of tokens.

For an image of size $H \times W$ with patch size $P$:

$$N_{\text{patches}} = \frac{H}{P} \times \frac{W}{P}$$

Each patch $p_i \in \mathbb{R}^{P \times P \times C}$ (height × width × channels) is flattened to $\mathbb{R}^{P^2 C}$ and linearly projected to $\mathbb{R}^d$:

$$z_i = p_i^{\text{flat}} W_E + \mathbf{e}_i^{\text{pos}}, \qquad W_E \in \mathbb{R}^{P^2C \times d}$$

where $\mathbf{e}_i^{\text{pos}}$ is a learned positional embedding for patch $i$.

### [CLS] Token

ViT prepends a learnable `[CLS]` token (class token) to the patch sequence. After passing through $L$ Transformer encoder blocks, the `[CLS]` token's output embedding is used as the global image representation:

```
Input:   [[CLS], patch_1, patch_2, ..., patch_N]  ← N+1 tokens
Output:  [CLS_out, out_1, out_2, ..., out_N]
                ↑ used as image embedding
```

### Patch Embedding Pipeline

```python
def extract_patches(image: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """
    image: (H, W, C) float32
    Returns: (N_patches, patch_size*patch_size*C) flattened patches
    """
    H, W, C = image.shape
    assert H % patch_size == 0 and W % patch_size == 0
    patches = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size, :]   # (P, P, C)
            patches.append(patch.flatten())                      # (P*P*C,)
    return np.array(patches)   # (N_patches, P*P*C)

def patch_embed(patches: np.ndarray, W_E: np.ndarray, pos_embed: np.ndarray,
                cls_token: np.ndarray) -> np.ndarray:
    """
    patches:   (N, P*P*C)
    W_E:       (P*P*C, d)
    pos_embed: (N+1, d)  — positional embeddings including [CLS]
    cls_token: (1, d)    — learnable [CLS] embedding
    Returns:   (N+1, d)  — token sequence ready for Transformer
    """
    z = patches @ W_E                           # (N, d)
    z = np.concatenate([cls_token, z], axis=0)  # (N+1, d)
    z = z + pos_embed                            # (N+1, d)
    return z
```

### ViT Architecture Sizes

| Model | Patch | Layers | d_model | Heads | Params |
|-------|-------|--------|---------|-------|--------|
| ViT-Ti | 16 | 12 | 192 | 3 | 6M |
| ViT-S | 16 | 12 | 384 | 6 | 22M |
| ViT-B | 16 | 12 | 768 | 12 | 86M |
| ViT-L | 16 | 24 | 1024 | 16 | 307M |
| ViT-H | 14 | 32 | 1280 | 16 | 632M |

For CLIP: ViT-B/32 uses 32×32 patches; ViT-L/14 uses 14×14 patches (more patches → richer spatial information).

---

## Cross-Modal Attention

### How Vision and Language Interact

In image captioning models (e.g., Flamingo, BLIP-2), the language decoder attends to vision encoder outputs via cross-attention:

$$\text{CrossAttn}(Q, K, V) = \text{softmax}\!\left(\frac{Q_{\text{text}} K_{\text{image}}^T}{\sqrt{d_k}}\right) V_{\text{image}}$$

Where:
- $Q$ = query from the text decoder's current hidden state: $Q = H_{\text{text}} W^Q$
- $K, V$ = keys and values from the image encoder's patch embeddings: $K = H_{\text{image}} W^K$, $V = H_{\text{image}} W^V$

Each decoder step can "look at" any image patch (spatial attention over patches).

### Gated Cross-Attention (Flamingo)

Flamingo (Alayrac et al., 2022) adds cross-attention layers between pre-trained frozen LLM layers via gating:

$$h = h + \text{tanh}(\alpha) \cdot \text{CrossAttn}(h, x_{\text{image}})$$

where $\alpha$ is a learnable scalar initialized to 0. At initialization, the cross-attention contribution is zero — the model starts as the original LLM and gradually learns to incorporate image information.

### BLIP-2 Q-Former

BLIP-2 (Li et al., 2023) introduces a lightweight Querying Transformer (Q-Former) between the frozen image encoder and frozen LLM:

```
Frozen ViT → Q-Former (32 learnable query tokens, cross-attention to image patches)
           → LLM input projections
           → Frozen LLM
```

Only the Q-Former (188M params) is trained; both the ViT and LLM are frozen. This efficiently bridges vision and language without full fine-tuning.

---

## Image Captioning Pipeline

### Architecture

```
Image → ViT patches → Encoder (self-attention over patches)
                    ↓ cross-attention keys/values
Text prefix → Transformer Decoder → next token prediction
```

The decoder generates captions autoregressively, conditioning each token on:
1. All previously generated tokens (causal self-attention)
2. All image patch embeddings (cross-attention)

### Training: Teacher Forcing

During training, the full ground-truth caption is fed as input (teacher forcing). The model predicts each token from the previous ground-truth token:

```
Input:  [BOS] "a" "dog" "running"
Target:       "a" "dog" "running" [EOS]
Loss: CE on each position (sum over all caption tokens)
```

Only caption tokens contribute to the loss — the image patch embeddings don't have a target output.

### Inference: Autoregressive Decoding

```python
def generate_caption(image_enc_out: np.ndarray, model, max_len: int = 50,
                     start_token: int = 0, end_token: int = 1) -> list:
    tokens = [start_token]
    for _ in range(max_len):
        logits = model.decode(
            decoder_input=tokens,
            encoder_output=image_enc_out,   # cross-attention keys/values
        )
        next_token = logits[-1].argmax()
        tokens.append(int(next_token))
        if next_token == end_token:
            break
    return tokens
```

**Beam search** for captioning: maintain top-$k$ partial captions at each step, expand each, keep top-$k$ by total log-probability.

---

## Zero-Shot Image Classification with CLIP

### How It Works

CLIP enables zero-shot classification by comparing an image embedding to text embeddings of class descriptions — no fine-tuning needed.

1. For each class $c$: encode the text prompt `f"a photo of a {c}"` → $\mathbf{t}_c \in \mathbb{R}^d$
2. Encode the query image → $\mathbf{v} \in \mathbb{R}^d$
3. Predict: $\hat{y} = \arg\max_c \cos(\mathbf{v}, \mathbf{t}_c)$

$$P(y = c \mid \mathbf{v}) = \frac{e^{\cos(\mathbf{v}, \mathbf{t}_c)/\tau}}{\sum_{c'} e^{\cos(\mathbf{v}, \mathbf{t}_{c'})/\tau}}$$

### Prompt Engineering for CLIP

The text prompt template dramatically affects accuracy:

| Template | ImageNet accuracy |
|----------|------------------|
| `f"{classname}"` | ~60% |
| `f"a photo of {classname}"` | ~63% |
| `f"a photo of a {classname}"` | ~76% |
| Ensemble of 80 templates | ~76.2% |

OpenAI found that averaging embeddings over 80 template variations (e.g., "itap of a {c}", "a bad photo of a {c}", "a photo of a small {c}") improved accuracy by ~3–4 points.

### Prompt Ensemble

```python
TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "art of a {}.",
    "a painting of the {}.",
    "a photo of the small {}.",
    "a photo of the large {}.",
]

def ensemble_text_embedding(encoder, classname: str) -> np.ndarray:
    embeddings = []
    for template in TEMPLATES:
        text = template.format(classname)
        emb  = encoder(text)                          # (d,)
        emb  = emb / np.linalg.norm(emb)             # L2 normalize
        embeddings.append(emb)
    mean_emb = np.array(embeddings).mean(axis=0)
    return mean_emb / np.linalg.norm(mean_emb)       # renormalize
```

### CLIP Applications

| Task | How |
|------|-----|
| Zero-shot classification | Image embedding vs. class text embeddings |
| Image-text retrieval | Return images with highest cosine sim to query text |
| Text-to-image generation | CLIP score as guidance signal (DALL-E, Stable Diffusion) |
| Visual Q&A | CLIP embeddings as conditioning for a language model |
| Image clustering | Cluster image embeddings in CLIP space |
| OOD detection | Low max-cosine-similarity across all class embeddings = unknown |

---

## Interview Q&A

**Q: Derive the InfoNCE loss for CLIP and explain why the temperature τ matters.**
**A:** Given a batch of N matched (image, text) pairs, normalize both embeddings to unit sphere. Compute $S_{ij} = \mathbf{v}_i \cdot \mathbf{t}_j / \tau$. For each image $i$, the target is the $i$-th text: $\mathcal{L}_{\text{image}} = -\frac{1}{N}\sum_i \log \text{softmax}(S_i)[i]$. Symmetrically for text. Temperature $\tau$ scales the logits: small $\tau$ makes the distribution sharper (harder negative mining — the model must distinguish very similar negatives), large $\tau$ makes it uniform (easy loss, less discriminative). CLIP learns $\log(1/\tau)$ as a parameter, starting at 2.66 ($\tau=0.07$).

**Q: How does ViT differ from a CNN for image classification?**
**A:** CNNs use sliding local kernels that build up hierarchical local features. ViT splits the image into fixed patches and applies global self-attention across all patches from the start. Key differences: (1) ViT has no inductive biases for locality or translation equivariance — it must learn spatial relationships from data. (2) ViT scales better with more data (less bias = learns more from scale). (3) CNNs outperform ViT on small datasets; ViT outperforms CNNs on ImageNet-21K+ scale. (4) ViT attention maps provide interpretability — each patch's attention to other patches.

**Q: What is the role of the [CLS] token in ViT?**
**A:** The [CLS] (class) token is a learnable embedding prepended to the patch sequence. Since all tokens attend to each other via global self-attention, the [CLS] token aggregates information from all patches across all Transformer layers. Its final output embedding is used as the global image representation (passed to the classification head). It plays the same role as [CLS] in BERT — a "collecting" token that doesn't correspond to any input position.

**Q: How does BLIP-2's Q-Former avoid catastrophic forgetting?**
**A:** Both the image encoder and LLM are frozen throughout Q-Former training. Only the Q-Former parameters (32 learnable query tokens, cross-attention layers, FFN) are updated. The 32 query tokens act as a learned interface: they attend to the image patches (cross-attention) and then their outputs are projected into the LLM's input space. Because the LLM sees only the Q-Former outputs (not raw image patches), the LLM's weights never change. The Q-Former learns to compress 256+ patch embeddings into 32 vectors that contain the LLM-relevant visual information.

**Q: Why does CLIP work for zero-shot classification without any class-specific training?**
**A:** CLIP trains on 400M (image, alt-text) pairs scraped from the web. The text encoder sees class names in natural context ("a photo of a dog by the lake") while the image encoder sees the corresponding photos. The contrastive loss aligns their embedding spaces such that the text "dog" ends up close to dog images — not because CLIP was trained on explicit classification labels, but because dogs appear in image-text pairs. At test time, comparing an image embedding to "a photo of a dog" vs. "a photo of a cat" exploits this learned alignment.

**Q: What is gated cross-attention in Flamingo and why is it initialized to zero?**
**A:** Flamingo inserts cross-attention layers between frozen LLM layers. The gate $\alpha$ is a learnable scalar with output $\text{tanh}(\alpha) \cdot \text{CrossAttn}(h, x_{\text{image}})$. At initialization, $\alpha = 0 \Rightarrow \text{tanh}(0) = 0$, so the cross-attention contributes nothing. The model starts as the original frozen LLM. As training progresses, $\alpha$ grows and cross-attention contribution increases. This prevents the random initialized cross-attention weights from corrupting the frozen LLM's language modeling capability at training start.

**Q: How do you evaluate a zero-shot CLIP model on a downstream classification task?**
**A:** Encode all class descriptions using the text encoder → $\{\mathbf{t}_c\}$. For each test image: encode with image encoder → $\mathbf{v}$, compute $\cos(\mathbf{v}, \mathbf{t}_c)$ for all classes, predict $\hat{y} = \arg\max_c$. Metrics: top-1 and top-5 accuracy vs. ground truth labels. For comparison, report: (1) zero-shot CLIP accuracy, (2) few-shot fine-tuned accuracy (how many shots needed to match CLIP zero-shot), (3) full supervised fine-tuned accuracy (upper bound). CLIP ViT-L/14 achieves 76.2% on ImageNet zero-shot, matching ResNet-101 supervised.

---

## Resources

**Papers:**
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — Radford et al., 2021
- [An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al., 2020 (ViT)
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) — Alayrac et al., 2022
- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and LLMs](https://arxiv.org/abs/2301.12597) — Li et al., 2023
- [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) — Liu et al., 2023
- [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) — Oord et al., 2018 (InfoNCE origin)

**Implementations:**
- OpenCLIP — open-source CLIP training + weights
- HuggingFace `CLIPModel` — inference-ready CLIP with `AutoProcessor`

---

*Modules 01–13 complete.*
