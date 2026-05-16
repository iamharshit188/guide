# Module 13 — Multimodal Models: CLIP, ViT, and Image-Language Fusion

> **Prerequisites:** Modules 06–07 (attention, transformers).
> **Estimated time:** 8–12 hours

## Why Multimodal?

Text alone describes the world. Images show it. Language models that can only read text are blind to the vast majority of information humans communicate — diagrams, photographs, charts, videos.

Multimodal models combine vision and language so a model can:
- Answer "What is in this image?"
- Retrieve images matching a text query ("a photo of a dog on a beach")
- Generate captions for images
- Reason about charts and diagrams

The core challenge: how do you align representations from two completely different modalities so that "a dog" in text and a photograph of a dog point to the same location in some shared representation space?

```
The alignment problem — bridging two worlds:

Text world:              Image world:
  "a dog"                 [pixels: 224×224×3]
  "cat on a mat"          [pixels: 224×224×3]
  "sunset over ocean"     [pixels: 224×224×3]

How do we measure: does "a dog" match this photo of a dog?

Solution — shared embedding space:
  Text encoder:    "a dog"   ──▶  [0.8, 0.2, -0.1, ...] (512-dim vector)
  Image encoder:   [photo]   ──▶  [0.7, 0.3, -0.2, ...] (512-dim vector)
                                        ↑
                    Similar vectors! cosine similarity ≈ 0.95

  "a cat":         [text]   ──▶  [-0.2, 0.9, 0.4, ...] (512-dim vector)
                                        ↑
                    Far from the dog photo! cosine similarity ≈ 0.1

CLIP (Contrastive Language-Image Pretraining) learns this shared space
by training on 400 million (image, caption) pairs from the internet.
```

---

> **Python prerequisite:** This module uses Python, NumPy, and ML libraries throughout. If you need a foundation or refresher, visit the **Languages → Python** guide and read **Section 21 — Python for ML & AI** before starting.

## 1. Vision Transformers (ViT)

### The Key Idea: Treat Image Patches as Tokens

CNNs process images with sliding convolution kernels. ViT (Dosovitskiy et al. 2020) asks: what if we just treat an image as a sequence of patches and feed it to a Transformer?

For a $224 \times 224$ image with $16 \times 16$ patches:
- Number of patches: $(224/16)^2 = 196$ patches
- Each patch: $16 \times 16 \times 3 = 768$ values → linearly projected to $d_\text{model}$
- Plus 1 CLS token → sequence length = 197

This sequence is processed exactly like text tokens in a standard Transformer encoder.

```
ViT — image to patch tokens:

Input image (224×224):               Patches (16×16 each):
┌─────────────────────┐              [P1][P2][P3]...[P14]
│    [img pixels]     │  split into  [P15][P16]...[P28]
│    224 × 224 × 3    │  ──────────▶ ...
│    (RGB channels)   │              [P183]...[P196]
└─────────────────────┘
                                     Each patch = 16×16×3 = 768 values

Flatten & project:
  P1   ──[Linear 768→512]──▶  e1 ∈ ℝ^512
  P2   ──[Linear 768→512]──▶  e2 ∈ ℝ^512
  ...
  P196 ──[Linear 768→512]──▶  e196 ∈ ℝ^512

Add CLS token + positional encoding:
  [CLS, e1, e2, ..., e196] → sequence of 197 tokens

Feed to Transformer encoder (same as BERT!):
  Each patch attends to every other patch
  CLS token aggregates global image information
  Output CLS representation = image embedding
```

```python
import numpy as np
import math


def section(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


class PatchEmbedding:
    """
    Convert an image tensor to a sequence of patch embeddings.
    
    Image: (H, W, C) → Patches: (n_patches, patch_size²×C) → Embeddings: (n_patches, d_model)
    """
    
    def __init__(self, image_size: int = 32, patch_size: int = 8, in_channels: int = 3, d_model: int = 64, seed: int = 42):
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        
        # Number of patches along each dimension
        self.n_patches_side = image_size // patch_size
        self.n_patches = self.n_patches_side ** 2
        
        # Flattened patch size: P×P×C values per patch
        self.patch_dim = patch_size * patch_size * in_channels
        
        # Linear projection: patch_dim → d_model
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (self.patch_dim + d_model))
        self.projection = rng.standard_normal((self.patch_dim, d_model)) * scale
    
    def extract_patches(self, image: np.ndarray) -> np.ndarray:
        """
        image: (H, W, C)
        Returns: (n_patches, patch_dim) — each row is a flattened patch
        """
        H, W, C = image.shape
        p = self.patch_size
        patches = []
        
        for i in range(0, H, p):      # slide down
            for j in range(0, W, p):  # slide right
                patch = image[i:i+p, j:j+p, :]      # (p, p, C)
                patches.append(patch.flatten())       # (p*p*C,)
        
        return np.array(patches)  # (n_patches, patch_dim)
    
    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        image: (H, W, C)
        Returns: (n_patches, d_model)
        """
        patches = self.extract_patches(image)    # (n_patches, patch_dim)
        return patches @ self.projection         # (n_patches, d_model)
    
    def __repr__(self) -> str:
        return (f"PatchEmbedding(image={self.image_size}×{self.image_size}, "
                f"patch={self.patch_size}×{self.patch_size}, "
                f"n_patches={self.n_patches}, d_model={self.d_model})")


section("Patch Embedding")
patch_embed = PatchEmbedding(image_size=32, patch_size=8, in_channels=3, d_model=64)
print(patch_embed)

# Simulate a batch of images
rng = np.random.default_rng(42)
image = rng.random((32, 32, 3))         # one image: (H, W, C)
patches = patch_embed.forward(image)    # (n_patches, d_model)

print(f"\nImage shape:  {image.shape}")
print(f"Patch count:  {patch_embed.n_patches} patches of size {patch_embed.patch_size}×{patch_embed.patch_size}")
print(f"Patch dim:    {patch_embed.patch_dim} values per patch")
print(f"Embeddings:   {patches.shape}")  # (16, 64)
```

### CLS Token and Position Encoding

```python
def add_cls_and_position_encoding(patch_embeddings: np.ndarray, d_model: int, seed: int = 0) -> np.ndarray:
    """
    Add a learnable CLS token (prepended) and positional encodings.
    
    patch_embeddings: (n_patches, d_model)
    Returns: (n_patches + 1, d_model)
    
    CLS token: learnable embedding initialized to zeros.
               After Transformer layers, CLS output = image representation.
    
    Positional encoding: unlike text, image patches have 2D positions.
    We use 1D sinusoidal encoding over the flattened patch sequence (simpler, works well).
    """
    n_patches, d = patch_embeddings.shape
    
    # CLS token: learnable, here initialized to zeros
    cls_token = np.zeros((1, d))  # (1, d_model)
    
    # Prepend CLS: (1 + n_patches, d_model)
    x = np.concatenate([cls_token, patch_embeddings], axis=0)
    
    seq_len = x.shape[0]  # n_patches + 1
    
    # Sinusoidal position encoding (same as Module 07)
    pos = np.arange(seq_len)[:, None]         # (seq_len, 1)
    dim = np.arange(d)[None, :]               # (1, d_model)
    angles = pos / (10000 ** (dim // 2 * 2 / d))
    
    pe = np.where(dim % 2 == 0, np.sin(angles), np.cos(angles))
    
    return x + pe  # position information added to each token


section("CLS Token and Positional Encoding")
d_model = 64
patches_with_pos = add_cls_and_position_encoding(patches, d_model)
print(f"After CLS + PE: {patches_with_pos.shape}")  # (17, 64): 16 patches + 1 CLS
```

### Full ViT Encoder

```python
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class MultiHeadSelfAttention:
    """Multi-head self-attention (same as Module 07 — both text and image use same mechanism)."""
    
    def __init__(self, d_model: int, n_heads: int, seed: int = 42):
        assert d_model % n_heads == 0
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model * 2))
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model
        
        self.W_q = rng.standard_normal((d_model, d_model)) * scale
        self.W_k = rng.standard_normal((d_model, d_model)) * scale
        self.W_v = rng.standard_normal((d_model, d_model)) * scale
        self.W_o = rng.standard_normal((d_model, d_model)) * scale
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (seq_len, d_model) → output: (seq_len, d_model)"""
        seq_len, d = x.shape
        
        Q = x @ self.W_q  # (seq_len, d_model)
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Split into heads: (n_heads, seq_len, d_k)
        def split(t):
            return t.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        
        Q, K, V = split(Q), split(K), split(V)
        
        # Scaled dot-product attention per head
        scores = Q @ K.transpose(0, 2, 1) / math.sqrt(self.d_k)  # (n_heads, seq, seq)
        attn = softmax(scores, axis=-1)
        out = attn @ V  # (n_heads, seq, d_k)
        
        # Merge heads and project
        out = out.transpose(1, 0, 2).reshape(seq_len, d)
        return out @ self.W_o


class LayerNorm:
    """Layer normalization: normalize across d_model dimension."""
    
    def __init__(self, d_model: int):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + 1e-5) + self.beta


class ViTEncoderBlock:
    """
    One Transformer encoder block for ViT.
    Identical to text Transformer block — images and text share architecture.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + d_ff))
        
        self.attention = MultiHeadSelfAttention(d_model, n_heads, seed)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Feed-forward: d_model → 4×d_model → d_model (standard expansion ratio)
        self.W1 = rng.standard_normal((d_model, d_ff)) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.standard_normal((d_ff, d_model)) * scale
        self.b2 = np.zeros(d_model)
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation — used in ViT/BERT instead of ReLU."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Pre-norm Transformer block: norm before attention (ViT convention)."""
        # Self-attention with residual
        x = x + self.attention.forward(self.norm1.forward(x))
        
        # FFN with residual
        h = self.gelu(self.norm2.forward(x) @ self.W1 + self.b1)
        x = x + h @ self.W2 + self.b2
        
        return x


class ViT:
    """
    Vision Transformer (ViT) image encoder.
    
    Architecture:
    Image → Patch Embeddings → CLS + PE → N Transformer Blocks → CLS output
    
    The CLS token output is the image representation used for classification/retrieval.
    """
    
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 8,
        in_channels: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        seed: int = 42
    ):
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d_model, seed)
        self.n_patches = self.patch_embed.n_patches
        self.d_model = d_model
        
        d_ff = d_model * 4  # standard FFN expansion
        self.blocks = [
            ViTEncoderBlock(d_model, n_heads, d_ff, seed=seed+i)
            for i in range(n_layers)
        ]
        self.norm = LayerNorm(d_model)
    
    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        image: (H, W, C)
        Returns: (d_model,) — CLS token embedding representing the whole image
        """
        # 1. Patch embeddings + CLS token + positional encoding
        x = self.patch_embed.forward(image)              # (n_patches, d_model)
        x = add_cls_and_position_encoding(x, self.d_model)  # (n_patches+1, d_model)
        
        # 2. Transformer encoder blocks
        for block in self.blocks:
            x = block.forward(x)
        
        x = self.norm.forward(x)
        
        # 3. Extract CLS token (position 0)
        return x[0]  # (d_model,) — global image representation
    
    @property
    def param_count(self) -> int:
        # Rough estimate: 4 attention matrices + 2 FFN matrices per block
        per_block = 4 * self.d_model**2 + 2 * self.d_model * (self.d_model * 4)
        return (self.n_patches + 1) * self.d_model + len(self.blocks) * per_block


section("ViT Forward Pass")
vit = ViT(image_size=32, patch_size=8, in_channels=3, d_model=64, n_heads=4, n_layers=2)
image = rng.random((32, 32, 3))
embedding = vit.forward(image)

print(f"Image shape:     {image.shape}")
print(f"n_patches:       {vit.n_patches}")
print(f"CLS embedding:   {embedding.shape}")  # (64,)
print(f"Approx params:   {vit.param_count:,}")
```

---

## 2. CLIP: Contrastive Language-Image Pretraining

CLIP (Radford et al. 2021) trains a vision encoder and text encoder jointly to produce aligned embeddings. The training signal: 400M (image, text description) pairs from the internet.

### The Contrastive Learning Objective

For a batch of $N$ (image, text) pairs, form an $N \times N$ matrix of cosine similarities. The correct pairs are on the diagonal. Train to maximize diagonal similarities and minimize off-diagonal:

$$\mathcal{L}_\text{CLIP} = -\frac{1}{2N} \left( \sum_i \log \frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}} + \sum_i \log \frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}} \right)$$

- $s_{ij}$: cosine similarity between image $i$ embedding and text $j$ embedding
- $\tau$ (temperature): controls sharpness of the distribution; learnable in CLIP

This is cross-entropy loss applied symmetrically: once for image→text (each image should match its text), once for text→image.

```python
class CLIPTextEncoder:
    """
    Simplified text encoder for CLIP.
    In production: Transformer with BPE tokenization.
    Here: mean of word embeddings + linear projection.
    """
    
    def __init__(self, vocab_size: int = 100, d_embed: int = 32, d_model: int = 64, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.embeddings = rng.standard_normal((vocab_size, d_embed)) * 0.1
        
        scale = np.sqrt(2.0 / (d_embed + d_model))
        self.projection = rng.standard_normal((d_embed, d_model)) * scale
        
        self.vocab: dict[str, int] = {}  # built during tokenize()
        self.d_model = d_model
    
    def tokenize(self, text: str) -> list[int]:
        """Convert text to token IDs (simple word-level tokenization)."""
        tokens = []
        for word in text.lower().split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) % 100  # wrap to vocab_size
            tokens.append(self.vocab[word])
        return tokens or [0]
    
    def forward(self, text: str) -> np.ndarray:
        """text → (d_model,) embedding"""
        tokens = self.tokenize(text)
        
        # Mean of token embeddings (simplified; real CLIP uses full Transformer)
        token_embeds = self.embeddings[tokens]     # (n_tokens, d_embed)
        mean_embed = token_embeds.mean(axis=0)     # (d_embed,) — average pooling
        
        projected = mean_embed @ self.projection   # (d_model,)
        
        # L2 normalize: CLIP embeddings are on the unit sphere
        return projected / (np.linalg.norm(projected) + 1e-9)


class CLIP:
    """
    CLIP model: aligned image and text encoders.
    
    Key property: after training, semantically similar image+text pairs
    have high cosine similarity in the shared embedding space.
    """
    
    def __init__(self, d_model: int = 64, temperature: float = 0.07, seed: int = 42):
        self.image_encoder = ViT(image_size=32, patch_size=8, d_model=d_model, n_layers=2, seed=seed)
        self.text_encoder = CLIPTextEncoder(d_model=d_model, seed=seed+1)
        
        # Temperature τ: CLIP uses a learnable log-temperature, initialized to log(1/0.07)
        self.log_temp = math.log(1.0 / temperature)
        self.d_model = d_model
    
    @property
    def temperature(self) -> float:
        return 1.0 / math.exp(self.log_temp)
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """image: (H, W, C) → (d_model,) unit vector"""
        emb = self.image_encoder.forward(image)
        return emb / (np.linalg.norm(emb) + 1e-9)
    
    def encode_text(self, text: str) -> np.ndarray:
        """text: string → (d_model,) unit vector"""
        return self.text_encoder.forward(text)  # already normalized
    
    def similarity_matrix(self, images: list[np.ndarray], texts: list[str]) -> np.ndarray:
        """
        Compute N×N cosine similarity matrix between image and text embeddings.
        Entry [i,j] = similarity between image i and text j.
        """
        image_embeds = np.array([self.encode_image(img) for img in images])  # (N, d_model)
        text_embeds = np.array([self.encode_text(t) for t in texts])          # (N, d_model)
        
        # Cosine similarity matrix: (N, N)
        # Both are already normalized, so dot product = cosine similarity
        return image_embeds @ text_embeds.T
    
    def contrastive_loss(self, images: list[np.ndarray], texts: list[str]) -> float:
        """
        CLIP InfoNCE contrastive loss.
        
        The diagonal of the similarity matrix should be maximized (matching pairs).
        Off-diagonal should be minimized (non-matching pairs).
        """
        sim = self.similarity_matrix(images, texts)  # (N, N)
        N = len(images)
        
        # Scale by temperature
        logits = sim / self.temperature  # (N, N)
        
        # Image-to-text cross-entropy: for each image, classify which text matches
        # labels are 0, 1, 2, ..., N-1 (diagonal)
        def cross_entropy_rows(logit_matrix: np.ndarray) -> float:
            """Cross-entropy along rows (each row = one query, find matching column)."""
            total = 0.0
            for i in range(N):
                row = logit_matrix[i]
                # log-sum-exp for numerical stability
                log_sum_exp = row.max() + math.log(sum(math.exp(v - row.max()) for v in row))
                total += row[i] - log_sum_exp  # log prob of correct class (diagonal)
            return -total / N
        
        # Symmetric: loss from both image-to-text and text-to-image
        loss_i2t = cross_entropy_rows(logits)
        loss_t2i = cross_entropy_rows(logits.T)
        
        return (loss_i2t + loss_t2i) / 2.0
    
    def zero_shot_classify(self, image: np.ndarray, class_names: list[str]) -> list[tuple[float, str]]:
        """
        Zero-shot classification: no fine-tuning needed.
        
        For each class, create a text prompt "a photo of a {class_name}".
        Return probabilities for each class.
        
        This is CLIP's famous capability — classify images using natural language labels.
        """
        image_embed = self.encode_image(image)  # (d_model,)
        
        # Create text embeddings for each class
        prompts = [f"a photo of a {name}" for name in class_names]
        text_embeds = np.array([self.encode_text(p) for p in prompts])  # (n_classes, d_model)
        
        # Cosine similarities
        similarities = text_embeds @ image_embed  # (n_classes,)
        
        # Softmax to get probabilities
        logits = similarities / self.temperature
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits) / np.exp(logits).sum()
        
        result = sorted(zip(probs.tolist(), class_names), reverse=True)
        return result
    
    def image_retrieval(
        self,
        query_text: str,
        image_library: list[tuple[np.ndarray, str]],
        top_k: int = 3
    ) -> list[tuple[float, str]]:
        """
        Text-to-image retrieval: find images matching a text description.
        
        query_text:    the search query
        image_library: list of (image_array, image_name) tuples
        """
        query_embed = self.encode_text(query_text)       # (d_model,)
        
        scores = []
        for img, name in image_library:
            img_embed = self.encode_image(img)
            sim = float(np.dot(query_embed, img_embed))  # cosine (both normalized)
            scores.append((sim, name))
        
        return sorted(scores, reverse=True)[:top_k]


section("CLIP Demo")
clip = CLIP(d_model=64, temperature=0.07)

# Simulate a small batch for contrastive training
images = [rng.random((32, 32, 3)) for _ in range(4)]
texts = [
    "a photo of a cat",
    "an image of a dog",
    "a picture of a mountain",
    "a photograph of the ocean",
]

loss = clip.contrastive_loss(images, texts)
print(f"Contrastive loss (untrained): {loss:.4f}")
print("(After training, matching pairs should have high similarity → lower loss)")

# Zero-shot classification
section("Zero-Shot Classification")
test_image = rng.random((32, 32, 3))
class_names = ["cat", "dog", "bird", "car", "airplane", "flower"]

predictions = clip.zero_shot_classify(test_image, class_names)
print("Zero-shot classification probabilities:")
for prob, name in predictions:
    bar = "█" * int(prob * 40)
    print(f"  {name:10s}: {prob:.4f} {bar}")

# Image retrieval
section("Text-to-Image Retrieval")
image_library = [
    (rng.random((32, 32, 3)), f"image_{i}") for i in range(10)
]
query = "a sunny beach with waves"
results = clip.image_retrieval(query, image_library, top_k=3)
print(f"Query: '{query}'")
print("Top 3 matches:")
for score, name in results:
    print(f"  {name}: similarity = {score:.4f}")
```

---

## 3. Cross-Modal Attention (Image + Text Fusion)

For tasks like Visual Question Answering (VQA) and image captioning, you need the model to jointly attend over both image patches and text tokens.

### Cross-Attention Mechanism

```python
class CrossModalAttention:
    """
    Cross-attention between two modalities.
    
    In image-to-text: queries come from text, keys/values from image patches.
    The text can "look at" relevant image regions.
    
    In text-to-image: reverse direction — image patches attend to text context.
    """
    
    def __init__(self, d_model: int, n_heads: int, seed: int = 42):
        assert d_model % n_heads == 0
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model * 2))
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_model = d_model
        
        # Separate projections for each modality
        self.W_q = rng.standard_normal((d_model, d_model)) * scale  # from query modality
        self.W_k = rng.standard_normal((d_model, d_model)) * scale  # from key modality
        self.W_v = rng.standard_normal((d_model, d_model)) * scale  # from value modality
        self.W_o = rng.standard_normal((d_model, d_model)) * scale
    
    def forward(self, query: np.ndarray, context: np.ndarray) -> np.ndarray:
        """
        query:   (seq_q, d_model) — e.g., text tokens asking questions
        context: (seq_k, d_model) — e.g., image patches providing visual context
        
        Returns: (seq_q, d_model) — query tokens updated with context information
        """
        seq_q = query.shape[0]
        seq_k = context.shape[0]
        
        Q = query @ self.W_q     # (seq_q, d_model) — from text
        K = context @ self.W_k   # (seq_k, d_model) — from image
        V = context @ self.W_v   # (seq_k, d_model) — from image
        
        # Split into heads
        def split_heads(t, seq_len):
            return t.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        
        Q = split_heads(Q, seq_q)  # (n_heads, seq_q, d_k)
        K = split_heads(K, seq_k)  # (n_heads, seq_k, d_k)
        V = split_heads(V, seq_k)  # (n_heads, seq_k, d_k)
        
        # Cross-attention: text queries attend over image key-values
        scores = Q @ K.transpose(0, 2, 1) / math.sqrt(self.d_k)  # (n_heads, seq_q, seq_k)
        attn_weights = softmax(scores, axis=-1)                   # attend over image patches
        
        out = attn_weights @ V  # (n_heads, seq_q, d_k)
        
        # Merge heads
        out = out.transpose(1, 0, 2).reshape(seq_q, self.d_model)
        return out @ self.W_o


section("Cross-Modal Attention")
d_model = 64
n_heads = 4
n_image_patches = 16
n_text_tokens = 8

cross_attn = CrossModalAttention(d_model=d_model, n_heads=n_heads)

# Simulate image patch embeddings and text token embeddings
image_patches = rng.standard_normal((n_image_patches, d_model))
text_tokens = rng.standard_normal((n_text_tokens, d_model))

# Text queries attending over image patches
fused_text = cross_attn.forward(query=text_tokens, context=image_patches)
print(f"Text tokens:     {text_tokens.shape}")
print(f"Image patches:   {image_patches.shape}")
print(f"Fused (text+img):{fused_text.shape}")  # (8, 64) — text updated with image context
```

### VQA Architecture Pattern

```python
class SimpleVQA:
    """
    Simple Visual Question Answering model.
    
    Pipeline:
    1. Encode image with ViT → patch embeddings
    2. Encode question with text encoder → token embeddings
    3. Cross-attention: question tokens attend over image patches
    4. Pool + classify → answer
    """
    
    def __init__(self, d_model: int = 64, n_heads: int = 4, n_answers: int = 10, seed: int = 42):
        rng = np.random.default_rng(seed)
        
        # Encoders
        self.image_encoder = ViT(image_size=32, patch_size=8, d_model=d_model, n_layers=2, seed=seed)
        self.text_encoder = CLIPTextEncoder(d_model=d_model, seed=seed+1)
        
        # Cross-modal attention: question attends over image
        self.cross_attention = CrossModalAttention(d_model=d_model, n_heads=n_heads, seed=seed+2)
        
        # Classification head: mean-pooled cross-attention output → answer logits
        scale = np.sqrt(2.0 / (d_model + n_answers))
        self.classifier = rng.standard_normal((d_model, n_answers)) * scale
        self.n_answers = n_answers
    
    def forward(self, image: np.ndarray, question: str) -> np.ndarray:
        """
        image:    (H, W, C)
        question: string
        Returns:  (n_answers,) logits — argmax is the predicted answer
        """
        # 1. Encode image as sequence of patch embeddings
        patch_embed = self.image_encoder.patch_embed.forward(image)  # (n_patches, d_model)
        
        # 2. Encode question as token embeddings
        question_embed = self.text_encoder.forward(question)  # (d_model,) — single vector
        question_seq = question_embed[None, :]                # (1, d_model) — treat as single token
        
        # 3. Cross-attention: question attends over image patches
        fused = self.cross_attention.forward(
            query=question_seq,    # (1, d_model) — question
            context=patch_embed    # (n_patches, d_model) — image context
        )  # (1, d_model)
        
        # 4. Classify fused representation
        fused_pooled = fused.mean(axis=0)           # (d_model,) — global context
        logits = fused_pooled @ self.classifier      # (n_answers,)
        
        return logits
    
    def predict(self, image: np.ndarray, question: str, answer_labels: list[str]) -> str:
        """Return the predicted answer as a string."""
        logits = self.forward(image, question)
        best_idx = int(logits.argmax())
        return answer_labels[best_idx] if best_idx < len(answer_labels) else f"answer_{best_idx}"


section("VQA Demo")
vqa = SimpleVQA(d_model=64, n_answers=5)

test_image = rng.random((32, 32, 3))
answer_labels = ["red", "blue", "large", "small", "yes"]

questions = [
    "What color is the object?",
    "Is the image large?",
    "What size is it?",
]

print("VQA Predictions:")
for q in questions:
    logits = vqa.forward(test_image, q)
    probs = softmax(logits)
    answer = vqa.predict(test_image, q, answer_labels)
    print(f"  Q: {q}")
    print(f"     A: {answer} (confidence: {probs.max():.3f})")
```

---

## 4. Image Captioning

Captioning is sequence generation conditioned on an image. The decoder generates tokens one-by-one, attending to both past tokens and image features.

```python
class ImageCaptioner:
    """
    Simple image captioning using a greedy decoder.
    
    Architecture:
    Image → ViT encoder → image tokens
    Decoder generates caption token-by-token using cross-attention to image.
    """
    
    VOCAB = ["<start>", "<end>", "a", "the", "photo", "image", "of", "showing",
             "cat", "dog", "bird", "car", "person", "tree", "sky", "water",
             "red", "blue", "green", "large", "small", "bright", "dark",
             "sitting", "running", "flying", "standing", "looking"]
    VOCAB_SIZE = len(VOCAB)
    
    def __init__(self, d_model: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = 0.1
        
        self.d_model = d_model
        self.word_embeddings = rng.standard_normal((self.VOCAB_SIZE, d_model)) * scale
        
        # Simple cross-attention: token embeddings attend over image features
        self.cross_attn = CrossModalAttention(d_model=d_model, n_heads=2, seed=seed)
        
        # Output projection: d_model → vocab_size
        self.lm_head = rng.standard_normal((d_model, self.VOCAB_SIZE)) * scale
        
        # Build word → index mapping
        self.word2idx = {w: i for i, w in enumerate(self.VOCAB)}
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """Simulate image features: (n_features, d_model)."""
        # In production: ViT output. Here: random but deterministic.
        rng = np.random.default_rng(abs(int(image.sum() * 1000)) % (2**31))
        return rng.standard_normal((8, self.d_model)) * 0.1
    
    def decode_step(self, token_idx: int, image_features: np.ndarray) -> np.ndarray:
        """
        Given current token and image features, predict next token logits.
        
        token_idx:      index of current token in vocab
        image_features: (n_img_tokens, d_model)
        Returns: (vocab_size,) logits for next token
        """
        # Embed current token
        token_embed = self.word_embeddings[token_idx][None, :]  # (1, d_model)
        
        # Cross-attend over image features
        fused = self.cross_attn.forward(query=token_embed, context=image_features)
        
        # Project to vocabulary
        return fused[0] @ self.lm_head  # (vocab_size,)
    
    def generate(self, image: np.ndarray, max_length: int = 8) -> str:
        """
        Greedily generate a caption for the image.
        
        max_length: maximum number of tokens to generate
        """
        image_features = self.encode_image(image)
        
        # Start with <start> token
        generated = [self.word2idx["<start>"]]
        
        for _ in range(max_length):
            current_token = generated[-1]
            logits = self.decode_step(current_token, image_features)
            
            # Greedy decoding: always pick highest-probability next token
            next_token = int(logits.argmax())
            generated.append(next_token)
            
            if next_token == self.word2idx["<end>"]:
                break
        
        # Convert indices back to words (exclude <start> and <end>)
        words = [
            self.VOCAB[idx] for idx in generated[1:]
            if self.VOCAB[idx] not in ["<start>", "<end>"]
        ]
        return " ".join(words) if words else "(empty caption)"


section("Image Captioning Demo")
captioner = ImageCaptioner(d_model=32)

test_images = [rng.random((32, 32, 3)) for _ in range(4)]

print("Generated captions (untrained model — random but demonstrates pipeline):")
for i, img in enumerate(test_images):
    caption = captioner.generate(img, max_length=6)
    print(f"  Image {i}: '{caption}'")
```

---

## 5. Contrastive Training Loop

```python
def contrastive_training_demo(clip_model: CLIP, n_steps: int = 50) -> list[float]:
    """
    Simulate one training step of CLIP contrastive learning.
    Shows how the loss evolves as embeddings align.
    """
    losses = []
    
    # Synthetic dataset: positive pairs are (image, matching_text)
    # We simulate "training" by manually nudging text encoder weights
    
    image_texts = [
        (rng.random((32, 32, 3)), "a photo of a cat sitting"),
        (rng.random((32, 32, 3)), "a picture of a running dog"),
        (rng.random((32, 32, 3)), "an image of a blue sky"),
        (rng.random((32, 32, 3)), "a photograph of ocean waves"),
    ]
    
    images = [x[0] for x in image_texts]
    texts = [x[1] for x in image_texts]
    
    for step in range(n_steps):
        loss = clip_model.contrastive_loss(images, texts)
        losses.append(loss)
        
        # Manually perturb text encoder weights (simulates gradient step)
        clip_model.text_encoder.projection += (
            np.random.default_rng(step).standard_normal(clip_model.text_encoder.projection.shape) * 0.001
        )
        
        if step % 10 == 0:
            sim_matrix = clip_model.similarity_matrix(images, texts)
            avg_diag = float(np.diag(sim_matrix).mean())
            print(f"  Step {step:3d}: loss={loss:.4f}, avg_match_sim={avg_diag:.4f}")
    
    return losses


section("CLIP Contrastive Training")
clip_model = CLIP(d_model=64)
losses = contrastive_training_demo(clip_model, n_steps=50)
print(f"\nInitial loss: {losses[0]:.4f}")
print(f"Final loss:   {losses[-1]:.4f}")
```

---

## 6. Interview Q&A

**Q: How does ViT differ from CNNs?**
CNNs: local receptive fields, translation equivariance, inductive bias toward local features. ViT: global receptive field from the first layer (every patch attends to every other patch), no built-in spatial bias. ViT needs more data to learn spatial structure that CNNs get for free. ViT scales better with data and compute; CNNs are more data-efficient.

**Q: What is the CLS token in ViT and why does it work?**
The CLS (classification) token is a learnable embedding prepended to the patch sequence. It has no spatial content — it "summarizes" all patches through self-attention. After N Transformer layers, CLS has attended to all patches and accumulates global information. Its output is used as the image representation for downstream tasks.

**Q: Explain CLIP's contrastive training objective.**
For a batch of N (image, text) pairs, CLIP computes an N×N similarity matrix. The N diagonal entries are the correct matches. It applies cross-entropy loss over rows (each image must identify its matching text from N candidates) and columns (each text must identify its matching image). The loss drives diagonal entries high and off-diagonal entries low. Scaling by temperature controls distribution sharpness.

**Q: What is zero-shot classification in CLIP?**
No fine-tuning needed. For each class, create a text prompt ("a photo of a {class}"). Encode all prompts. Encode the test image. Compute cosine similarities between image and all class text embeddings. Softmax these similarities to get class probabilities. The class with highest similarity is the prediction.

**Q: What is cross-modal attention?**
Standard self-attention: queries, keys, values all from the same sequence. Cross-modal attention: queries from one modality, keys/values from another. For VQA: text queries attend over image patch keys/values — each word in the question can look at relevant image regions. Enables modality-specific information to flow in one direction.

**Q: What's the difference between CLIP and a captioning model?**
CLIP: embedding model. Given image and text, outputs a score (cosine similarity). No generation. Used for retrieval, zero-shot classification, text-guided image search. Captioning model: generative model. Given image, outputs text. Requires an autoregressive decoder conditioned on image features. CLIP embeddings are often used as input to captioning decoders.

---

## 7. Cheat Sheet

```
VIT ARCHITECTURE
  Input: (H, W, C) image
  Patch: P×P pixels → flatten → project to d_model
  n_patches: (H/P)² + 1 CLS token
  CLS output: global image representation
  
CLIP KEY POINTS
  Train: N pairs → N×N sim matrix → cross-entropy on diagonal
  Temperature τ: controls distribution sharpness (CLIP uses 0.07)
  Both encoders produce unit-norm embeddings
  Zero-shot: classify using text prompts, no labels needed
  Retrieval: cosine similarity between image/text embeddings
  
CROSS-MODAL ATTENTION
  Q from one modality, K/V from other
  VQA: Q=text, K/V=image patches
  Image captioning: Q=generated tokens, K/V=image patches
  
CONTRASTIVE LOSS
  InfoNCE: -log(e^(s_ii/τ) / Σ_j e^(s_ij/τ))
  Applied symmetrically (image→text and text→image)
  Requires large batch size for good negatives (CLIP: 32768)
  
PRETRAINING SCALE
  CLIP: 400M (image, text) pairs, ViT-L/14 (307M params)
  OpenCLIP: open reproduction, ViT-H/14 on LAION-2B
  BLIP: adds generative captioning loss to contrastive
  LLaVA: ViT encoder + linear projection + LLaMA decoder
```

---

## Mini-Project: Image Search Engine

Build a CLIP-inspired semantic image search engine that retrieves images by text description.

```python
# image_search.py
import numpy as np
import math


def section(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ── Simplified encoders ──────────────────────────────────────────
class SimpleImageEncoder:
    """Deterministic image encoder: statistical moments → embedding."""
    
    def __init__(self, d_model: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        # Projection from image statistics to embedding space
        self.W = rng.standard_normal((12, d_model)) * 0.1
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract 12 simple image statistics as features."""
        return np.array([
            image.mean(),                           # average brightness
            image.std(),                            # contrast
            image[:, :, 0].mean(),                  # red channel
            image[:, :, 1].mean(),                  # green channel
            image[:, :, 2].mean(),                  # blue channel
            image[:16, :, :].mean(),                # top half brightness
            image[16:, :, :].mean(),                # bottom half brightness
            image[:, :16, :].mean(),                # left half
            image[:, 16:, :].mean(),                # right half
            image.max(),                            # brightest pixel
            image.min(),                            # darkest pixel
            float(np.diff(image.reshape(-1)).std()), # high-freq detail
        ])
    
    def encode(self, image: np.ndarray) -> np.ndarray:
        features = self.extract_features(image)
        emb = features @ self.W
        return emb / (np.linalg.norm(emb) + 1e-9)


class SimpleTextEncoder:
    """Text encoder: word overlap with color/object vocabulary → embedding."""
    
    COLOR_WORDS = {
        "red": [1,0,0,0,0,0,0,0], "blue": [0,1,0,0,0,0,0,0],
        "green": [0,0,1,0,0,0,0,0], "bright": [0,0,0,1,0,0,0,0],
        "dark": [0,0,0,0,1,0,0,0], "light": [0,0,0,0,0,1,0,0],
        "colorful": [0,0,0,0,0,0,1,0], "warm": [0,0,0,0,0,0,0,1],
    }
    SCENE_WORDS = {
        "outdoor": [1,0,0,0], "indoor": [0,1,0,0],
        "nature": [0,0,1,0], "urban": [0,0,0,1],
    }
    
    def __init__(self, d_model: int = 32, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.W = rng.standard_normal((12, d_model)) * 0.1
    
    def encode(self, text: str) -> np.ndarray:
        words = set(text.lower().split())
        
        color_vec = [0.0] * 8
        for word, vec in self.COLOR_WORDS.items():
            if word in words:
                color_vec = [max(c, v) for c, v in zip(color_vec, vec)]
        
        scene_vec = [0.0] * 4
        for word, vec in self.SCENE_WORDS.items():
            if word in words:
                scene_vec = [max(s, v) for s, v in zip(scene_vec, vec)]
        
        feature_vec = np.array(color_vec + scene_vec)
        emb = feature_vec @ self.W
        return emb / (np.linalg.norm(emb) + 1e-9)


# ── Search Engine ────────────────────────────────────────────────
class ImageSearchEngine:
    
    def __init__(self, d_model: int = 32):
        self.image_encoder = SimpleImageEncoder(d_model=d_model)
        self.text_encoder = SimpleTextEncoder(d_model=d_model)
        self.index: list[dict] = []
    
    def add_image(self, image: np.ndarray, name: str, tags: list[str]) -> None:
        """Index an image with its name and tag list."""
        emb = self.image_encoder.encode(image)
        self.index.append({
            "name": name,
            "tags": tags,
            "embedding": emb,
            "image": image,
        })
    
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Find images matching a text query."""
        query_emb = self.text_encoder.encode(query)
        
        scored = []
        for item in self.index:
            sim = float(np.dot(query_emb, item["embedding"]))
            scored.append({**item, "score": sim})
        
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
    
    def batch_search(self, queries: list[str], top_k: int = 3) -> dict:
        """Run multiple searches at once."""
        return {q: self.search(q, top_k) for q in queries}
    
    def similarity_matrix(self) -> np.ndarray:
        """N×N cosine similarity matrix among all indexed images."""
        embeds = np.array([item["embedding"] for item in self.index])
        return embeds @ embeds.T
    
    def tag_based_recall(self, query: str, top_k: int = 3) -> float:
        """
        Evaluation: fraction of top-k results that have at least one
        query word in their tags (approximate precision).
        """
        results = self.search(query, top_k)
        query_words = set(query.lower().split())
        relevant = sum(
            1 for r in results
            if any(tag.lower() in query_words or w in set(r["tags"])
                   for w in query_words for tag in r["tags"])
        )
        return relevant / max(len(results), 1)


def generate_synthetic_image(category: str, seed: int) -> np.ndarray:
    """Generate a synthetic image with properties matching a category."""
    rng = np.random.default_rng(seed)
    image = rng.random((32, 32, 3)) * 0.3 + 0.35  # base neutral image
    
    modifications = {
        "red":      lambda img: (img.__setitem__((slice(None), slice(None), 0), np.clip(img[:,:,0] + 0.5, 0, 1)), img)[1],
        "blue":     lambda img: (img.__setitem__((slice(None), slice(None), 2), np.clip(img[:,:,2] + 0.5, 0, 1)), img)[1],
        "green":    lambda img: (img.__setitem__((slice(None), slice(None), 1), np.clip(img[:,:,1] + 0.5, 0, 1)), img)[1],
        "bright":   lambda img: np.clip(img + 0.4, 0, 1),
        "dark":     lambda img: np.clip(img - 0.3, 0, 1),
    }
    
    for cat, fn in modifications.items():
        if cat in category.lower():
            image = fn(image)
    
    return image


def main():
    section("Build Image Library")
    
    rng = np.random.default_rng(42)
    engine = ImageSearchEngine(d_model=32)
    
    # Create synthetic image library
    image_catalog = [
        ("sunset_beach",    ["bright", "warm", "outdoor", "nature"]),
        ("forest_path",     ["green", "nature", "outdoor", "dark"]),
        ("city_night",      ["dark", "urban", "outdoor", "light"]),
        ("red_flower",      ["red", "colorful", "nature", "bright"]),
        ("blue_ocean",      ["blue", "nature", "outdoor", "bright"]),
        ("indoor_cafe",     ["warm", "indoor", "light"]),
        ("green_meadow",    ["green", "bright", "outdoor", "nature"]),
        ("dark_forest",     ["dark", "nature", "outdoor", "green"]),
        ("red_car",         ["red", "urban", "bright"]),
        ("blue_sky",        ["blue", "bright", "outdoor", "nature"]),
        ("colorful_market", ["colorful", "bright", "outdoor", "urban"]),
        ("night_city",      ["dark", "urban", "light"]),
    ]
    
    for i, (name, tags) in enumerate(image_catalog):
        # Generate synthetic image based on tags
        main_tag = tags[0]
        image = generate_synthetic_image(main_tag, seed=i)
        engine.add_image(image, name, tags)
    
    print(f"Indexed {len(engine.index)} images")
    
    section("Text-to-Image Search")
    queries = [
        "bright outdoor nature",
        "dark urban city",
        "red colorful",
        "blue outdoor nature",
        "bright indoor warm",
    ]
    
    for query in queries:
        results = engine.search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for r in results:
            print(f"  {r['name']:20s} score={r['score']:.4f}  tags={r['tags']}")
    
    section("Evaluation: Tag-Based Recall")
    print(f"{'Query':<30} {'Recall@3':>10}")
    print("-" * 42)
    for query in queries:
        recall = engine.tag_based_recall(query, top_k=3)
        print(f"{query:<30} {recall:>10.3f}")
    
    section("Image Similarity Matrix (first 5×5)")
    sim_matrix = engine.similarity_matrix()
    print("Cosine similarities between images:")
    names = [item["name"][:12] for item in engine.index[:5]]
    header = "             " + "  ".join(f"{n:>12}" for n in names)
    print(header)
    for i, row_name in enumerate(names):
        row = "  ".join(f"{sim_matrix[i,j]:>12.4f}" for j in range(5))
        print(f"{row_name:13s}{row}")
    
    section("Zero-Shot Classification")
    classes = ["bright outdoor", "dark urban", "red object", "blue nature", "green scene"]
    
    test_img = generate_synthetic_image("bright outdoor", seed=99)
    query_emb = engine.text_encoder.encode("bright outdoor nature")
    
    class_embeds = np.array([engine.text_encoder.encode(c) for c in classes])
    img_emb = engine.image_encoder.encode(test_img)
    
    sims = class_embeds @ img_emb
    probs = softmax(sims / 0.07)
    
    print("Zero-shot classification of 'bright outdoor' image:")
    for cls, prob in sorted(zip(classes, probs.tolist()), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:20s}: {prob:.4f} {bar}")


if __name__ == "__main__":
    main()
```

### Expected Output

```
============================================================
Build Image Library
============================================================
Indexed 12 images

============================================================
Text-to-Image Search
============================================================

Query: 'bright outdoor nature'
  blue_sky             score=0.4821  tags=['blue', 'bright', 'outdoor', 'nature']
  green_meadow         score=0.4503  tags=['green', 'bright', 'outdoor', 'nature']
  sunset_beach         score=0.3891  tags=['bright', 'warm', 'outdoor', 'nature']

============================================================
Zero-Shot Classification
============================================================
Zero-shot classification of 'bright outdoor' image:
  bright outdoor       : 0.8234 ████████████████████████
  green scene          : 0.0721 ██
  blue nature          : 0.0512 █
```
