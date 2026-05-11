# Module 06 — GenAI Core

> **Run:**
> ```bash
> cd src/06-genai
> python word2vec.py
> python attention.py
> python multihead_attention.py
> python positional_encoding.py
> python kv_cache.py
> ```

---

## Prerequisites & Overview

**Prerequisites:** Modules 01–05 (matrix math, softmax, gradient descent, neural network layers). No NLP background needed.
**Estimated time:** 8–12 hours (the attention and KV cache sections are the most important)

### Why This Module Matters
These five primitives — embeddings, attention, multi-head attention, positional encoding, KV cache — are the building blocks of every modern LLM. Understanding them at the mathematical level is what enables you to read a new model paper (LLaMA, Mistral, Gemma) and immediately understand its architectural choices. Module 07 (Transformers) assembles these into a full model.

### Primitive Map

| Primitive | Core Equation | Role in LLMs |
|-----------|--------------|-------------|
| Word embeddings | $\mathbf{e}_w \in \mathbb{R}^d$, trained via skip-gram | Token representation |
| Scaled dot-product attention | $\text{softmax}(QK^T/\sqrt{d_k})V$ | Token-to-token information routing |
| Multi-head attention | $h$ parallel attention heads + projection | Capture multiple relationship types simultaneously |
| Positional encoding | Sinusoidal or RoPE | Inject token order (self-attention is permutation-invariant) |
| KV cache | Store past $K$, $V$ tensors across decoding steps | $O(T^2) \to O(T)$ per-token FLOPs during inference |

### Before You Start
- Know that softmax converts a vector of real numbers to a probability distribution (Module 01/02)
- Understand matrix multiplication shapes: $(m \times k)(k \times n) = (m \times n)$ (Module 01)
- Know what a neural network layer does: $\mathbf{y} = \sigma(W\mathbf{x} + \mathbf{b})$ (Module 05)
- Know what "embedding" means at a high level: a lookup table mapping discrete tokens to dense vectors

### Intuition: Why Attention?
Before attention, RNNs processed tokens sequentially — each token's hidden state carried information about all prior tokens through a **bottleneck**. Attention bypasses the bottleneck: every token can **directly attend** to every other token in $O(1)$ paths. The cost is $O(n^2)$ memory and compute per layer, which Modules 07+ address.

---

## 01 — Word Embeddings

### Distributional Hypothesis

Words that appear in similar contexts have similar meanings. Embedding methods operationalise this: map each token $w$ to a dense vector $\mathbf{e}_w \in \mathbb{R}^d$ such that dot products reflect semantic relatedness.

### Word2Vec — Skip-Gram

Given centre word $w_c$, predict context word $w_o$ within window $m$:

$$P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})}{\sum_{w=1}^{V} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_c})}$$

- $\mathbf{v}_{w_c} \in \mathbb{R}^d$ — **input** (centre) embedding
- $\mathbf{u}_{w_o} \in \mathbb{R}^d$ — **output** (context) embedding
- $V$ — vocabulary size

Training objective (negative log-likelihood over corpus):

$$\mathcal{L} = -\sum_{t=1}^{T} \sum_{-m \le j \le m, j \ne 0} \log P(w_{t+j} \mid w_t)$$

The softmax denominator sums over all $V$ words — $O(V)$ per step, intractable for large vocabularies.

### Negative Sampling

Replace the full softmax with a binary classification: does this (centre, context) pair actually co-occur?

$$\mathcal{L}_{\text{NEG}} = -\log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) - \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n} \left[\log \sigma(-\mathbf{u}_{w_k}^\top \mathbf{v}_{w_c})\right]$$

- $K$ negative samples per positive pair (typical: 5–20 for small corpora, 2–5 for large)
- $P_n(w) \propto f(w)^{3/4}$ — unigram distribution raised to power $3/4$ (smooths out frequent words)
- Gradient complexity: $O(K \cdot d)$ instead of $O(V \cdot d)$

**Gradients:**

$$\frac{\partial \mathcal{L}_{\text{NEG}}}{\partial \mathbf{v}_{w_c}} = (\sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) - 1)\mathbf{u}_{w_o} + \sum_{k=1}^{K} \sigma(\mathbf{u}_{w_k}^\top \mathbf{v}_{w_c})\mathbf{u}_{w_k}$$

$$\frac{\partial \mathcal{L}_{\text{NEG}}}{\partial \mathbf{u}_{w_o}} = (\sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c}) - 1)\mathbf{v}_{w_c}$$

$$\frac{\partial \mathcal{L}_{\text{NEG}}}{\partial \mathbf{u}_{w_k}} = \sigma(\mathbf{u}_{w_k}^\top \mathbf{v}_{w_c})\mathbf{v}_{w_c}$$

### CBOW vs Skip-Gram

| | CBOW | Skip-Gram |
|---|---|---|
| Task | Predict centre from context average | Predict context from centre |
| Training speed | Faster (averages context) | Slower (multiple predictions per centre) |
| Rare words | Worse | Better |
| Frequent words | Better | Acceptable |
| Interview answer | "CBOW averages context embeddings — fast but loses positional info" | "Skip-gram trains $2m$ binary classifiers per token — better for rare words" |

### Subword Models (FastText Extension)

Word2Vec treats "running" and "runner" as unrelated. FastText decomposes each word into character $n$-grams (default: 3–6):

$$\mathbf{v}_w = \frac{1}{|G_w|} \sum_{g \in G_w} \mathbf{z}_g$$

where $G_w$ is the set of $n$-grams in $w$ plus the whole word token `<running>`. Enables OOV (out-of-vocabulary) word representations by summing $n$-gram vectors.

---

## 02 — Sentence Transformers

Word2Vec produces context-independent embeddings ("bank" has one vector regardless of meaning). BERT-family models produce contextualised embeddings, but using `[CLS]` token directly gives poor sentence-level similarity scores.

**Sentence-BERT (SBERT)** fine-tunes a BERT model with Siamese network architecture on NLI data:

$$\text{similarity}(s_1, s_2) = \cos(\text{mean-pool}(\text{BERT}(s_1)),\ \text{mean-pool}(\text{BERT}(s_2)))$$

Mean pooling over token embeddings (not just `[CLS]`) is empirically better for semantic similarity tasks.

**Cosine similarity as a distance:**

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

For unit-normalised vectors, $\cos(\mathbf{a}, \mathbf{b}) = \mathbf{a}^\top \mathbf{b}$, and L2 distance $\|\mathbf{a} - \mathbf{b}\|^2 = 2 - 2\cos(\mathbf{a}, \mathbf{b})$. Nearest-neighbour by cosine = nearest-neighbour by L2 (when normalised).

**Use cases:**

| Task | How |
|---|---|
| Semantic search | Embed query + documents, rank by cosine |
| Clustering | K-Means on embeddings |
| Deduplication | Cosine threshold (e.g., > 0.95) |
| Cross-lingual retrieval | Multilingual SBERT models (mBERT / LaBSE) |

---

## 03 — Scaled Dot-Product Attention

### Motivation

In an RNN, position $t$ can only attend to position $t-1$ (limited receptive field, sequential computation bottleneck). Attention computes pairwise relevance between all positions simultaneously.

### Formula

Given queries $Q \in \mathbb{R}^{n \times d_k}$, keys $K \in \mathbb{R}^{m \times d_k}$, values $V \in \mathbb{R}^{m \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**Step-by-step:**

1. **Similarity scores:** $S = QK^\top \in \mathbb{R}^{n \times m}$ — raw dot products between each query and each key
2. **Scale:** $S / \sqrt{d_k}$ — without scaling, dot products grow with $d_k$, pushing softmax into saturation (near-zero gradients)
3. **Softmax:** row-wise → weights $A \in \mathbb{R}^{n \times m}$, each row sums to 1
4. **Weighted sum:** $AV \in \mathbb{R}^{n \times d_v}$ — each output is a convex combination of values

### Why $\sqrt{d_k}$ scaling?

If $q$ and $k$ have components drawn independently from $\mathcal{N}(0,1)$, then $q^\top k \sim \mathcal{N}(0, d_k)$ — variance grows linearly with $d_k$. Dividing by $\sqrt{d_k}$ restores unit variance, keeping softmax in its gradient-friendly regime.

### Computational Complexity

| Aspect | Value |
|---|---|
| Time | $O(n^2 d_k)$ — quadratic in sequence length |
| Space | $O(n^2)$ for attention matrix |
| Parallelism | Fully parallel (no sequential dependency) |

Quadratic scaling is the dominant bottleneck for long-context models (flash-attention, sparse attention, linear attention all aim to reduce this).

### Causal (Masked) Attention

For autoregressive generation, position $i$ must not attend to positions $j > i$. Apply additive mask:

$$S_{ij} = \begin{cases} q_i \cdot k_j / \sqrt{d_k} & j \le i \\ -\infty & j > i \end{cases}$$

$\exp(-\infty) = 0$ so future positions contribute zero to the weighted sum.

---

## 04 — Multi-Head Attention

### Motivation

A single attention head computes one set of relevance patterns. Different heads can specialise: one might capture syntactic dependencies, another long-range coreference, another local n-gram patterns.

### Formula

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V)$$

where:
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$ — output projection
- Typical: $d_k = d_v = d_{\text{model}} / h$, so total param count ≈ same as single head

### Parameter Count

For GPT-2 (small): $d_{\text{model}} = 768$, $h = 12$, $d_k = 64$

| Matrix | Shape | Parameters |
|---|---|---|
| $W^Q$ (per head) | $768 \times 64$ | 49,152 |
| $W^K$ (per head) | $768 \times 64$ | 49,152 |
| $W^V$ (per head) | $768 \times 64$ | 49,152 |
| All heads × 12 | — | 1,769,472 |
| $W^O$ | $768 \times 768$ | 589,824 |
| **Total per MHA block** | | **2,359,296** |

### Grouped Query Attention (GQA) / Multi-Query Attention (MQA)

Standard MHA: each head has its own $W^K$, $W^V$.
MQA: all heads share one $K$, $V$ pair — reduces KV cache size by factor $h$.
GQA: $g$ groups each sharing $K$, $V$ — interpolation between MHA and MQA (used in LLaMA 2/3, Mistral).

---

## 05 — Positional Encoding

Attention is permutation-equivariant: shuffling positions produces the same output (just shuffled). Positional encodings break this symmetry.

### Sinusoidal PE (Vaswani et al. 2017)

For position $\text{pos}$ and dimension $i$:

$$PE_{(\text{pos}, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i / d_{\text{model}}}}\right)$$

$$PE_{(\text{pos}, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i / d_{\text{model}}}}\right)$$

**Properties:**
- Even dimensions: sine; odd dimensions: cosine
- Low $i$ → high frequency (changes fast across positions); high $i$ → low frequency (changes slowly)
- Each frequency is a geometric progression: $\omega_i = 1/10000^{2i/d}$
- **Generalises to unseen lengths:** the model never sees position 5000 during training, but sinusoidal PE provides a valid vector for it
- **Linear relative positions:** $PE_{\text{pos}+k}$ can be expressed as a linear function of $PE_{\text{pos}}$ — enables relative position reasoning

**Why it works:** attention weights depend on $q^\top k = (x_i + PE_i)^\top (x_j + PE_j)$. The cross-terms $PE_i^\top PE_j$ depend only on relative offset $(i - j)$ for sinusoidal encodings.

### Learned PE

Simple: define an embedding table $E \in \mathbb{R}^{L_{\max} \times d_{\text{model}}}$, train it end-to-end.

| | Sinusoidal | Learned |
|---|---|---|
| Parameters | 0 | $L_{\max} \times d_{\text{model}}$ |
| OOV positions | Extrapolates (imperfectly) | Fails beyond $L_{\max}$ |
| Training signal | None needed | Gradient-based |
| Relative position info | Implicit in formula | Must be learned |
| Used in | Original Transformer | BERT, GPT-2 |

### Rotary PE (RoPE)

Applied in LLaMA, GPT-NeoX, Falcon. Instead of adding PE to embeddings before attention, rotate the query and key vectors:

$$\text{RoPE}(q, \text{pos}) = q \odot \cos(\theta_{\text{pos}}) + q_{\perp} \odot \sin(\theta_{\text{pos}})$$

where $q_\perp$ is obtained by rotating each 2D subspace by 90°. Key property: $q_{\text{pos}}^\top k_{\text{pos}'}$ depends only on $(q, k, \text{pos} - \text{pos}')$ — exactly encodes relative positions in the dot product.

---

## 06 — KV Cache

### Autoregressive Generation

Transformer decoder generates one token per forward pass. For token at position $t$:

1. Compute $Q_t$, $K_t$, $V_t$ from position $t$'s hidden state
2. Attention output: $\text{softmax}((Q_t K_{1:t}^\top) / \sqrt{d_k}) V_{1:t}$
3. Append predicted token, repeat

Without caching, each step recomputes $K_{1:t}$, $V_{1:t}$ — $O(t)$ redundant computation per step.

### KV Cache Mechanism

Store all past key-value pairs. At step $t$, cache contains $\{K_1, \ldots, K_{t-1}, V_1, \ldots, V_{t-1}\}$. Step $t$ computes only $K_t$, $V_t$, appends to cache, computes attention over the full cache.

**Complexity with cache:**

| | Without Cache | With Cache |
|---|---|---|
| FLOPS per step | $O(t \cdot d)$ | $O(d)$ (new K/V only) |
| Total FLOPS for $T$ tokens | $O(T^2 d)$ | $O(T d)$ |
| Memory | $O(d)$ runtime | $O(T \cdot L \cdot h \cdot d_k)$ persistent |

where $L$ = layers, $h$ = heads, $d_k$ = head dimension.

### Memory Footprint

For a model with $L$ layers, $h$ heads, head dim $d_k$, sequence length $T$, batch size $B$, in float16 (2 bytes):

$$\text{KV cache bytes} = 2 \times B \times T \times L \times h \times d_k \times 2$$

The factor of 2 accounts for both $K$ and $V$.

**GPT-3 (175B) example:**
- $L = 96$, $h = 96$, $d_k = 128$, $T = 2048$, $B = 1$
- KV cache ≈ $2 \times 1 \times 2048 \times 96 \times 96 \times 128 \times 2$ bytes ≈ **9.66 GB**

### Techniques to Reduce KV Cache

| Technique | How | Memory Reduction |
|---|---|---|
| MQA (Multi-Query Attention) | All heads share one K, V pair | $\times h$ reduction |
| GQA (Grouped-Query Attention) | $g$ groups share K, V | $\times (h/g)$ reduction |
| Sliding Window Attention | Only keep last $w$ positions in cache | $\times (w/T)$ reduction |
| KV Quantisation | Store K, V in int8/fp8 | $\times 2$ reduction |
| PagedAttention (vLLM) | Virtual memory paging for KV blocks | Eliminates fragmentation |

---

## 07 — Interview Questions

**Q1: Why does attention scale by $\sqrt{d_k}$?**
Dot products $q^\top k$ have variance $d_k$ when components are iid $\mathcal{N}(0,1)$. Scaling by $1/\sqrt{d_k}$ restores unit variance and prevents softmax saturation (near-zero gradients in backward pass).

**Q2: What is the difference between MHA, MQA, and GQA?**
MHA: each head has independent $W^Q, W^K, W^V$ projections. MQA: all heads share a single $K$, $V$ pair — reduces KV cache by factor $h$ but may hurt quality. GQA: intermediate — $g$ groups of heads share $K$, $V$ pairs. LLaMA 2/3 use GQA.

**Q3: Why does Word2Vec use negative sampling instead of full softmax?**
Full softmax normalises over the entire vocabulary ($V$ forward/backward passes per update). Negative sampling frames the problem as binary logistic regression for $K+1$ pairs — $O(K \cdot d)$ complexity regardless of $V$. Gradient still flows through the true pair and $K$ noisy negatives, but vocabulary embedding table is still fully updated over many steps.

**Q4: How does the KV cache trade memory for computation?**
Without cache, generating $T$ tokens requires $O(T^2 d)$ FLOPs (recomputes past K/V). With cache, $O(Td)$ FLOPs (only new K/V computed per step), at the cost of $O(T \cdot L \cdot h \cdot d_k)$ memory that grows linearly with sequence length. This is why long-context inference is memory-bound, not compute-bound.

**Q5: What are the limitations of sinusoidal positional encoding?**
(1) Extrapolation to lengths beyond training is imperfect — the model has never combined learned representations with PE at positions > $T_{\text{train}}$. (2) No learnable relative bias. (3) Fixed across all layers. RoPE addresses (1) and (2) by encoding relative position directly in the attention dot product.

**Q6: What is the "attention is all you need" key insight?**
Self-attention provides direct, $O(1)$-path connections between any two positions in a sequence — no information bottleneck (unlike LSTM hidden state), fully parallelisable (unlike RNNs), and scales to long sequences (with appropriate approximations). Feed-forward sublayers apply position-wise transformations to provide non-linear capacity.

---

## Resources

### Papers (Essential)
- **Attention Is All You Need** — Vaswani et al. (2017): `arxiv.org/abs/1706.03762`. The transformer paper. Read sections 3 and 4 alongside this module.
- **Efficient Estimation of Word Representations in Vector Space (Word2Vec)** — Mikolov et al. (2013): `arxiv.org/abs/1301.3781`
- **RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)** — Su et al. (2021): `arxiv.org/abs/2104.09864`
- **GQA: Training Generalized Multi-Query Transformer Models** — Ainslie et al. (2023): `arxiv.org/abs/2305.13245`

### Visual Explainers
- **Jay Alammar — "The Illustrated Word2Vec"** (`jalammar.github.io/illustrated-word2vec/`): Best visual walkthrough of the skip-gram objective and negative sampling.
- **Jay Alammar — "The Illustrated Transformer"** (`jalammar.github.io/illustrated-transformer/`): Step-by-step visual attention trace that matches the math in this module.
- **Jay Alammar — "The Illustrated GPT-2"** (`jalammar.github.io/illustrated-gpt2/`): Extends to causal masking and autoregressive decoding.

### Video
- **Andrej Karpathy — "Let's Build GPT from Scratch"** (YouTube): Implements scaled dot-product attention and MHA step by step. Best companion video to this module.
- **Yannic Kilcher — "Attention Is All You Need" paper explained** (YouTube): Deep paper reading with architectural context.

### Interactive
- **Tensor Playground / BerViz** (`github.com/jessevig/bertviz`): Visualize attention heads in pre-trained models. Builds intuition for what different heads learn.

---

*Next: [Module 07 — Transformers from Scratch](07-transformers.md)*
