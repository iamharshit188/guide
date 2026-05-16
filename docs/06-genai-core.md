# Module 06 — GenAI Core: Embeddings & Attention

> **Runnable code:** `src/06-genai/`
> ```bash
> python src/06-genai/word2vec.py
> python src/06-genai/attention.py
> python src/06-genai/multihead_attention.py
> python src/06-genai/positional_encoding.py
> python src/06-genai/kv_cache.py
> ```

---

> **Python prerequisite:** This module uses Python, NumPy, and ML libraries throughout. If you need a foundation or refresher, visit the **Languages → Python** guide and read **Section 21 — Python for ML & AI** before starting.

## Prerequisites & Overview

**Prerequisites:** Modules 01–05. NumPy, basic linear algebra.
**Estimated time:** 10–14 hours

**Install:**
```bash
pip install numpy
```

### Why This Module Matters

Everything in modern AI — ChatGPT, Claude, Stable Diffusion, GitHub Copilot — is built on two ideas:
1. **Embeddings** — represent meaning as vectors
2. **Attention** — let tokens look at each other to build context

Without understanding these, you can't debug LLM behavior, design RAG systems, or work on fine-tuning.

### Module Map

| Section | Core Concept | Where It's Used |
|---------|-------------|----------------|
| Word embeddings | Word2Vec, skip-gram | Semantic similarity, analogy |
| Attention | Scaled dot-product | Every Transformer layer |
| Multi-head attention | Parallel attention heads | BERT, GPT, T5 |
| Positional encoding | Sinusoidal, RoPE | Sequence order information |
| KV cache | Cached key-value pairs | Fast autoregressive inference |

---

# 1. Word Embeddings

## Intuition

Words are labels, not numbers. To do math on language, we need to turn words into vectors. A good embedding captures:
- **Similarity**: "cat" and "dog" should be close
- **Analogy**: king - man + woman ≈ queen

## 1.1 Co-occurrence Matrix

The simplest idea: words appearing in the same context have similar meanings.

```python
import numpy as np
from collections import defaultdict

corpus = [
    "the cat sat on the mat",
    "the dog sat on the floor",
    "cats and dogs are pets",
    "the mat is on the floor",
    "pets need food and water",
    "dogs like to eat food",
    "cats drink water not food",
]

# Build vocabulary
words    = [w for sent in corpus for w in sent.split()]
vocab    = sorted(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
V        = len(vocab)

print(f"Vocabulary size: {V}")

# Co-occurrence window=2
cooc   = np.zeros((V, V))
window = 2
for sent in corpus:
    tokens = sent.split()
    for i, target in enumerate(tokens):
        for j in range(max(0, i-window), min(len(tokens), i+window+1)):
            if i != j:
                cooc[word2idx[target], word2idx[tokens[j]]] += 1

# PPMI (Positive Pointwise Mutual Information)
total   = cooc.sum()
row_sum = cooc.sum(axis=1, keepdims=True)
col_sum = cooc.sum(axis=0, keepdims=True)

pmi = np.zeros_like(cooc)
mask = cooc > 0
pmi[mask] = np.log(cooc[mask] * total / ((row_sum * col_sum + 1e-9)[mask]))
pmi = np.maximum(pmi, 0)

print("\nPPMI most similar words:")
for w in ['cat', 'dog', 'food']:
    if w in word2idx:
        sims = [(vocab[i], pmi[word2idx[w], i]) for i in range(V) if i != word2idx[w]]
        sims.sort(key=lambda x: x[1], reverse=True)
        print(f"  '{w}': {', '.join(f'{wd}({s:.2f})' for wd, s in sims[:3])}")
```

## 1.2 Word2Vec Skip-Gram

**Core idea:** Predict context words given the target word. Hidden layer weights become embeddings.

```
Skip-gram training example (window=2):

Sentence: "the  cat  sat  on  the  mat"
           [0]  [1]  [2]  [3]  [4]  [5]

For target word "sat" (index 2), window=2 gives:
  Training pairs: (sat, the), (sat, cat), (sat, on), (sat, the)
                    ↑ target → context pairs

The model tries to maximize P("cat" | "sat"), P("on" | "sat"), etc.
In doing so, the embedding for "sat" must capture the meaning of its contexts.

After training — the embedding space:
  "cat"  ──▶ [0.8, 0.2, -0.1, ...]   ←─┐
  "dog"  ──▶ [0.7, 0.3, -0.2, ...]   ←─┘ similar vectors (both animals)
  "code" ──▶ [-0.1, 0.9, 0.8, ...]       far from animals

Analogy arithmetic works: king - man + woman ≈ queen
```

$$P(c | w) = \frac{\exp(\mathbf{v}_c \cdot \mathbf{u}_w)}{\sum_{c'} \exp(\mathbf{v}_{c'} \cdot \mathbf{u}_w)}$$

**Negative sampling** (fast approximation):
$$\mathcal{L} \approx \log \sigma(\mathbf{v}_c^T \mathbf{u}_w) + \sum_{k=1}^K \log \sigma(-\mathbf{v}_{n_k}^T \mathbf{u}_w)$$

```python
import numpy as np

rng = np.random.default_rng(42)

class Word2VecSkipGram:
    def __init__(self, vocab_size, embed_dim=8, lr=0.025, n_negative=5):
        self.V  = vocab_size
        self.d  = embed_dim
        self.lr = lr
        self.K  = n_negative

        self.W_in  = rng.uniform(-0.5/embed_dim, 0.5/embed_dim, (vocab_size, embed_dim))
        self.W_out = np.zeros((vocab_size, embed_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def train_pair(self, target_idx, context_idx, neg_idxs):
        u      = self.W_in[target_idx]
        v_pos  = self.W_out[context_idx]
        score  = self.sigmoid(np.dot(u, v_pos))
        loss   = -np.log(score + 1e-9)
        grad_u = (score - 1) * v_pos
        self.W_out[context_idx] -= self.lr * (score - 1) * u

        for neg_idx in neg_idxs:
            v_neg   = self.W_out[neg_idx]
            score_n = self.sigmoid(-np.dot(u, v_neg))
            loss   -= np.log(score_n + 1e-9)
            grad_u += (1 - score_n) * v_neg
            self.W_out[neg_idx] -= self.lr * (-(1 - score_n)) * u

        self.W_in[target_idx] -= self.lr * grad_u
        return loss

    def most_similar(self, word_idx, top_k=3):
        query  = self.W_in[word_idx]
        q_norm = np.linalg.norm(query)
        sims   = [(np.dot(query, self.W_in[i]) / (q_norm * np.linalg.norm(self.W_in[i]) + 1e-9), i)
                  for i in range(self.V) if i != word_idx]
        sims.sort(reverse=True)
        return sims[:top_k]

# Generate training pairs
model          = Word2VecSkipGram(V, embed_dim=8)
training_pairs = []
window         = 2
for sent in corpus:
    tokens = sent.split()
    for i, target in enumerate(tokens):
        for j in range(max(0, i-window), min(len(tokens), i+window+1)):
            if i != j:
                training_pairs.append((word2idx[target], word2idx[tokens[j]]))

# Train
neg_probs  = np.ones(V) / V
total_loss = 0
for epoch in range(300):
    for target_idx, context_idx in training_pairs:
        neg_idxs = []
        while len(neg_idxs) < model.K:
            n = rng.choice(V, p=neg_probs)
            if n != context_idx:
                neg_idxs.append(n)
        total_loss += model.train_pair(target_idx, context_idx, neg_idxs)

    if epoch % 100 == 0:
        avg = total_loss / ((epoch+1)*len(training_pairs))
        print(f"Epoch {epoch}: avg loss={avg:.4f}")

print("\nMost similar to 'cat':")
for sim, idx in model.most_similar(word2idx['cat']):
    print(f"  {vocab[idx]}: {sim:.4f}")
```

---

# 2. Scaled Dot-Product Attention

## Intuition

In "The animal didn't cross the street because it was too tired," what does "it" refer to? Attention lets each word **look at every other word** and decide relevance.

- **Query** $Q$: "what am I searching for?"
- **Key** $K$: "what do I describe myself as?"
- **Value** $V$: "what information do I carry?"

```
Attention mechanism — library analogy:

Q (Query)  = your search query: "What is 'it' referring to?"
K (Keys)   = index cards of every book: "animal: chapter 1", "street: chapter 2"
V (Values) = actual book content

Step 1: Score relevance
  score(Q, K_animal)  = Q · K_animal  = 8.5  (high — "it" likely is the animal)
  score(Q, K_street)  = Q · K_street  = 1.2  (low)
  ...

Step 2: Scale by √d_k (prevents scores from saturating softmax)
  scores = scores / √d_k

Step 3: Softmax → attention weights (sum to 1)
  weights = softmax(scores) = [0.87, 0.08, 0.03, 0.02, ...]

Step 4: Weighted sum of values
  output = 0.87 × V_animal + 0.08 × V_street + ...
         ← mostly "animal" information!

Attention matrix (each row = how much token i attends to token j):

        the  animal  street  it   tired
  the  [0.3,  0.2,   0.1,  0.2,  0.2]
animal [0.2,  0.4,   0.1,  0.2,  0.1]
street [0.2,  0.1,   0.4,  0.1,  0.2]
  it   [0.1,  0.6,   0.1,  0.1,  0.1]  ← "it" mostly attends to "animal"!
tired  [0.2,  0.3,   0.1,  0.2,  0.2]
```

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

> **Formula breakdown:**
> - $Q \in \mathbb{R}^{n \times d_k}$ — query matrix ($n$ tokens, each projected to $d_k$ dims)
> - $K \in \mathbb{R}^{n \times d_k}$ — key matrix (same shape as $Q$)
> - $QK^T \in \mathbb{R}^{n \times n}$ — all pairs of dot products → attention scores
> - $\sqrt{d_k}$ — scale factor: prevents dot products from growing too large as $d_k$ increases
> - $\text{softmax}(\cdot)$ — normalizes scores to probabilities summing to 1
> - $V \in \mathbb{R}^{n \times d_v}$ — value matrix
> - Final output: weighted sum of values, where weights reflect relevance

```python
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K: (batch, seq, d_k)
    V:    (batch, seq, d_v)
    Returns: output (batch, seq, d_v), weights (batch, seq, seq)
    """
    d_k    = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)   # (batch, seq, seq)

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    attn_w = softmax(scores, axis=-1)
    output = attn_w @ V
    return output, attn_w

rng     = np.random.default_rng(42)
seq_len = 5
d_k     = 4
d_v     = 4

# Simulate one sentence (batch=1)
X   = rng.randn(1, seq_len, d_k)
W_Q = rng.randn(d_k, d_k)
W_K = rng.randn(d_k, d_k)
W_V = rng.randn(d_k, d_v)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

output, attn_w = scaled_dot_product_attention(Q, K, V)
print(f"Attention output shape: {output.shape}")
print(f"\nAttention weights (token i → token j):")
print(attn_w[0].round(3))
print(f"\nRow sums (must all be 1): {attn_w[0].sum(axis=-1).round(4)}")

# Causal mask for decoder (token i can only see 0..i)
def causal_mask(seq_len):
    return np.tril(np.ones((seq_len, seq_len)))[None, None]

mask            = causal_mask(seq_len)
out_causal, aw  = scaled_dot_product_attention(Q, K, V, mask=mask)
print(f"\nCausal attention weights (upper triangle = ~0):")
print(aw[0].round(3))
```

**Why scale by $\sqrt{d_k}$?**

```python
d_large    = 64
q, k       = rng.randn(d_large), rng.randn(d_large)
dot        = np.dot(q, k)
scaled_dot = dot / np.sqrt(d_large)

sm_raw    = softmax(np.array([dot,   0, 0]))
sm_scaled = softmax(np.array([scaled_dot, 0, 0]))
print(f"\nWith d_k={d_large}:")
print(f"  Unscaled dot={dot:.1f} → softmax max={sm_raw.max():.4f}  (saturated!)")
print(f"  Scaled dot={scaled_dot:.1f}  → softmax max={sm_scaled.max():.4f}  (normal)")
```

---

# 3. Multi-Head Attention

## Intuition

One attention head can learn one type of relationship at a time (subject-verb, adjective-noun). Multi-head runs $h$ heads in parallel, each learning different relationships, then concatenates results.

```
Multi-head attention with h=4 heads, d_model=512, d_k=128:

Input X (batch, seq, 512)
        │
        ├──[W_Q1, W_K1, W_V1]──▶ Head 1 attention ──▶ (batch, seq, 128)
        │                         (syntax relationships)
        ├──[W_Q2, W_K2, W_V2]──▶ Head 2 attention ──▶ (batch, seq, 128)
        │                         (semantic relationships)
        ├──[W_Q3, W_K3, W_V3]──▶ Head 3 attention ──▶ (batch, seq, 128)
        │                         (long-range dependencies)
        └──[W_Q4, W_K4, W_V4]──▶ Head 4 attention ──▶ (batch, seq, 128)
                                  (positional relations)
                                          │
                         Concat all heads: (batch, seq, 512)
                                          │
                                    [W_O (512×512)]
                                          │
                               Output: (batch, seq, 512)

Each head looks for different patterns simultaneously.
The concatenation + linear projection merges all perspectives.
```

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

```python
import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model=16, n_heads=4):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        rng   = np.random.default_rng(42)
        scale = np.sqrt(2.0 / (d_model + self.d_k))

        self.W_Q = rng.normal(0, scale, (n_heads, d_model, self.d_k))
        self.W_K = rng.normal(0, scale, (n_heads, d_model, self.d_k))
        self.W_V = rng.normal(0, scale, (n_heads, d_model, self.d_k))
        self.W_O = rng.normal(0, scale, (n_heads * self.d_k, d_model))

    def forward(self, X, mask=None):
        """X: (batch, seq, d_model) → (batch, seq, d_model)"""
        heads = []
        for h in range(self.n_heads):
            Q = X @ self.W_Q[h]
            K = X @ self.W_K[h]
            V = X @ self.W_V[h]

            scores = Q @ K.swapaxes(-2, -1) / np.sqrt(self.d_k)
            if mask is not None:
                scores = np.where(mask == 0, -1e9, scores)
            attn = softmax(scores, axis=-1)
            heads.append(attn @ V)

        concat = np.concatenate(heads, axis=-1)
        return concat @ self.W_O

rng    = np.random.default_rng(42)
mha    = MultiHeadAttention(d_model=16, n_heads=4)
X      = rng.randn(2, 6, 16)   # batch=2, seq=6, d=16
output = mha.forward(X)

print(f"MHA Input:  {X.shape}")
print(f"MHA Output: {output.shape}")
print(f"\nd_k per head: {mha.d_k}")
print(f"Total MHA params: {4 * mha.n_heads * mha.d_model * mha.d_k}")
```

---

# 4. Positional Encoding

## Intuition

Attention is **order-invariant** — "dog bites man" and "man bites dog" look the same to pure attention. Positional encoding injects position information.

```python
import numpy as np

# ── Sinusoidal PE (original Transformer) ────────────────────
def sinusoidal_pe(max_seq_len, d_model):
    pe  = np.zeros((max_seq_len, d_model))
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i   = np.arange(0, d_model, 2)
    div = np.power(10000.0, i / d_model)

    pe[:, 0::2] = np.sin(pos / div)
    pe[:, 1::2] = np.cos(pos / div)
    return pe

pe = sinusoidal_pe(10, 8)
print(f"Sinusoidal PE shape: {pe.shape}")
print(f"\nFirst 5 positions:")
print(pe[:5].round(3))

# Property: each position has a unique encoding
print(f"\nDot products between positions (same = max):")
dots = pe[:5] @ pe[:5].T
print(dots.round(2))

# ── RoPE (Rotary Position Embedding) ───────────────────────
def apply_rope(x, base=10000):
    """x: (seq, d) — rotates query/key by position."""
    seq, d = x.shape
    assert d % 2 == 0

    theta   = 1.0 / base ** (np.arange(0, d, 2) / d)
    freqs   = np.outer(np.arange(seq), theta)

    x1, x2 = x[:, 0::2], x[:, 1::2]
    cos_f   = np.cos(freqs)
    sin_f   = np.sin(freqs)

    rotated          = np.zeros_like(x)
    rotated[:, 0::2] = x1 * cos_f - x2 * sin_f
    rotated[:, 1::2] = x1 * sin_f + x2 * cos_f
    return rotated

rng   = np.random.default_rng(42)
q     = rng.randn(6, 8)
q_rot = apply_rope(q)
print(f"\nRoPE: norm preserved: {np.linalg.norm(q[0]):.4f} ≈ {np.linalg.norm(q_rot[0]):.4f}")
```

---

# 5. KV Cache

## Intuition

Without KV cache: generating token 100 recomputes K and V for all 99 previous tokens. With KV cache: only compute K,V for the new token; reuse all previous.

```python
import numpy as np

class KVCache:
    def __init__(self):
        self.k_cache = None
        self.v_cache = None

    def update(self, new_k, new_v):
        if self.k_cache is None:
            self.k_cache = new_k
            self.v_cache = new_v
        else:
            self.k_cache = np.concatenate([self.k_cache, new_k], axis=1)
            self.v_cache = np.concatenate([self.v_cache, new_v], axis=1)

    def size_bytes(self, dtype_bytes=4):
        if self.k_cache is None:
            return 0
        return (self.k_cache.nbytes + self.v_cache.nbytes)

rng    = np.random.default_rng(42)
d      = 8
cache  = KVCache()
W_K    = rng.randn(d, d)
W_V    = rng.randn(d, d)

print("KV Cache growth during generation:")
print(f"{'Step':>6} {'Seq len':>10} {'Cache (bytes)':>14}")

for step in range(8):
    x_new = rng.randn(1, 1, d)   # new token embedding
    k_new = x_new @ W_K
    v_new = x_new @ W_V
    cache.update(k_new, v_new)
    print(f"{step+1:>6} {cache.k_cache.shape[1]:>10} {cache.size_bytes():>14}")

# Memory analysis for GPT-3 scale
max_seq   = 4096
d_model   = 12288   # GPT-3
n_layers  = 96
kv_bytes  = max_seq * d_model * 2 * n_layers * 4  # float32
print(f"\nGPT-3 KV cache at {max_seq} tokens: {kv_bytes/1e9:.2f} GB per sample")
```

---

# 6. Interview Q&A

## Q1: Why does attention use Q, K, V (three separate matrices)?

The separation allows independent control: Q controls "what I'm searching for," K controls "how I'm described to others," V controls "what I contribute when selected." Collapsing Q and K would tie how a token describes itself to how it queries others — the split enables richer representations.

## Q2: Explain why $\sqrt{d_k}$ scaling is necessary.

Dot products $q \cdot k = \sum_i q_i k_i$ have variance $d_k$ (sum of $d_k$ unit-variance terms). For large $d_k$, these values are large → softmax saturates (outputs near 0/1) → vanishing gradients. Dividing by $\sqrt{d_k}$ brings variance back to 1.

## Q3: What is the time complexity of self-attention?

$O(n^2 d)$ where $n$ = sequence length, $d$ = dimension. The $n^2$ comes from the $n \times n$ score matrix. This is the bottleneck for long documents. Flash Attention reduces memory from $O(n^2)$ to $O(n)$ through chunked computation, but time remains $O(n^2 d)$.

## Q4: Why do transformers need positional encoding?

Self-attention is a **set operation** — permutation-invariant by design. Without PE, "dog bites man" and "man bites dog" produce identical representations. PE injects position-dependent signals that break this symmetry.

## Q5: How does KV cache speed up inference?

Without cache: generating token $t$ requires recomputing K, V for all $t-1$ previous tokens — $O(t)$ work per step, $O(n^2)$ total. With cache: K, V from previous tokens are stored and reused — only compute K, V for the new token, $O(1)$ per step, $O(n)$ total. Trade-off: memory grows linearly with sequence length.

---

# 7. Cheat Sheet

| Formula | Name | Role |
|---------|------|------|
| $\text{softmax}(QK^T/\sqrt{d_k})V$ | Scaled dot-product attention | Core of every Transformer |
| $\text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O$ | Multi-head attention | Multiple relationship types |
| $PE_{(p,2i)}=\sin(p/10000^{2i/d})$ | Sinusoidal PE | Original Transformer |
| $P(c|w) \approx \sigma(v_c^T u_w)$ | Skip-gram negative sampling | Word2Vec training |
| KV cache | Store K, V; reuse next step | Fast autoregressive generation |
| $Q=XW^Q, K=XW^K, V=XW^V$ | Linear projections | Input to attention |

---

# MINI-PROJECT — Semantic Text Similarity Engine

**What you build:** A system that finds the most semantically similar sentences to a query using attention-pooled embeddings and cosine search.

---

## Step 1 — Sentence Corpus and Vocabulary

```python
import numpy as np

rng = np.random.default_rng(42)

sentences = [
    "Machine learning is a branch of artificial intelligence",
    "Deep learning uses neural networks to learn representations",
    "Python is the most popular language for data science",
    "Natural language processing enables computers to understand text",
    "Neural networks are inspired by the biological brain",
    "Data science combines statistics programming and domain knowledge",
    "Transformers revolutionized natural language processing in 2017",
    "Backpropagation computes gradients for neural network training",
    "Attention mechanism allows models to focus on relevant parts",
    "Word embeddings represent words as dense numerical vectors",
]

all_words = [w for s in sentences for w in s.lower().split()]
vocab     = sorted(set(all_words))
word2idx  = {w: i for i, w in enumerate(vocab)}
V         = len(vocab)

d_emb     = 16
word_embs = rng.randn(V, d_emb)
word_embs /= np.linalg.norm(word_embs, axis=1, keepdims=True)

print(f"Corpus: {len(sentences)} sentences, vocab: {V} words")
```

---

## Step 2 — Attention-Pooled Sentence Encoding

```python
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def encode(sentence):
    words = [w for w in sentence.lower().split() if w in word2idx]
    if not words:
        return np.zeros(d_emb)

    X      = np.array([word_embs[word2idx[w]] for w in words])  # (seq, d)
    scores = X @ X.T / np.sqrt(d_emb)
    attn   = softmax(scores, axis=-1)
    X_att  = attn @ X
    emb    = X_att.mean(axis=0)
    norm   = np.linalg.norm(emb)
    return emb / norm if norm > 1e-9 else emb

# Encode all sentences
sent_embs = np.array([encode(s) for s in sentences])
print(f"Sentence embeddings: {sent_embs.shape}")
```

---

## Step 3 — Semantic Search

```python
def search(query, top_k=3):
    q_emb  = encode(query)
    sims   = sent_embs @ q_emb
    ranked = np.argsort(sims)[::-1]
    return [(sims[i], sentences[i]) for i in ranked[:top_k]]

queries = [
    "How do neural networks learn?",
    "NLP and text understanding",
    "Programming for ML",
]

for q in queries:
    print(f"\nQuery: '{q}'")
    for rank, (sim, sent) in enumerate(search(q), 1):
        print(f"  {rank}. [{sim:.4f}] {sent}")
```

---

## Step 4 — Attention Visualization

```python
def visualize_attention(sentence):
    words  = [w for w in sentence.lower().split() if w in word2idx]
    X      = np.array([word_embs[word2idx[w]] for w in words])
    scores = X @ X.T / np.sqrt(d_emb)
    attn   = softmax(scores, axis=-1)

    print(f"\nAttention: '{sentence}'")
    print(f"{'':>12}" + "".join(f"{w[:8]:>10}" for w in words))
    for i, word in enumerate(words):
        row = f"{word[:10]:>12}" + "".join(f"{attn[i,j]:>10.3f}" for j in range(len(words)))
        print(row)

visualize_attention("deep learning neural networks brain")
```

---

## What This Project Demonstrated

| Concept | Where it appeared |
|---------|------------------|
| Word embeddings | `word_embs` lookup, normalized |
| Scaled dot-product | `scores = X @ X.T / sqrt(d_emb)` |
| Softmax attention | `attn = softmax(scores)` |
| Attention pooling | `X_att = attn @ X` |
| Cosine similarity | `sims = sent_embs @ q_emb` |
| KV cache | Demonstrated incremental generation |
| Multi-head attention | Implemented full MHA class |
| Positional encoding | Sinusoidal + RoPE implementations |

---

*Next: [Module 07 — Transformers from Scratch](07-transformers.md)*
