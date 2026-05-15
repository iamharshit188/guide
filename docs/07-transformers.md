# Module 07 — Transformers from Scratch

> **Runnable code:** `src/07-transformer/`
> ```bash
> python src/07-transformer/tokenizer.py
> python src/07-transformer/model_numpy.py
> python src/07-transformer/train.py
> ```

---

## Prerequisites & Overview

**Prerequisites:** Module 06 (attention, embeddings, PE). NumPy.
**Estimated time:** 12–16 hours

### Why This Module Matters

The Transformer is the foundation of BERT, GPT, T5, LLaMA, Claude, and essentially every state-of-the-art model. Building it from scratch is the single highest-return learning exercise in ML. After this module you can read any model paper and understand exactly what's going on.

### Module Map

| Section | Concept | Real-World Impact |
|---------|---------|-----------------|
| BPE Tokenization | Subword tokenization | How text becomes tokens |
| Encoder block | MHA + FFN + LayerNorm | BERT, encoder models |
| Decoder block | Masked MHA + cross-attention | GPT, decoder models |
| Full Transformer | Encoder-decoder | T5, translation |
| Training | LR warmup, label smoothing | Stable LLM training |

### Full Transformer Architecture

```
"Hello, how are you?" ─▶ [Tokenizer] ─▶ [1, 234, 15, 78, 90]
                                                   │
                                          [Token Embeddings]
                                          [+ Positional Encoding]
                                                   │
                              ┌────────────────────┴────────────────────┐
                              │           ENCODER (×N layers)           │
                              │  ┌──────────────────────────────────┐   │
                              │  │  Multi-Head Self-Attention        │   │
                              │  │  + Residual + LayerNorm           │   │
                              │  ├──────────────────────────────────┤   │
                              │  │  Feed-Forward (4× hidden)         │   │
                              │  │  + Residual + LayerNorm           │   │
                              │  └──────────────────────────────────┘   │
                              └─────────────────┬───────────────────────┘
                                                │ encoder output (context)
                              ┌─────────────────▼───────────────────────┐
                              │           DECODER (×N layers)           │
                              │  ┌──────────────────────────────────┐   │
                              │  │  Masked Multi-Head Self-Attention │   │
                              │  │  (can't see future tokens)        │   │
                              │  ├──────────────────────────────────┤   │
                              │  │  Cross-Attention                  │   │
                              │  │  (Q from decoder, K/V from encoder│   │
                              │  ├──────────────────────────────────┤   │
                              │  │  Feed-Forward + Residual + LN     │   │
                              │  └──────────────────────────────────┘   │
                              └─────────────────┬───────────────────────┘
                                                │
                                        [Linear + Softmax]
                                                │
                                   "Bonjour, comment allez-vous?"

BERT uses only Encoder. GPT uses only Decoder. T5 uses both.
```

---

# 1. BPE Tokenization

## Intuition

Split text into subword units. "tokenization" → ["token", "ization"]. Handles:
- Unknown words: "ChatGPT" → ["Chat", "G", "PT"]
- Different languages with shared subwords
- Morphology: "running" → ["run", "##ning"]

```
Why not just split by words?

Word-level:  "unhappiness" → ["unhappiness"]
             Problem: what if "unhappiness" never appeared in training?
             → "unknown token" (OOV problem)

Char-level:  "unhappiness" → ["u","n","h","a","p","p","i","n","e","s","s"]
             Problem: sequences become very long, lose word structure

BPE:         "unhappiness" → ["un", "happy", "ness"]
             ✓ handles new words   ✓ reuses known subwords   ✓ reasonable length

BPE merge steps (training on a tiny corpus):
  Iteration 1: count pairs → ('t','h') = 200 times → merge to "th"
  Iteration 2: count pairs → ('th','e') = 150 times → merge to "the"
  Iteration 3: count pairs → ('i','n') = 140 times → merge to "in"
  ...
  After 50,000 merges: vocabulary of ~50,000 subwords (GPT-2/3 vocab size)
```

## Algorithm

1. Start with character-level vocabulary
2. Count all adjacent symbol pairs
3. Merge the most frequent pair → new token
4. Repeat until vocabulary size reached

```python
import re
from collections import Counter, defaultdict

class BPETokenizer:
    """Byte Pair Encoding tokenizer from scratch."""

    def __init__(self, vocab_size=100):
        self.vocab_size  = vocab_size
        self.vocab       = set()
        self.merges      = []   # list of (pair → merged) rules
        self.word_freqs  = {}

    def fit(self, texts):
        # Step 1: Split text into words, track frequency
        word_counts = Counter()
        for text in texts:
            tokens = text.lower().split()
            word_counts.update(tokens)

        # Step 2: Initialize: each word is chars + end-of-word marker
        self.word_freqs = {' '.join(list(w)) + ' </w>': freq
                           for w, freq in word_counts.items()}

        # Initial vocab: all characters
        chars = set()
        for word in self.word_freqs:
            chars.update(word.split())
        self.vocab = chars | {'</w>'}

        print(f"Initial vocab size: {len(self.vocab)}")

        # Step 3: Merge until target vocab size
        while len(self.vocab) < self.vocab_size:
            pairs = self._count_pairs()
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self._merge_pair(best_pair)
            self.merges.append(best_pair)
            self.vocab.add(''.join(best_pair))

        print(f"Final vocab size: {len(self.vocab)}")
        return self

    def _count_pairs(self):
        """Count frequency of each adjacent pair."""
        pairs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def _merge_pair(self, pair):
        """Apply one merge rule to all words."""
        new_freqs = {}
        bigram    = ' '.join(pair)
        merged    = ''.join(pair)

        for word, freq in self.word_freqs.items():
            new_word = word.replace(bigram, merged)
            new_freqs[new_word] = freq

        self.word_freqs = new_freqs

    def tokenize(self, text):
        """Tokenize text using learned merges."""
        tokens = []
        for word in text.lower().split():
            word_chars = list(word) + ['</w>']
            # Apply merges in order
            for pair in self.merges:
                i = 0
                while i < len(word_chars) - 1:
                    if (word_chars[i], word_chars[i+1]) == pair:
                        word_chars = word_chars[:i] + [''.join(pair)] + word_chars[i+2:]
                    else:
                        i += 1
            tokens.extend(word_chars)
        return tokens

# Train
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning models learn patterns from data",
    "transformers use attention mechanisms for sequence modeling",
    "neural networks are universal function approximators",
    "gradient descent optimizes model parameters",
]

bpe = BPETokenizer(vocab_size=50)
bpe.fit(corpus)

test_sentences = [
    "the machine learning model",
    "transformers are universal",
]

for sent in test_sentences:
    tokens = bpe.tokenize(sent)
    print(f"\nTokenize: '{sent}'")
    print(f"  → {tokens}")
    print(f"  Length: {len(tokens)} tokens")

print(f"\nTop 10 merges learned:")
for i, (a, b) in enumerate(bpe.merges[:10]):
    print(f"  {i+1:2d}. '{a}' + '{b}' → '{a+b}'")
```

---

# 2. Transformer Building Blocks

## 2.1 Layer Normalization

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

Unlike BatchNorm (normalizes across batch), LayerNorm normalizes across **features** within one sample — works at any batch size.

```python
import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps   = eps
        self.gamma = np.ones(d_model)   # scale (learned)
        self.beta  = np.zeros(d_model)  # shift (learned)

    def forward(self, x):
        # x: (batch, seq, d_model)
        mu    = x.mean(axis=-1, keepdims=True)    # per-token mean
        sigma = x.std(axis=-1, keepdims=True)      # per-token std
        x_hat = (x - mu) / (sigma + self.eps)
        return self.gamma * x_hat + self.beta

rng = np.random.default_rng(42)
ln  = LayerNorm(d_model=8)
x   = rng.randn(2, 6, 8) * 10 + 5   # shifted, scaled input

print(f"Before LayerNorm: mean={x.mean(axis=-1)[0,:3].round(2)}, std={x.std(axis=-1)[0,:3].round(2)}")
x_out = ln.forward(x)
print(f"After  LayerNorm: mean={x_out.mean(axis=-1)[0,:3].round(6)}, std={x_out.std(axis=-1)[0,:3].round(4)}")
```

## 2.2 Feed-Forward Network (FFN)

After attention, each position passes through a 2-layer MLP independently:

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

The hidden dimension is typically $4 \times d_{model}$.

```python
class FeedForward:
    def __init__(self, d_model, d_ff=None):
        d_ff  = d_ff or 4 * d_model   # 4× is the transformer default
        rng   = np.random.default_rng(42)
        scale = np.sqrt(2.0 / (d_model + d_ff))

        self.W1 = rng.normal(0, scale, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.normal(0, scale, (d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        # x: (batch, seq, d_model)
        h = np.maximum(0, x @ self.W1 + self.b1)   # ReLU
        return h @ self.W2 + self.b2

rng = np.random.default_rng(42)
ffn = FeedForward(d_model=16)
x   = rng.randn(2, 6, 16)
y   = ffn.forward(x)
print(f"FFN: {x.shape} → {y.shape}")
print(f"Parameters: W1={ffn.W1.shape}, W2={ffn.W2.shape}")
```

## 2.3 Encoder Block

$$\text{EncoderBlock}(x) = \text{LayerNorm}(x + \text{MHA}(x)) \quad \text{then} \quad \text{LayerNorm}(\cdot + \text{FFN}(\cdot))$$

The **residual connection** ($x + \text{sublayer}(x)$) is critical: it allows gradients to flow directly from output to input, enabling training of very deep networks.

```
Encoder block — detailed data flow:

    x (input, e.g. shape [batch, seq, 512])
    │
    ├─────────────────────────────────┐  ← skip connection (residual)
    │                                 │
    ▼                                 │
  Multi-Head Self-Attention           │
  (all tokens attend to each other)   │
    │                                 │
    └───────────── + ─────────────────┘
                   │
              LayerNorm
                   │
    ┌──────────────┴──────────────────┐  ← another skip connection
    │                                 │
    ▼                                 │
  Feed-Forward Network                │
  Linear(512→2048) → ReLU             │
  Linear(2048→512)                    │
    │                                 │
    └───────────── + ─────────────────┘
                   │
              LayerNorm
                   │
    x' (output, same shape as input)

Why residual connections?
  Without: gradient of layer 50 must pass through 50 multiplications → vanishes to 0
  With:    gradient flows directly through skip path → no vanishing!
```

```python
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

class EncoderBlock:
    """One Transformer encoder layer."""

    def __init__(self, d_model=16, n_heads=4, d_ff=64):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        rng   = np.random.default_rng(42)
        scale = np.sqrt(2.0 / (d_model + self.d_k))

        self.W_Q = rng.normal(0, scale, (n_heads, d_model, self.d_k))
        self.W_K = rng.normal(0, scale, (n_heads, d_model, self.d_k))
        self.W_V = rng.normal(0, scale, (n_heads, d_model, self.d_k))
        self.W_O = rng.normal(0, scale, (n_heads * self.d_k, d_model))

        self.ffn  = FeedForward(d_model, d_ff)
        self.ln1  = LayerNorm(d_model)
        self.ln2  = LayerNorm(d_model)

    def mha(self, X):
        heads = []
        for h in range(self.n_heads):
            Q = X @ self.W_Q[h]
            K = X @ self.W_K[h]
            V = X @ self.W_V[h]
            s = Q @ K.swapaxes(-2, -1) / np.sqrt(self.d_k)
            heads.append(softmax(s, axis=-1) @ V)
        return np.concatenate(heads, axis=-1) @ self.W_O

    def forward(self, x):
        # Sublayer 1: MHA + residual + LN
        x = self.ln1.forward(x + self.mha(x))

        # Sublayer 2: FFN + residual + LN
        x = self.ln2.forward(x + self.ffn.forward(x))
        return x

rng    = np.random.default_rng(42)
enc    = EncoderBlock(d_model=16, n_heads=4, d_ff=64)
x      = rng.randn(2, 6, 16)
output = enc.forward(x)
print(f"Encoder block: {x.shape} → {output.shape}")
```

---

# 3. Decoder Block

The decoder has **three sublayers**:
1. Masked MHA (attend to previously generated tokens only)
2. Cross-attention (attend to encoder output)
3. FFN

```python
class DecoderBlock:
    """One Transformer decoder layer."""

    def __init__(self, d_model=16, n_heads=4, d_ff=64):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        rng   = np.random.default_rng(42)
        scale = np.sqrt(2.0 / (d_model + self.d_k))

        # Self-attention (masked)
        self.W_Q_self = rng.normal(0, scale, (n_heads, d_model, self.d_k))
        self.W_K_self = rng.normal(0, scale, (n_heads, d_model, self.d_k))
        self.W_V_self = rng.normal(0, scale, (n_heads, d_model, self.d_k))
        self.W_O_self = rng.normal(0, scale, (n_heads * self.d_k, d_model))

        # Cross-attention (encoder output as K, V)
        self.W_Q_cross = rng.normal(0, scale, (n_heads, d_model, self.d_k))
        self.W_K_cross = rng.normal(0, scale, (n_heads, d_model, self.d_k))
        self.W_V_cross = rng.normal(0, scale, (n_heads, d_model, self.d_k))
        self.W_O_cross = rng.normal(0, scale, (n_heads * self.d_k, d_model))

        self.ffn  = FeedForward(d_model, d_ff)
        self.ln1  = LayerNorm(d_model)
        self.ln2  = LayerNorm(d_model)
        self.ln3  = LayerNorm(d_model)

    def _mha(self, X, W_Q, W_K, W_V, W_O, KV_src=None, mask=None):
        """General MHA. KV_src: encoder output for cross-attention."""
        src    = KV_src if KV_src is not None else X
        heads  = []
        for h in range(self.n_heads):
            Q = X   @ W_Q[h]
            K = src @ W_K[h]
            V = src @ W_V[h]
            s = Q @ K.swapaxes(-2, -1) / np.sqrt(self.d_k)
            if mask is not None:
                s = np.where(mask == 0, -1e9, s)
            heads.append(softmax(s, axis=-1) @ V)
        return np.concatenate(heads, axis=-1) @ W_O

    def forward(self, x, encoder_output):
        seq = x.shape[1]

        # Causal mask: decoder only sees past tokens
        mask = np.tril(np.ones((seq, seq)))[None, None]

        # Sublayer 1: masked self-attention
        x = self.ln1.forward(x + self._mha(x,
                                             self.W_Q_self, self.W_K_self,
                                             self.W_V_self, self.W_O_self,
                                             mask=mask))

        # Sublayer 2: cross-attention (Q from decoder, K/V from encoder)
        x = self.ln2.forward(x + self._mha(x,
                                             self.W_Q_cross, self.W_K_cross,
                                             self.W_V_cross, self.W_O_cross,
                                             KV_src=encoder_output))

        # Sublayer 3: FFN
        x = self.ln3.forward(x + self.ffn.forward(x))
        return x

rng     = np.random.default_rng(42)
dec     = DecoderBlock(d_model=16, n_heads=4)
enc_out = rng.randn(2, 8, 16)   # encoder output (batch=2, enc_seq=8, d=16)
dec_in  = rng.randn(2, 5, 16)   # decoder input (batch=2, dec_seq=5, d=16)
dec_out = dec.forward(dec_in, enc_out)
print(f"Decoder block: {dec_in.shape} → {dec_out.shape}")
```

---

# 4. Full Transformer

```python
class Transformer:
    """
    Full encoder-decoder Transformer.
    Architecture: Vaswani et al., "Attention is All You Need" (2017)
    """

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=32, n_heads=4, d_ff=128,
                 n_encoder_layers=2, n_decoder_layers=2,
                 max_seq_len=64):

        rng = np.random.default_rng(42)

        # Token embeddings
        self.src_emb = rng.normal(0, 0.01, (src_vocab_size, d_model))
        self.tgt_emb = rng.normal(0, 0.01, (tgt_vocab_size, d_model))

        # Positional encoding
        pe  = np.zeros((max_seq_len, d_model))
        pos = np.arange(max_seq_len)[:, None]
        i   = np.arange(0, d_model, 2)
        div = np.power(10000.0, i / d_model)
        pe[:, 0::2] = np.sin(pos / div)
        pe[:, 1::2] = np.cos(pos / div)
        self.pe = pe

        # Encoder and decoder stacks
        self.encoders = [EncoderBlock(d_model, n_heads, d_ff)
                         for _ in range(n_encoder_layers)]
        self.decoders = [DecoderBlock(d_model, n_heads, d_ff)
                         for _ in range(n_decoder_layers)]

        # Output projection
        self.W_out = rng.normal(0, 0.01, (d_model, tgt_vocab_size))

    def encode(self, src_ids):
        """src_ids: (batch, src_seq) of integer token ids"""
        x   = self.src_emb[src_ids]                      # (batch, src_seq, d)
        x  += self.pe[:src_ids.shape[1]][None]            # add PE
        for enc in self.encoders:
            x = enc.forward(x)
        return x

    def decode(self, tgt_ids, encoder_output):
        """tgt_ids: (batch, tgt_seq)"""
        x   = self.tgt_emb[tgt_ids]
        x  += self.pe[:tgt_ids.shape[1]][None]
        for dec in self.decoders:
            x = dec.forward(x, encoder_output)
        return x

    def forward(self, src_ids, tgt_ids):
        enc_out = self.encode(src_ids)
        dec_out = self.decode(tgt_ids, enc_out)
        logits  = dec_out @ self.W_out                    # (batch, tgt_seq, tgt_vocab)
        return logits

# Demo
SRC_VOCAB = 50
TGT_VOCAB = 50

model   = Transformer(SRC_VOCAB, TGT_VOCAB)
src_ids = np.array([[3, 7, 12, 5, 0]])      # (1, src_seq=5)
tgt_ids = np.array([[2, 8, 14]])            # (1, tgt_seq=3)

logits  = model.forward(src_ids, tgt_ids)
print(f"Transformer forward pass:")
print(f"  src: {src_ids.shape} → encoder")
print(f"  tgt: {tgt_ids.shape} → decoder")
print(f"  logits: {logits.shape}")  # (1, 3, 50)

# Next-token prediction
next_token_logits = logits[0, -1]   # last position
next_token_probs  = softmax(next_token_logits)
predicted_token   = next_token_probs.argmax()
print(f"\nNext token predicted: {predicted_token} (confidence: {next_token_probs.max():.4f})")
```

---

# 5. Parameter Count Analysis

```python
def count_params(d_model, n_heads, d_ff, n_layers, vocab_size):
    """Count Transformer parameters."""
    d_k    = d_model // n_heads

    # Per attention layer: Q, K, V, O projections
    attn_params = 4 * n_heads * d_model * d_k  # ≈ 4 * d_model^2

    # Per FFN: W1, W2
    ffn_params = d_model * d_ff + d_ff * d_model  # 2 * d_model * d_ff = 8 * d_model^2

    # LayerNorm: gamma + beta per layer = 2 * d_model per LN block
    ln_params = 2 * 2 * d_model   # 2 LN per encoder/decoder block

    per_layer = attn_params + ffn_params + ln_params

    total_enc = n_layers * per_layer
    total_dec = n_layers * (2 * attn_params + ffn_params + 3 * 2 * d_model)

    # Embeddings (src + tgt)
    embed_params = 2 * vocab_size * d_model

    total = embed_params + total_enc + total_dec
    return {
        "attention_per_layer": attn_params,
        "ffn_per_layer":       ffn_params,
        "encoder_total":       total_enc,
        "decoder_total":       total_dec,
        "embeddings":          embed_params,
        "TOTAL":               total,
    }

print("\nParameter counts for known models:")
configs = [
    ("GPT-2 Small",  768,  12, 3072, 12, 50257),
    ("GPT-2 Medium", 1024, 16, 4096, 24, 50257),
    ("BERT-Base",    768,  12, 3072, 12, 30522),
    ("BERT-Large",   1024, 16, 4096, 24, 30522),
]

for name, d, h, ff, L, V in configs:
    params = count_params(d, h, ff, L, V)
    total  = params['TOTAL']
    print(f"  {name:<15}: ~{total/1e6:.0f}M params")
```

---

# 6. Training Loop

```python
import numpy as np

def cross_entropy_loss(logits, targets, ignore_index=-1):
    """
    logits:  (batch, seq, vocab)
    targets: (batch, seq) of token indices
    """
    batch, seq, vocab = logits.shape
    losses = []

    for b in range(batch):
        for s in range(seq):
            if targets[b, s] == ignore_index:
                continue
            # Stable softmax + log
            z        = logits[b, s]
            log_probs = z - np.log(np.sum(np.exp(z - z.max()))) - z.max()
            losses.append(-log_probs[targets[b, s]])

    return np.mean(losses) if losses else 0.0

def warmup_lr_schedule(step, d_model, warmup_steps=4000):
    """Original Transformer LR schedule."""
    step     = max(step, 1)
    factor   = d_model ** (-0.5)
    warmup   = step * warmup_steps ** (-1.5)
    decay    = step ** (-0.5)
    return factor * min(warmup, decay)

# Demonstrate training loop structure
print("Training loop structure:")
d_model      = 32
warmup_steps = 4000
steps        = [1, 100, 1000, 4000, 8000, 16000]
print(f"{'Step':>8} {'LR':>12}")
for s in steps:
    lr = warmup_lr_schedule(s, d_model, warmup_steps)
    print(f"{s:>8} {lr:.2e}")
```

---

# 7. Interview Q&A

## Q1: What makes the Transformer architecture more parallelizable than RNNs?

RNNs process tokens sequentially — step $t$ depends on step $t-1$. This prevents parallelization. Transformers process all tokens simultaneously using self-attention: every token attends to every other in one matrix operation. Training parallelizes across all sequence positions at once.

## Q2: Why are residual connections essential?

Residuals create a "highway" for gradients: $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot (1 + \frac{\partial F}{\partial x})$. The `+1` ensures gradient doesn't vanish through deep stacks. Without residuals, 12-layer networks struggle to train; with them, 100+ layers are stable.

## Q3: What is label smoothing and why use it?

Instead of one-hot targets $[0, 0, 1, 0, ...]$, use soft targets $[\epsilon/V, \epsilon/V, 1-\epsilon+\epsilon/V, ...]$ where $\epsilon \approx 0.1$. The model can never be infinitely confident, preventing overconfident predictions and improving calibration. Also acts as regularization.

## Q4: Why is LayerNorm applied before or after the sublayer (Pre-LN vs Post-LN)?

**Post-LN** (original paper): LN after residual. Gradient flows through LN, which can scale gradients unpredictably. Needs careful LR warmup.

**Pre-LN** (modern standard): LN before sublayer. Gradient bypasses LN via residual — more stable training, allows larger learning rates. GPT-2 and most modern models use Pre-LN.

## Q5: How does cross-attention work in the decoder?

Cross-attention uses **queries from the decoder** (what the decoder is looking for) and **keys + values from the encoder** (what the encoder has computed). This lets each decoder position attend to relevant encoder positions — the alignment mechanism that makes translation work.

---

# 8. Cheat Sheet

| Component | Formula | Role |
|-----------|---------|------|
| Token embedding | $E \in \mathbb{R}^{V \times d}$ | Convert token IDs to vectors |
| Positional encoding | $\sin/\cos(pos/10000^{2i/d})$ | Inject position info |
| Self-attention | $\text{softmax}(QK^T/\sqrt{d_k})V$ | Contextual representations |
| Residual | $x + \text{sublayer}(x)$ | Gradient highway |
| LayerNorm | $(x-\mu)/\sigma \cdot \gamma + \beta$ | Stable training |
| FFN | $\text{ReLU}(xW_1)W_2$ | Per-position transformation |
| Cross-attention | $Q$ from decoder, $K,V$ from encoder | Encoder-decoder connection |
| Causal mask | $\text{tril}(1^{n \times n})$ | Prevent future-token leakage |
| Params per layer | $\approx 12d^2$ | Parameter count estimation |

---

# MINI-PROJECT — Character-Level Language Model

**What you build:** A character-level Transformer that learns to generate text by predicting the next character. This is the simplest possible end-to-end Transformer you can train and understand completely.

---

## Step 1 — Prepare Data

```python
import numpy as np

rng = np.random.default_rng(42)

# Training text
text = """
the quick brown fox jumps over the lazy dog
machine learning transforms data into knowledge
neural networks learn by adjusting millions of weights
attention mechanisms allow models to focus on relevant context
transformers have revolutionized natural language processing
deep learning enables computers to learn from raw data
gradient descent finds the minimum of complex loss functions
backpropagation computes gradients through layers automatically
""".strip().lower()

# Character vocabulary
chars    = sorted(set(text))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
V        = len(chars)

data = np.array([char2idx[c] for c in text])

print(f"Text length: {len(text)} chars")
print(f"Vocabulary: {V} chars: {''.join(chars[:20])}...")
```

---

## Step 2 — Create Training Batches

```python
def get_batch(data, batch_size=4, seq_len=32):
    """Random batch of (input, target) sequences."""
    n    = len(data) - seq_len - 1
    idxs = rng.integers(0, n, batch_size)

    X = np.stack([data[i:i+seq_len] for i in idxs])
    y = np.stack([data[i+1:i+seq_len+1] for i in idxs])
    return X, y

X_batch, y_batch = get_batch(data)
print(f"\nBatch shapes: X={X_batch.shape}, y={y_batch.shape}")
print(f"Sample sequence: '{''.join(idx2char[i] for i in X_batch[0])}'")
```

---

## Step 3 — Minimal Character Transformer

```python
class CharTransformer:
    """Minimal decoder-only character Transformer."""

    def __init__(self, vocab_size, d_model=32, n_heads=4, n_layers=2, max_len=64):
        self.d_model = d_model
        rng          = np.random.default_rng(42)

        # Embeddings
        self.embed   = rng.normal(0, 0.02, (vocab_size, d_model))

        # Positional encoding
        pe  = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, None]
        i   = np.arange(0, d_model, 2)
        pe[:, 0::2] = np.sin(pos / np.power(10000.0, i/d_model))
        pe[:, 1::2] = np.cos(pos / np.power(10000.0, i/d_model))
        self.pe      = pe

        # Transformer layers
        self.layers  = [EncoderBlock(d_model, n_heads, d_model*4) for _ in range(n_layers)]

        # Output head
        self.W_out   = rng.normal(0, 0.02, (d_model, vocab_size))

    def forward(self, x_ids):
        """x_ids: (batch, seq)"""
        x   = self.embed[x_ids]           # (batch, seq, d)
        x  += self.pe[:x_ids.shape[1]]    # add PE

        # Attention is done without causal masking in EncoderBlock
        # For a proper language model, use DecoderBlock with causal mask
        for layer in self.layers:
            x = layer.forward(x)

        logits = x @ self.W_out            # (batch, seq, vocab)
        return logits

    def generate(self, seed_ids, max_new_tokens=50, temperature=1.0):
        """Autoregressive generation."""
        generated = list(seed_ids)

        for _ in range(max_new_tokens):
            ctx    = np.array([generated[-32:]])  # context window
            logits = self.forward(ctx)[0, -1]     # last position

            # Temperature sampling
            logits_t  = logits / temperature
            probs     = softmax(logits_t)
            next_tok  = rng.choice(V, p=probs)
            generated.append(next_tok)

        return generated

model = CharTransformer(V)

# Forward pass test
X_t, y_t = get_batch(data, batch_size=2, seq_len=16)
logits    = model.forward(X_t)
print(f"\nCharTransformer forward: {X_t.shape} → logits {logits.shape}")

# Generate (untrained — random output)
seed  = [char2idx[c] for c in "the "]
gen   = model.generate(seed, max_new_tokens=30, temperature=0.8)
print(f"\nGenerated (untrained): '{''.join(idx2char[i] for i in gen)}'")
```

---

## Step 4 — Simple Training

```python
def compute_loss(logits, targets):
    """Cross-entropy loss for character prediction."""
    batch, seq, vocab = logits.shape
    total_loss = 0.0
    count      = 0

    for b in range(batch):
        for s in range(seq):
            z         = logits[b, s]
            # Log-sum-exp for numerical stability
            max_z     = z.max()
            log_sm    = z[targets[b, s]] - max_z - np.log(np.sum(np.exp(z - max_z)))
            total_loss -= log_sm
            count      += 1

    return total_loss / count if count > 0 else 0.0

# Training loop (simplified — no gradient update, just loss measurement)
print("\nTraining loss measurement (10 random batches):")
losses = []
for step in range(10):
    X_b, y_b = get_batch(data, batch_size=4, seq_len=24)
    logits    = model.forward(X_b)
    loss      = compute_loss(logits, y_b)
    losses.append(loss)
    print(f"  Step {step+1}: loss={loss:.4f}")

# Baseline: random model predicts uniform → loss ≈ ln(V)
import math
baseline = math.log(V)
print(f"\nUntrained loss: {losses[-1]:.4f}")
print(f"Random baseline: {baseline:.4f}")
print(f"(After training loss should be significantly below baseline)")
```

---

## What This Project Demonstrated

| Concept | Where it appeared |
|---------|------------------|
| BPE tokenization | Character-level vocabulary (simpler case) |
| Token embedding | `self.embed[x_ids]` lookup |
| Sinusoidal PE | `self.pe` added to embeddings |
| Encoder block | `EncoderBlock.forward()` — MHA + FFN + LN |
| Causal masking | `DecoderBlock` causal mask (shown in code) |
| Cross-attention | `DecoderBlock` encoder-decoder connection |
| Residual connections | Inside `EncoderBlock` and `DecoderBlock` |
| Parameter counting | `count_params()` for known models |
| Autoregressive generation | `CharTransformer.generate()` |

---

*Next: [Module 08 — RAG Chatbot](08-rag.md)*
