# Module 07 — Transformers from Scratch

> **Run:**
> ```bash
> cd src/07-transformer
> python tokenizer.py
> python model.py
> python model_numpy.py
> python train.py
> # C++ (requires g++):
> g++ -O2 -std=c++17 -o model_cpp model.cpp && ./model_cpp
> ```

---

## Prerequisites & Overview

**Prerequisites:** Module 06 (attention, MHA, positional encoding, KV cache). Module 05 (backprop, optimizers, LayerNorm). Python + NumPy; PyTorch optional (scripts degrade gracefully without it).
**Estimated time:** 8–12 hours (the PyTorch model and C++ implementation are optional depth dives)

### Why This Module Matters
Building a full transformer from scratch — tokenizer, encoder, decoder, training loop — is the single most effective way to develop true understanding of GPT-style models. After this module, reading the LLaMA or Mistral architecture papers feels like reading pseudocode, not a foreign language.

### What Gets Built

| Script | What It Implements | Language |
|--------|--------------------|----------|
| `tokenizer.py` | BPE from scratch: merge rules, encode/decode, OOV handling | Python |
| `model.py` | Full encoder-decoder transformer with Pre-LN, causal mask, greedy decode | PyTorch |
| `model_numpy.py` | Encoder-only transformer, verified against PyTorch to <1e-4 error | NumPy |
| `model.cpp` | Encoder forward pass with MHA + FFN + LayerNorm — no dependencies | C++17 |
| `train.py` | Sequence reversal task, transformer LR schedule, label smoothing, beam search | PyTorch |

### Before You Start
- Understand scaled dot-product attention: $\text{softmax}(QK^T/\sqrt{d_k})V$ (Module 06)
- Know what LayerNorm does (Module 05)
- Know what teacher forcing means in seq2seq (Module 06 Q&A section)

### Mental Model: Encoder vs Decoder
- **Encoder** (BERT-style): all tokens attend to all tokens (bidirectional). Used for classification, NER, embedding tasks.
- **Decoder** (GPT-style): each token attends only to past tokens (causal mask). Used for text generation.
- **Encoder-Decoder** (T5, original transformer): encoder processes source, decoder generates target conditioned on encoder output via cross-attention. Used for translation, summarization.

---

## 01 — BPE Tokenization

### Motivation

Character-level tokenisation: vocabulary of ~256, but very long sequences. Word-level: short sequences but OOV problem. BPE (Byte-Pair Encoding) finds a middle ground via a data-driven subword vocabulary.

### Algorithm

**Training:**

1. Initialise vocabulary with all characters present in corpus + special tokens (`<unk>`, `<pad>`, `<bos>`, `<eos>`)
2. Represent each word as a sequence of characters with a word-end marker: `"low"` → `['l', 'o', 'w</w>']`
3. Repeat for $k$ merge steps:
   a. Count all adjacent symbol pairs across the corpus
   b. Find the most frequent pair $(a, b)$
   c. Merge into a new symbol $ab$ everywhere it occurs
   d. Add merge rule $(a, b) \to ab$ to the merge table
4. Vocabulary = initial characters ∪ all merged symbols

**Encoding (inference):**

Given a new word, apply stored merge rules in order — earlier merges apply first. Produces a sequence of subword tokens.

### Complexity

| Operation | Time |
|---|---|
| Training | $O(V \cdot k \cdot N)$ where $N$ = corpus size, $k$ = merges |
| Encoding one word | $O(k \cdot |w|)$ |
| Vocabulary size | $|chars| + k$ |

### Sentencepiece / Tiktoken

Modern tokenisers (GPT-4 uses cl100k_base, LLaMA uses SentencePiece with BPE) operate on raw UTF-8 bytes — no pre-tokenisation step, handles any language and code. Tiktoken (OpenAI) applies BPE on bytes via compiled regex pre-split patterns.

---

## 02 — Transformer Architecture

### Full Architecture

$$\text{Transformer}(x, y) = \text{Decoder}(\text{Encoder}(x),\ y)$$

**Encoder:** $N_e$ identical layers, each:
$$\text{EncoderLayer}(x) = \text{LayerNorm}(x + \text{MHA}(x,x,x))$$
$$\text{EncoderLayer}_{\text{out}} = \text{LayerNorm}(\cdot + \text{FFN}(\cdot))$$

**Decoder:** $N_d$ identical layers, each:
$$z = \text{LayerNorm}(y + \text{MaskedMHA}(y,y,y))$$
$$z' = \text{LayerNorm}(z + \text{CrossAttn}(z,\text{enc},\text{enc}))$$
$$\text{DecoderLayer}_{\text{out}} = \text{LayerNorm}(z' + \text{FFN}(z'))$$

### Layer Norm

Per-sample normalisation across features (not batch):

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu = \frac{1}{d}\sum_i x_i$, $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$, and $\gamma, \beta \in \mathbb{R}^d$ are learnable.

Used in transformers (not BatchNorm) because: (1) works with variable-length sequences, (2) no dependency on batch size, (3) identical behaviour at train and inference.

### Feed-Forward Sublayer

Position-wise MLP applied identically to each position:

$$\text{FFN}(x) = \max(0,\ x W_1 + b_1) W_2 + b_2$$

- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$
- Typical: $d_{\text{ff}} = 4 d_{\text{model}}$
- Contains no cross-position interaction — attention handles that

Modern variants use GELU activation and SwiGLU (LLaMA):

$$\text{SwiGLU}(x, W, V, W_2) = (x W \odot \text{swish}(x V)) W_2$$

where $\text{swish}(x) = x \cdot \sigma(x)$. Requires 3 matrices instead of 2 but gives better performance.

### Residual Connections

Every sublayer wraps with `x + sublayer(x)`. Two design choices:

| | Pre-LN | Post-LN (original paper) |
|---|---|---|
| LN position | Before sublayer | After sublayer + residual |
| Gradient flow | Cleaner (unimpeded residual) | Can have vanishing gradients at init |
| Training stability | More stable, needs no LR warmup | Needs careful warmup |
| Used in | GPT-2+, most modern LLMs | Original 2017 paper |

---

## 03 — Parameter Count

For a single Transformer block with $d_{\text{model}}$, $h$ heads, $d_{\text{ff}} = 4d_{\text{model}}$:

| Component | Parameters |
|---|---|
| MHA: $W^Q, W^K, W^V$ | $3 d_{\text{model}}^2$ |
| MHA: $W^O$ | $d_{\text{model}}^2$ |
| FFN: $W_1, b_1$ | $d_{\text{model}} \cdot d_{\text{ff}} + d_{\text{ff}}$ |
| FFN: $W_2, b_2$ | $d_{\text{ff}} \cdot d_{\text{model}} + d_{\text{model}}$ |
| LayerNorm ×2 | $4 d_{\text{model}}$ |
| **Total per block** | $\approx 12 d_{\text{model}}^2$ |

For a $L$-layer model: $\approx 12 L d_{\text{model}}^2$ (plus embedding table $V d_{\text{model}}$).

**GPT-2 small check:** $L=12$, $d_{\text{model}}=768$

$$12 \times 12 \times 768^2 = 84,934,656 \approx 85\text{M}$$

Embedding: $50257 \times 768 = 38.6\text{M}$. Total ≈ 117M. ✓

---

## 04 — Training

### Loss Function

For language modelling (decoder-only), cross-entropy over next-token prediction:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(w_t \mid w_{<t}; \theta)$$

Teacher forcing: during training, always feed ground-truth previous tokens (not model predictions) — stabilises training but creates train/inference discrepancy (exposure bias).

### Learning Rate Schedule (Transformer)

$$\text{lr}(t) = d_{\text{model}}^{-0.5} \cdot \min\left(t^{-0.5},\ t \cdot t_{\text{warmup}}^{-1.5}\right)$$

- Linearly increases for $t \le t_{\text{warmup}}$ steps (default: 4000)
- Decays as $1/\sqrt{t}$ afterwards
- Peak LR occurs at $t = t_{\text{warmup}}$: $\text{lr}_{\text{peak}} = d_{\text{model}}^{-0.5} \cdot t_{\text{warmup}}^{-0.5}$

### Gradient Clipping

$$\text{if } \|\mathbf{g}\|_2 > \tau: \quad \mathbf{g} \leftarrow \tau \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|_2}$$

Clips the global gradient norm (not per-parameter). Typical $\tau = 1.0$. Prevents exploding gradients which are common in deep transformer training.

### Label Smoothing

Instead of one-hot targets, use:

$$q'(y) = (1 - \epsilon) q(y) + \epsilon / V$$

where $\epsilon = 0.1$ typically. Prevents the model from becoming overconfident, improves generalisation (BLEU score gains on translation tasks).

---

## 05 — Inference: Decoding Strategies

### Greedy Decoding

$$w_t = \arg\max_w P(w \mid w_{<t})$$

Fast but suboptimal — can miss high-probability sequences requiring a lower-probability first token.

### Beam Search

Maintain $B$ candidate sequences ("beams"), expand each by top-$B$ tokens, keep top-$B$ combined:

$$\text{score}(Y) = \frac{1}{|Y|^\alpha} \sum_{t} \log P(y_t \mid y_{<t})$$

$\alpha$ is length penalty (typically 0.6–0.7). Minimises length bias.

### Sampling-Based Methods

| Method | Formula | Parameter |
|---|---|---|
| Temperature | $P_\tau(w) \propto \exp(\log P(w) / \tau)$ | $\tau \in (0,1]$ sharpens; $\tau > 1$ flattens |
| Top-$k$ | Sample from top $k$ tokens only | $k = 50$ typical |
| Top-$p$ (nucleus) | Sample from smallest set with cumulative $P \ge p$ | $p = 0.9$ typical |
| Top-$k$ + Top-$p$ | Apply both filters, sample from intersection | — |

---

## 06 — Transformer Variants

| Model | Architecture | Key Change |
|---|---|---|
| BERT | Encoder-only | Bidirectional, MLM + NSP pretraining |
| GPT family | Decoder-only | Causal LM, no encoder |
| T5 | Encoder-decoder | All NLP tasks as text-to-text |
| LLaMA | Decoder-only | RoPE, SwiGLU, GQA, RMSNorm |
| Mistral | Decoder-only | GQA + sliding window attention |

**RMSNorm** (LLaMA): drops the mean subtraction from LayerNorm:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

Fewer parameters, empirically similar performance, faster computation.

---

## 07 — Interview Questions

**Q1: Why use Pre-LN instead of Post-LN?**
Post-LN: residual branch passes through LayerNorm before being added back — at initialisation, the gradient must flow through LN's learnable scale $\gamma$ (initialised near 1), making early gradients small. Pre-LN places LayerNorm inside the residual branch, leaving the residual path clean — gradient flows unimpeded through the addition, stabilising training without warmup.

**Q2: What is the role of the FFN sublayer if attention already mixes information?**
Attention is a (weighted) linear operation over values — it mixes positions but applies a linear transformation. FFN adds position-wise non-linearity. The attention + FFN pair is analogous to a key-value lookup (attention) followed by a non-linear transform (FFN). Interpretability work suggests FFN layers store factual associations (keys as patterns, values as "memories").

**Q3: Why does teacher forcing cause exposure bias?**
During training, the decoder always receives correct previous tokens. At inference, it receives its own (potentially wrong) predictions. This distributional shift between train and inference inputs degrades quality on long sequences — errors accumulate. Mitigations: scheduled sampling (gradually replace gold tokens with model predictions), REINFORCE, minimum Bayes risk decoding.

**Q4: How does BPE handle OOV words?**
BPE degrades gracefully: unknown words are decomposed to their characters (always in vocabulary). In byte-level BPE (GPT-4 tiktoken), input is first converted to UTF-8 bytes — the 256-byte vocabulary guarantees zero OOV for any input.

**Q5: What is the computational bottleneck in a Transformer?**
For short sequences ($n \ll d_{\text{model}}$): FFN dominates ($O(n d_{\text{ff}} d_{\text{model}})$ per layer). For long sequences ($n \gg d_{\text{model}}$): attention dominates ($O(n^2 d_{\text{model}})$). Crossover at $n \approx d_{\text{ff}} = 4 d_{\text{model}}$. For $d_{\text{model}} = 512$: crossover at $n \approx 2048$.

**Q6: What is gradient clipping and why is it critical for Transformers?**
Transformer gradients can explode transiently during training (large attention logits, near-zero LayerNorm denominators). Gradient clipping caps the global $\ell_2$ norm of all parameter gradients at threshold $\tau$ — if $\|\mathbf{g}\| > \tau$, all gradients are rescaled by $\tau / \|\mathbf{g}\|$. This preserves gradient direction while bounding the step size. Without it, a single bad batch can corrupt model weights.

---

## Resources

### Papers (Must-Read)
- **Attention Is All You Need** — Vaswani et al. (2017): `arxiv.org/abs/1706.03762`. Read alongside this module; every equation maps directly to a section here.
- **BERT: Pre-training of Deep Bidirectional Transformers** — Devlin et al. (2018): `arxiv.org/abs/1810.04805`
- **Language Models are Few-Shot Learners (GPT-3)** — Brown et al. (2020): `arxiv.org/abs/2005.14165`. Motivates why scale matters.
- **LLaMA 2** — Touvron et al. (2023): `arxiv.org/abs/2307.09288`. Modern architectural choices: RoPE, SwiGLU, GQA, RMSNorm.

### Visual Explainers
- **The Annotated Transformer** (`nlp.seas.harvard.edu/annotated-transformer/`): Harvard NLP. Line-by-line PyTorch implementation with equation annotations. The best technical reference.
- **Jay Alammar — "The Illustrated Transformer"** (`jalammar.github.io/illustrated-transformer/`): Tensor flow diagram that makes encoder-decoder cross-attention visual.

### Code References
- **nanoGPT** — Andrej Karpathy (`github.com/karpathy/nanoGPT`): ~300-line GPT-2 implementation in PyTorch. Read after finishing `model.py` in this module.
- **minGPT** — Andrej Karpathy (`github.com/karpathy/minGPT`): Slightly more structured version with training loop. Maps cleanly to `train.py`.
- **Hugging Face Transformers** (`github.com/huggingface/transformers`): Production reference implementation. `modeling_gpt2.py` is ~800 lines and directly comparable to the architecture here.

### Video
- **Andrej Karpathy — "Let's Build GPT from Scratch"** (YouTube, 2h): Implements a GPT decoder step by step, live. Closest video companion to this module.
- **Yannic Kilcher — Transformer paper walkthrough** (YouTube): Deep equation-level explanation with context.

---

*Next: [Module 08 — RAG Chatbot](08-rag.md)*
