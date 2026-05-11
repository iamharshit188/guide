"""
Image captioning with cross-modal attention from scratch.
Covers: cross-modal attention (Q from text, K/V from image),
        gated cross-attention (Flamingo), Q-Former concept (BLIP-2),
        teacher forcing, autoregressive inference — pure NumPy.
"""

import numpy as np

rng = np.random.default_rng(42)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + eps)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-9)


# ── Cross-Modal Attention ─────────────────────────────────────────
class CrossModalAttention:
    """
    Cross-attention where:
      Q = from text tokens     (the decoder querying the image)
      K = from image tokens    (what the image offers)
      V = from image tokens

    This is how the text decoder reads the visual context.
    In standard cross-attention: seq_q and seq_kv can differ in length.
    """

    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        scale = d_model ** -0.5

        self.W_Q = rng.standard_normal((d_model, d_model)) * scale
        self.W_K = rng.standard_normal((d_model, d_model)) * scale
        self.W_V = rng.standard_normal((d_model, d_model)) * scale
        self.W_O = rng.standard_normal((d_model, d_model)) * scale

    def forward(self, x_text: np.ndarray, x_image: np.ndarray) -> np.ndarray:
        """
        x_text:  (seq_t, d_model) — text tokens (source of Q)
        x_image: (seq_i, d_model) — image tokens (source of K, V)
        Returns: (seq_t, d_model)
        """
        seq_t = x_text.shape[0]
        seq_i = x_image.shape[0]
        h, dh = self.n_heads, self.d_head

        Q = x_text  @ self.W_Q    # (seq_t, d_model)
        K = x_image @ self.W_K    # (seq_i, d_model)
        V = x_image @ self.W_V    # (seq_i, d_model)

        # Split heads
        Q = Q.reshape(seq_t, h, dh).transpose(1, 0, 2)   # (h, seq_t, dh)
        K = K.reshape(seq_i, h, dh).transpose(1, 0, 2)   # (h, seq_i, dh)
        V = V.reshape(seq_i, h, dh).transpose(1, 0, 2)   # (h, seq_i, dh)

        scale = dh ** -0.5
        attn = softmax((Q @ K.transpose(0, 2, 1)) * scale)  # (h, seq_t, seq_i)
        out  = attn @ V                                       # (h, seq_t, dh)
        out  = out.transpose(1, 0, 2).reshape(seq_t, self.d_model)

        return out @ self.W_O, attn   # return attention weights for analysis


# ── Gated Cross-Attention (Flamingo) ──────────────────────────────
class GatedCrossAttention:
    """
    Flamingo-style gated cross-attention block.

    x = x + tanh(α_attn) · CrossAttn(LayerNorm(x), visual_tokens)
    x = x + tanh(α_ffn)  · FFN(LayerNorm(x))

    α_attn and α_ffn are learnable scalars initialized to 0.
    tanh(0) = 0 → at init, visual signals are zeroed out.
    This ensures the frozen LM is NOT disrupted at the start of training.
    As α is learned, visual information gradually enters the text stream.
    """

    def __init__(self, d_model: int, n_heads: int):
        self.cross_attn = CrossModalAttention(d_model, n_heads)
        self.d_model = d_model

        # Gating scalars (initialized to 0 → tanh(0)=0 → no visual signal at init)
        self.alpha_attn = 0.0
        self.alpha_ffn  = 0.0

        # FFN weights
        scale = d_model ** -0.5
        self.W1 = rng.standard_normal((d_model, 4 * d_model)) * scale
        self.W2 = rng.standard_normal((4 * d_model, d_model)) * scale

    def gelu(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))

    def forward(self, text_tokens: np.ndarray,
                visual_tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        text_tokens:   (seq_t, d_model)
        visual_tokens: (seq_v, d_model)
        Returns: (seq_t, d_model), attn_weights
        """
        # Gated cross-attention
        ca_out, attn = self.cross_attn.forward(
            layer_norm(text_tokens), visual_tokens
        )
        text_tokens = text_tokens + np.tanh(self.alpha_attn) * ca_out

        # Gated FFN
        ffn_in  = layer_norm(text_tokens)
        ffn_out = self.gelu(ffn_in @ self.W1) @ self.W2
        text_tokens = text_tokens + np.tanh(self.alpha_ffn) * ffn_out

        return text_tokens, attn


# ── Q-Former (BLIP-2 concept) ─────────────────────────────────────
class QFormer:
    """
    Simplified Q-Former (Querying Transformer from BLIP-2).

    32 learned query tokens attend to image features via cross-attention.
    Query tokens also attend to each other via self-attention.
    The Q-Former output (32, d_model) is the 'visual prompt' for the LLM.

    Advantage: compresses variable-length image tokens into fixed 32 tokens,
    aligning the vision backbone output with the LLM's input space.
    """

    def __init__(self, n_queries: int = 32, d_model: int = 64, n_heads: int = 4):
        self.n_queries = n_queries
        self.d_model   = d_model
        self.queries   = rng.standard_normal((n_queries, d_model)) * 0.02
        self.cross_attn = CrossModalAttention(d_model, n_heads)

        scale = d_model ** -0.5
        # Self-attention among queries
        self.W_Qs = rng.standard_normal((d_model, d_model)) * scale
        self.W_Ks = rng.standard_normal((d_model, d_model)) * scale
        self.W_Vs = rng.standard_normal((d_model, d_model)) * scale
        self.W_Os = rng.standard_normal((d_model, d_model)) * scale

    def self_attn(self, x: np.ndarray) -> np.ndarray:
        seq, d = x.shape
        Q = x @ self.W_Qs
        K = x @ self.W_Ks
        V = x @ self.W_Vs
        scale = d ** -0.5
        attn = softmax((Q @ K.T) * scale)
        return (attn @ V) @ self.W_Os

    def forward(self, image_tokens: np.ndarray) -> np.ndarray:
        """
        image_tokens: (N, d_model) — visual feature tokens from ViT
        Returns:      (n_queries, d_model) — compressed visual prompt
        """
        q = layer_norm(self.queries)
        # Self-attention among queries
        q = q + self.self_attn(q)
        # Cross-attention to image
        ca, _ = self.cross_attn.forward(layer_norm(q), image_tokens)
        q = q + ca
        return layer_norm(q)


# ── Token Vocabulary ──────────────────────────────────────────────
VOCAB = {
    "<BOS>": 0, "<EOS>": 1, "<PAD>": 2,
    "a": 3, "an": 4, "the": 5,
    "cat": 6, "dog": 7, "bird": 8, "car": 9,
    "sits": 10, "runs": 11, "flies": 12, "drives": 13,
    "on": 14, "in": 15, "near": 16, "above": 17,
    "grass": 18, "road": 19, "sky": 20, "park": 21,
    "white": 22, "black": 23, "red": 24, "blue": 25,
}
ID2WORD = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)


# ── Captioning Decoder ────────────────────────────────────────────
class CaptioningDecoder:
    """
    Minimal language model decoder for image captioning.
    Uses cross-modal attention to incorporate visual context.

    Training: teacher forcing (feed ground-truth tokens)
    Inference: autoregressive (feed previous predictions)
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4):
        self.d_model = d_model
        scale = d_model ** -0.5

        self.token_emb = rng.standard_normal((vocab_size, d_model)) * 0.02
        self.pos_emb   = rng.standard_normal((64, d_model)) * 0.02   # max_len=64

        # Self-attention (causal — for autoregressive decoding)
        self.W_Q_self = rng.standard_normal((d_model, d_model)) * scale
        self.W_K_self = rng.standard_normal((d_model, d_model)) * scale
        self.W_V_self = rng.standard_normal((d_model, d_model)) * scale
        self.W_O_self = rng.standard_normal((d_model, d_model)) * scale

        # Cross-modal attention
        self.cross_attn = CrossModalAttention(d_model, n_heads)

        # Output projection to vocab
        self.W_out = rng.standard_normal((d_model, vocab_size)) * scale

    def causal_self_attn(self, x: np.ndarray) -> np.ndarray:
        """Causal (masked) self-attention over text tokens."""
        seq, d = x.shape
        Q = x @ self.W_Q_self
        K = x @ self.W_K_self
        V = x @ self.W_V_self
        scale = d ** -0.5
        attn_raw = (Q @ K.T) * scale
        # Causal mask: position i cannot attend to j > i
        mask = np.tril(np.ones((seq, seq), dtype=bool))
        attn_raw = np.where(mask, attn_raw, -1e9)
        attn = softmax(attn_raw)
        return (attn @ V) @ self.W_O_self

    def forward(self, token_ids: np.ndarray,
                visual_tokens: np.ndarray) -> np.ndarray:
        """
        token_ids:     (seq_t,)  — input token indices
        visual_tokens: (seq_v, d_model)
        Returns:       (seq_t, vocab_size) — logits
        """
        seq_t = len(token_ids)
        x = self.token_emb[token_ids] + self.pos_emb[:seq_t]   # (seq_t, d_model)
        x = layer_norm(x)

        # Causal self-attention
        x = x + self.causal_self_attn(layer_norm(x))

        # Cross-modal attention (text queries image)
        ca_out, _ = self.cross_attn.forward(layer_norm(x), visual_tokens)
        x = x + ca_out

        x = layer_norm(x)
        return x @ self.W_out   # (seq_t, vocab_size)

    def teacher_forcing_loss(self, input_ids: np.ndarray, target_ids: np.ndarray,
                             visual_tokens: np.ndarray) -> float:
        """
        Teacher forcing: feed ground-truth tokens at each step.
        Loss = CrossEntropy(logits[t], target[t]) for t in 1..T
        input_ids:  [<BOS>, w1, w2, ..., wT-1]
        target_ids: [w1,    w2, w3, ..., wT, <EOS>]
        """
        logits = self.forward(input_ids, visual_tokens)  # (T, vocab)
        losses = []
        for t in range(len(target_ids)):
            lp = logits[t] - np.log(np.exp(logits[t]).sum() + 1e-9)
            losses.append(-lp[target_ids[t]])
        return float(np.mean(losses))

    def greedy_decode(self, visual_tokens: np.ndarray,
                      max_len: int = 12) -> list[int]:
        """
        Autoregressive greedy decoding.
        At each step: feed all generated tokens so far → take argmax.
        """
        ids = [VOCAB["<BOS>"]]
        for _ in range(max_len):
            logits = self.forward(np.array(ids), visual_tokens)
            next_id = int(np.argmax(logits[-1]))
            if next_id == VOCAB["<EOS>"]:
                break
            ids.append(next_id)
        return ids[1:]   # strip <BOS>

    def beam_search(self, visual_tokens: np.ndarray,
                    beam_width: int = 3, max_len: int = 12) -> list[int]:
        """
        Beam search: maintain top-k candidates at each step.
        Score = sum of log-probs / length (length-normalized).
        """
        # Each beam: (score, token_ids_list)
        beams = [(0.0, [VOCAB["<BOS>"]])]

        for _ in range(max_len):
            candidates = []
            for score, ids in beams:
                logits = self.forward(np.array(ids), visual_tokens)
                log_probs = logits[-1] - np.log(np.exp(logits[-1]).sum() + 1e-9)
                top_k = np.argsort(-log_probs)[:beam_width]
                for tok in top_k:
                    new_score = score + float(log_probs[tok])
                    candidates.append((new_score, ids + [int(tok)]))

            # Keep top beam_width by score normalized by length
            candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
            beams = candidates[:beam_width]

            # Stop if all beams ended with <EOS>
            if all(b[1][-1] == VOCAB["<EOS>"] for b in beams):
                break

        best = beams[0][1]
        return [t for t in best[1:] if t != VOCAB["<EOS>"]]   # strip <BOS>/<EOS>


def tokens_to_str(ids: list[int]) -> str:
    return " ".join(ID2WORD.get(i, "?") for i in ids)


def main():
    section("1. CROSS-MODAL ATTENTION")
    print("""
  Standard self-attention: Q, K, V all from same sequence.
  Cross-modal attention:   Q from text decoder, K and V from image encoder.

  At each text position t:
    Q_t = text_token_t · W_Q          (what the text is looking for)
    K_v = image_token_v · W_K         (what each image patch offers)
    V_v = image_token_v · W_V         (the content to extract)
    α_tv = softmax(Q_t · K_v^T / √dh) (how much token t attends to patch v)
    out_t = Σ_v α_tv · V_v            (weighted visual summary for token t)

  Result: each text token 'sees' a weighted blend of visual patches.
  Alignment learned: 'cat' → attends to animal-shaped patches, etc.
""")

    d_model, n_heads = 64, 4
    cma = CrossModalAttention(d_model, n_heads)

    seq_t = 6   # 6 text tokens
    seq_i = 16  # 16 image patches (4×4 grid)
    text_tok = rng.standard_normal((seq_t, d_model))
    img_tok  = rng.standard_normal((seq_i, d_model))

    out, attn_weights = cma.forward(text_tok, img_tok)
    print(f"  Text tokens: {text_tok.shape}  →  Cross-attn output: {out.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}  (heads, seq_t, seq_i)")
    print(f"\n  Per-head attention entropy (lower = more focused):")
    for h in range(n_heads):
        aw = attn_weights[h]
        ent = float(-np.sum(aw * np.log(aw + 1e-9), axis=-1).mean())
        peak = float(aw.max())
        print(f"    head {h}: entropy={ent:.4f}  peak_attn={peak:.4f}")

    section("2. GATED CROSS-ATTENTION (FLAMINGO)")
    print("""
  Flamingo freezes a pre-trained LLM and inserts gated cross-attention
  layers between the frozen LM layers.

  Gate:   x = x + tanh(α) · CrossAttn(LN(x), visual_tokens)
  Init:   α = 0  →  tanh(0) = 0  →  visual signal is zero at init
  Effect: frozen LM is NOT disrupted at start; vision learned gradually

  Why freeze the LM?
  - Preserves in-context learning + language generation quality
  - Only ≈1–2B new params (cross-attn + gates) vs. 70B LM params
  - Enables few-shot image captioning without full multimodal pre-training
""")

    gca = GatedCrossAttention(d_model, n_heads)
    text_in  = rng.standard_normal((seq_t, d_model))
    vis_feat = rng.standard_normal((seq_i, d_model))

    out_init, _ = gca.forward(text_in, vis_feat)
    diff_init = np.linalg.norm(out_init - text_in)
    print(f"  α=0 (init): |output - input| = {diff_init:.6f}  (≈0 = visual signal gated out)")

    # Simulate trained state: α = 1.0
    gca.alpha_attn = 1.0
    gca.alpha_ffn  = 1.0
    out_trained, _ = gca.forward(text_in, vis_feat)
    diff_trained = np.linalg.norm(out_trained - text_in)
    print(f"  α=1.0 (trained): |output - input| = {diff_trained:.4f}  (visual signal flows)")

    section("3. Q-FORMER (BLIP-2 CONCEPT)")
    print("""
  BLIP-2 introduces the Q-Former between a frozen image encoder and LLM.
  32 learned query tokens → cross-attend to image patches → compress to 32 tokens
  Those 32 tokens become the visual prompt for the LLM (via a linear projection).

  Architecture:
    Image encoder → (N, d_v) image tokens
    Q-Former:       32 queries + cross-attn → (32, d_q)
    Linear:         (32, d_q) → (32, d_llm)   align to LLM input space
    LLM:            visual_prompt + text_tokens → generation

  Why 32 queries?
  - Fixed input length regardless of image resolution
  - Forces compression → learns semantically meaningful summary
  - Much cheaper than passing 196+ image tokens directly to LLM
""")

    n_patches = 49   # 7×7 for 32px image, simulated
    img_feats = rng.standard_normal((n_patches, d_model))

    qformer = QFormer(n_queries=8, d_model=d_model, n_heads=n_heads)  # 8 for demo
    visual_prompt = qformer.forward(img_feats)
    print(f"  Image features:  {img_feats.shape}  ({n_patches} patches)")
    print(f"  Visual prompt:   {visual_prompt.shape}  (compressed to 8 query tokens)")
    print(f"  Compression:     {n_patches}×{d_model} → {visual_prompt.shape[0]}×{d_model}")

    section("4. TEACHER FORCING vs. AUTOREGRESSIVE")
    print("""
  Teacher Forcing (training):
    Input:   [<BOS>, w1, w2, ..., wT-1]  (ground-truth shifted right)
    Target:  [w1,    w2, ..., wT, <EOS>]
    At each position t: ground-truth token is fed (not model's prediction)
    Loss: CrossEntropy(logits[t], target[t])

    Advantage: stable gradients, no error compounding during training
    Disadvantage: exposure bias — model never sees own mistakes during train

  Autoregressive (inference):
    Feed <BOS>, sample w1, feed [<BOS>, w1], sample w2, ...
    Stop when <EOS> or max_len reached

  Scheduled sampling: gradually replace gold tokens with model predictions
  during training to reduce exposure bias.
""")

    decoder = CaptioningDecoder(VOCAB_SIZE, d_model, n_heads)
    vis = visual_prompt   # use Q-Former output as visual context

    # Teacher forcing
    caption  = [VOCAB["a"], VOCAB["cat"], VOCAB["sits"], VOCAB["on"], VOCAB["grass"]]
    input_ids  = np.array([VOCAB["<BOS>"]] + caption[:-1])
    target_ids = np.array(caption)
    loss = decoder.teacher_forcing_loss(input_ids, target_ids, vis)
    print(f"  Teacher forcing loss (untrained model): {loss:.4f}")
    print(f"  Expected ~log(VOCAB_SIZE) = {np.log(VOCAB_SIZE):.4f}")

    section("5. GREEDY DECODE vs. BEAM SEARCH")
    print("  Generating captions (random weights — output is noise):")

    greedy_ids  = decoder.greedy_decode(vis, max_len=8)
    beam_ids    = decoder.beam_search(vis, beam_width=3, max_len=8)

    print(f"  Greedy: [{tokens_to_str(greedy_ids)}]  ({len(greedy_ids)} tokens)")
    print(f"  Beam-3: [{tokens_to_str(beam_ids)}]  ({len(beam_ids)} tokens)")

    print("""
  Decoding strategies:
  ┌──────────────┬──────────────────────────────────────────────────┐
  │ Strategy     │ Properties                                       │
  ├──────────────┼──────────────────────────────────────────────────┤
  │ Greedy       │ Fast, deterministic, often suboptimal sequences  │
  │ Beam search  │ Better quality, O(beam_width) more compute       │
  │ Top-k        │ Diverse, fast; k=50 common                       │
  │ Top-p        │ Nucleus sampling; p=0.9 standard                 │
  │ Temperature  │ Scale logits; τ<1 sharp, τ>1 flat               │
  └──────────────┴──────────────────────────────────────────────────┘
""")

    section("6. CAPTIONING ARCHITECTURE COMPARISON")
    print("""
  ┌────────────────┬─────────────────────────────────────────────────────┐
  │ Model          │ Architecture                                        │
  ├────────────────┼─────────────────────────────────────────────────────┤
  │ Show&Tell      │ CNN → LSTM decoder (no cross-attn)                  │
  │ Show,Attend    │ CNN + spatial attention → LSTM (Bahdanau 2015)      │
  │ OSCAR          │ ViT + BERT, region-tag alignment pre-training       │
  │ BLIP           │ ViT + Med-Encoder-Decoder, bootstrapped captions    │
  │ BLIP-2         │ Frozen ViT + Q-Former + Frozen LLM (OPT/FlanT5)    │
  │ Flamingo       │ Frozen ViT + gated cross-attn + Frozen LLM (Chinch)│
  │ LLaVA          │ Frozen CLIP-ViT + MLP proj + Frozen LLaMA          │
  │ GPT-4V         │ Proprietary; likely similar architecture            │
  └────────────────┴─────────────────────────────────────────────────────┘

  Trend: freeze both image encoder (CLIP-ViT) and LLM, train only
  the 'bridge' (Q-Former, linear projection, or gated cross-attn).
  This preserves model quality while adding multimodal capability.
""")


if __name__ == "__main__":
    main()
