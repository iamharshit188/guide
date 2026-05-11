# Project 07 — Shakespeare GPT

> **Difficulty:** Advanced · **Module:** 07 — Transformers from Scratch
> **Skills:** BPE tokenizer, GPT decoder, LR schedule, gradient clipping, label smoothing, sampling strategies

---

## What You'll Build

A character-GPT trained on Shakespeare's complete works. BPE tokenizer, 4-layer GPT decoder (d=128, 4 heads, ctx=128), transformer LR schedule, gradient clipping, label smoothing. Train/val loss every 100 steps. Checkpoint every 500 steps. Generate 5 samples at temperature 1.0, 0.7, and top-p=0.9. Print parameter breakdown by component.

---

## Skills Exercised

- BPE tokenizer: merge loop on Shakespeare corpus (target vocab ~1000–2000)
- GPT decoder block: masked MHA + FFN + Pre-LN
- Parameter count: $12Ld^2$ per block + embeddings
- Transformer LR schedule: $d^{-0.5}\min(t^{-0.5}, t \cdot t_w^{-1.5})$
- Gradient clipping: `clip_grad_norm(params, max_norm=1.0)`
- Label smoothing: $\tilde{y}_i = (1-\epsilon)y_i + \epsilon/V$, $\epsilon=0.1$
- Temperature sampling, top-k, top-p (nucleus) sampling

---

## Approach

### Phase 1 — Tokenizer
```
download shakespeare.txt (~1.1M chars)
train BPE: start with char vocab, merge most frequent pair each step
target: 1000–2000 merges
encode full corpus → token ids
build train/val split: first 90% train, last 10% val
```

### Phase 2 — Model architecture
```
config: n_layers=4, d_model=128, n_heads=4, d_ff=512, ctx_len=128, dropout=0.1

Embedding: token_embed(vocab_size, 128) + pos_embed(128, 128)

GPTBlock:
    Pre-LayerNorm → MHA (causal mask) → residual
    Pre-LayerNorm → FFN (Linear→GELU→Linear) → residual

Head: LayerNorm → Linear(128, vocab_size)

Parameter count table:
  token_embed:  vocab_size × 128
  pos_embed:    128 × 128
  per block:    ~12 × 128² = 196,608
  4 blocks:     786,432
  head:         128 × vocab_size
  TOTAL: print this
```

### Phase 3 — Training loop
```
optimizer: AdamW(lr=0, beta1=0.9, beta2=0.95, weight_decay=0.1)
schedule: transformer warmup (t_warmup=100 steps)

for step in range(max_steps):
    x, y = get_batch(train_data, batch_size=32, ctx_len=128)
    logits = model(x)                       # (B, T, V)
    loss = label_smooth_ce(logits, y)
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    optimizer.step(); scheduler.step()
    optimizer.zero_grad()

    if step % 100 == 0:
        val_loss = evaluate(model, val_data)
        print(f"step {step}: train={loss:.4f} val={val_loss:.4f}")

    if step % 500 == 0:
        save_checkpoint(model, optimizer, step, val_loss)
```

### Phase 4 — Sampling
```
def temperature_sample(logits, temp=1.0):
    probs = softmax(logits / temp)
    return np.random.choice(len(probs), p=probs)

def top_p_sample(logits, p=0.9):
    probs = softmax(logits)
    sorted_idx = np.argsort(probs)[::-1]
    cumprob = np.cumsum(probs[sorted_idx])
    cutoff = sorted_idx[cumprob <= p]
    # renormalize over cutoff tokens, sample

generate(model, seed_text, max_new=200, temp=1.0):
    tokens = tokenizer.encode(seed_text)
    for _ in range(max_new):
        ctx = tokens[-ctx_len:]
        logits = model(ctx)[-1]         # last token logits
        next_tok = temperature_sample(logits, temp)
        tokens.append(next_tok)
    return tokenizer.decode(tokens)
```

### Phase 5 — Output
```
=== Parameter Breakdown ===
token_embed:   vocab_size × 128 = ...
pos_embed:     128 × 128        = ...
4 × GPTBlock:  4 × 196,608     = ...
lm_head:       128 × vocab_size = ...
TOTAL: ...

=== Training (every 100 steps) ===
step   0: train=7.1234 val=7.1289
step 100: train=4.2341 val=4.5123
...

=== Generated Samples (temp=1.0) ===
[sample 1 text]
[sample 2 text]

=== Generated Samples (temp=0.7) ===
...

=== Generated Samples (top-p=0.9) ===
...
```

---

## Checkpoints

| Phase | What correct output looks like |
|-------|-------------------------------|
| 1 | BPE encodes "HAMLET:" to ~3–5 tokens; decode(encode(text)) == text |
| 2 | Parameter count matches your manual calculation ±5% |
| 3 | Val loss < 2.5 by end of training (CPU ≈ 20–30 min at 5000 steps) |
| 4 | temp=0.7 output is less random than temp=1.0; top-p concentrates mass on likely tokens |
| 5 | Checkpoints saved as `.npz` or `.pkl`; reload and continue training |

---

## Extensions

1. **Repetition penalty** — divide logit of recently generated tokens by `penalty=1.3`; compare output diversity vs. base temp=1.0.
2. **Beam search** — implement beam search (width=4); compare log-prob of beam output vs. greedy; beam should have higher total log-prob.
3. **Scale to 6 layers** — increase `n_layers=6`, `d_model=192`, `n_heads=6`; compare final val loss and total parameter count; estimate training time increase.

---

## Hints

<details><summary>Hint 1 — BPE merge loop</summary>
Count all adjacent token-pair frequencies. Merge the most frequent pair into a new token. Repeat. Stop at target vocab size. Use a list-of-lists representation for the corpus to mutate efficiently.
</details>

<details><summary>Hint 2 — Causal mask</summary>
Upper-triangular mask: <code>mask = np.triu(np.ones((T,T)), k=1) * -1e9</code>. Add to attention scores before softmax so future positions have near-zero weight.
</details>

<details><summary>Hint 3 — Label smoothing implementation</summary>
<code>smooth_y = (1 - eps) * one_hot(y) + eps / vocab_size</code>. Then loss = <code>-np.sum(smooth_y * log_softmax(logits))</code>.
</details>

<details><summary>Hint 4 — Gradient clipping without PyTorch</summary>
Collect all gradients into a list, compute global norm: <code>total_norm = sqrt(sum(||g||² for g in grads))</code>. If > max_norm, scale all grads by <code>max_norm / total_norm</code>.
</details>

<details><summary>Hint 5 — Getting the model to train on CPU</summary>
ctx_len=128, batch_size=32, d_model=128, 4 layers is the sweet spot for CPU training. AdamW with lr_max=3e-4 and 100-step warmup converges in ~5000 steps (~20 min on modern CPU).
</details>

---

*Back to [Module 07 — Transformers from Scratch](../07-transformers.md)*
