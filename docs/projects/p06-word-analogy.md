# Project 06 — Word Analogy & Similarity Explorer

> **Difficulty:** Intermediate · **Module:** 06 — GenAI Core
> **Skills:** Word2Vec skip-gram, negative sampling, cosine similarity, analogy solving, KV cache stub

---

## What You'll Build

Train a Word2Vec skip-gram model with negative sampling on a 10K-sentence corpus. Evaluate cosine nearest-neighbors for 5 test words, solve 10 analogy pairs with vector arithmetic, stub an autoregressive generator with a KV cache structure, and print accuracy + similarity tables.

---

## Skills Exercised

- Skip-gram training: for each center word, predict context words within window
- Negative sampling loss: $\mathcal{L} = -\log\sigma(\mathbf{v}_o^T\mathbf{v}_c) - \sum_{k}\log\sigma(-\mathbf{v}_k^T\mathbf{v}_c)$
- Noise distribution: $P_n(w) \propto f(w)^{3/4}$
- Cosine similarity: $\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}$
- Analogy: $\text{nearest}(\mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c)$, excluding a, b, c from candidates
- KV cache: dict mapping token_id → (key_vector, value_vector), simulating autoregressive generation

---

## Approach

### Phase 1 — Corpus prep
```
corpus: download Project Gutenberg text (e.g. Pride and Prejudice ~10K sentences)
  or use: nltk.corpus.brown / gutenberg
tokenize: lowercase, split on whitespace+punctuation, min_freq=5
build vocab: {word: idx}, idx_to_word reverse map
build noise distribution P_n: (freq^0.75) / sum(freq^0.75)
```

### Phase 2 — Skip-gram data generator
```
for sentence in corpus:
    for center_pos, center_word in enumerate(sentence):
        for offset in range(-window, window+1):
            if offset == 0: continue
            context_word = sentence[center_pos + offset]
            yield (center_idx, context_idx)
```

### Phase 3 — Model + training loop
```
W_in  = rng.normal(0, 0.01, (vocab_size, embed_dim))   # center embeddings
W_out = np.zeros((vocab_size, embed_dim))               # context embeddings

for epoch in range(n_epochs):
    for center, context in training_pairs:
        neg_samples = sample from P_n, excluding center and context
        # forward: v_c = W_in[center], v_o = W_out[context], v_k = W_out[neg_k]
        # loss = -log(sigmoid(v_o @ v_c)) - sum(log(sigmoid(-v_k @ v_c)))
        # backward: compute gradients, update W_in[center] and W_out entries
    print epoch loss
```

### Phase 4 — Evaluation
```
nearest_neighbors(word, top_k=5):
    v = W_in[word_idx]  # normalized
    sims = W_in @ v / (||W_in|| * ||v||)
    return top_k indices excluding word itself

analogy(a, b, c, top_k=1):
    query = W_in[b] - W_in[a] + W_in[c]
    sims = W_in @ query / (||W_in|| * ||query||)
    exclude indices for a, b, c
    return top_k

test words: ["king", "woman", "good", "run", "city"]
test analogies: [("man","king","woman"), ("paris","france","berlin"),
                 ("good","better","bad"), ...]
```

### Phase 5 — KV cache stub
```
kv_cache = {}   # token_id → (k_vec, v_vec)

def generate_step(token_id, W_in, W_out):
    if token_id not in kv_cache:
        k_vec = W_in[token_id]
        v_vec = W_out[token_id]
        kv_cache[token_id] = (k_vec, v_vec)
    k, v = kv_cache[token_id]
    # simulate attention: score against all cached keys
    keys = np.array([kv_cache[t][0] for t in kv_cache])
    attn = softmax(keys @ k / np.sqrt(embed_dim))
    vals = np.array([kv_cache[t][1] for t in kv_cache])
    output = attn @ vals
    return output

# generate 3 continuations of length 10
for seed_word in ["king", "city", "run"]:
    tokens = [word_to_idx[seed_word]]
    for _ in range(10):
        out = generate_step(tokens[-1], W_in, W_out)
        next_token = np.argmax(out @ W_in.T)
        tokens.append(next_token)
    print(" ".join(idx_to_word[t] for t in tokens))
```

---

## Checkpoints

| Phase | What correct output looks like |
|-------|-------------------------------|
| 1 | `vocab_size` between 2K–8K depending on corpus; noise dist sums to 1.0 |
| 2 | Training pairs count ≈ `n_sentences × avg_len × 2×window` |
| 3 | Loss decreases each epoch; epoch 10 loss < epoch 1 loss by ≥20% |
| 4 | `nearest("king")` includes "queen", "prince", or "monarch"; analogy king:queen ≈ man:woman |
| 5 | Cache hit avoids recomputing k/v for seen tokens; generated text is token sequences (not necessarily coherent) |

---

## Extensions

1. **FastText subword n-grams** — for each word, add n-grams (3–6 chars) to the vocabulary; use sum of n-gram embeddings as word representation; compare OOV handling.
2. **OOV comparison table** — test 10 OOV words (misspellings, rare words); FastText handles them via n-grams; Word2Vec returns <UNK>.
3. **Semantic drift** — train on two halves of corpus separately; compare top-10 neighbors of 5 words across both models; words that drift most = context-dependent.

---

## Hints

<details><summary>Hint 1 — Sigmoid numerical stability</summary>
<code>def sigmoid(x): return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))</code>
</details>

<details><summary>Hint 2 — Gradient for negative sampling</summary>
dL/dv_c = (σ(v_o·v_c) - 1)·v_o + Σ σ(v_k·v_c)·v_k<br>
dL/dv_o = (σ(v_o·v_c) - 1)·v_c<br>
dL/dv_k = σ(v_k·v_c)·v_c (for each negative sample k)
</details>

<details><summary>Hint 3 — Normalizing for cosine similarity</summary>
Pre-normalize all embeddings once: <code>W_norm = W_in / np.linalg.norm(W_in, axis=1, keepdims=True)</code>. Then all similarity queries are just dot products.
</details>

<details><summary>Hint 4 — Corpus source</summary>
<code>import urllib.request; urllib.request.urlretrieve("https://www.gutenberg.org/files/1342/1342-0.txt", "pride.txt")</code> — Pride and Prejudice is public domain.
</details>

<details><summary>Hint 5 — KV cache analogy to Transformer</summary>
In a real Transformer, the cache stores the K and V projections from all previous positions. Your stub mimics this: once a token's K/V is computed, it's reused in all future steps without recomputation. The memory grows O(T) with sequence length T.
</details>

---

*Back to [Module 06 — GenAI Core](../06-genai-core.md)*
