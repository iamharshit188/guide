"""
Word2Vec skip-gram with negative sampling — pure NumPy.
Covers: vocabulary building, noise distribution, gradient updates, cosine similarity eval.
pip install numpy
"""

import numpy as np
import math
from collections import Counter

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Corpus and vocabulary
# ---------------------------------------------------------------------------

CORPUS = """
the king loves the queen and the queen loves the king
the man is brave and the woman is wise
the dog chases the cat and the cat runs away
machine learning is a field of artificial intelligence
deep learning uses neural networks with many layers
transformers use attention mechanisms for sequence modeling
word embeddings capture semantic meaning of words
the cat sat on the mat near the dog
paris is to france as berlin is to germany
man is to king as woman is to queen
"""

def build_vocab(text, min_count=1):
    tokens = text.lower().split()
    counts = Counter(tokens)
    vocab = [w for w, c in counts.items() if c >= min_count]
    vocab = sorted(vocab)
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    freq = np.array([counts[w] for w in vocab], dtype=float)
    return w2i, i2w, freq

def build_skipgram_pairs(text, w2i, window=2):
    tokens = [t for t in text.lower().split() if t in w2i]
    pairs = []
    for t, center in enumerate(tokens):
        c_idx = w2i[center]
        lo = max(0, t - window)
        hi = min(len(tokens), t + window + 1)
        for j in range(lo, hi):
            if j != t:
                pairs.append((c_idx, w2i[tokens[j]]))
    return pairs


# ---------------------------------------------------------------------------
# Noise distribution P_n(w) ∝ f(w)^(3/4)
# ---------------------------------------------------------------------------

def noise_distribution(freq):
    p = freq ** 0.75
    return p / p.sum()


# ---------------------------------------------------------------------------
# Skip-gram model
# ---------------------------------------------------------------------------

class SkipGramNS:
    """Skip-gram with negative sampling."""

    def __init__(self, vocab_size, embed_dim, rng):
        self.V = vocab_size
        self.d = embed_dim
        scale = 1.0 / math.sqrt(embed_dim)
        # centre embeddings (input)
        self.W_in  = rng.uniform(-scale, scale, (vocab_size, embed_dim))
        # context embeddings (output)
        self.W_out = rng.uniform(-scale, scale, (vocab_size, embed_dim))

    def _sigmoid(self, x):
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    def train_pair(self, center_idx, context_idx, neg_indices, lr):
        """SGD step for one (centre, context) positive pair + K negatives."""
        v_c = self.W_in[center_idx]          # (d,)
        u_o = self.W_out[context_idx]        # (d,)
        u_k = self.W_out[neg_indices]        # (K, d)

        # positive loss: -log σ(u_o · v_c)
        s_pos = self._sigmoid(u_o @ v_c)
        # negative loss: -sum log σ(-u_k · v_c)
        s_neg = self._sigmoid(u_k @ v_c)     # (K,)

        # gradient w.r.t. v_c
        d_vc = (s_pos - 1.0) * u_o + (s_neg @ u_k)   # (d,)

        # gradient w.r.t. u_o (positive)
        d_uo = (s_pos - 1.0) * v_c

        # gradient w.r.t. u_k (negatives) — shape (K, d)
        d_uk = np.outer(s_neg, v_c)

        # update
        self.W_in[center_idx]    -= lr * d_vc
        self.W_out[context_idx]  -= lr * d_uo
        self.W_out[neg_indices]  -= lr * d_uk

        # loss for monitoring
        loss = -math.log(s_pos + 1e-9) - np.sum(np.log(1.0 - s_neg + 1e-9))
        return loss

    def get_embedding(self, idx):
        # average of in + out embeddings (common practice)
        return (self.W_in[idx] + self.W_out[idx]) / 2.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, pairs, noise_dist, K, epochs, lr, rng, log_every=1000):
    V = model.V
    indices = np.arange(V)
    total_loss = 0.0
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        shuffled = rng.permutation(len(pairs))
        for pi in shuffled:
            center_idx, context_idx = pairs[pi]
            # sample K negatives — avoid positive
            neg = rng.choice(indices, size=K, replace=True, p=noise_dist)
            neg = np.where(neg == context_idx,
                           (neg + 1) % V,
                           neg)
            loss = model.train_pair(center_idx, context_idx, neg, lr)
            epoch_loss += loss
            step += 1
        avg = epoch_loss / max(len(pairs), 1)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={avg:.4f}")
    return model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def most_similar(word, model, w2i, i2w, topn=5):
    if word not in w2i:
        return []
    idx = w2i[word]
    vec = model.get_embedding(idx)
    V = model.V
    sims = []
    for i in range(V):
        if i == idx:
            continue
        sims.append((cosine_similarity(vec, model.get_embedding(i)), i2w[i]))
    sims.sort(reverse=True)
    return sims[:topn]

def analogy(a, b, c, model, w2i, i2w, topn=3):
    """a : b :: c : ?   →  vec(b) - vec(a) + vec(c)"""
    missing = [w for w in [a, b, c] if w not in w2i]
    if missing:
        return f"OOV: {missing}"
    va = model.get_embedding(w2i[a])
    vb = model.get_embedding(w2i[b])
    vc = model.get_embedding(w2i[c])
    target = vb - va + vc
    exclude = {w2i[a], w2i[b], w2i[c]}
    sims = []
    for i in range(model.V):
        if i in exclude:
            continue
        sims.append((cosine_similarity(target, model.get_embedding(i)), i2w[i]))
    sims.sort(reverse=True)
    return sims[:topn]


# ---------------------------------------------------------------------------
# Noise distribution demo
# ---------------------------------------------------------------------------

def demo_noise_dist(freq, i2w):
    section("NOISE DISTRIBUTION P_n(w) ∝ f(w)^(3/4)")
    raw = freq / freq.sum()
    smooth = noise_distribution(freq)
    print(f"{'Word':<18} {'Raw freq':>10} {'Smooth P_n':>12}")
    top5 = np.argsort(freq)[-5:][::-1]
    for i in top5:
        print(f"  {i2w[i]:<16} {raw[i]:>10.4f} {smooth[i]:>12.4f}")
    print("\n  Smoothing reduces dominance of frequent words.")
    print(f"  Max raw: {raw.max():.4f}  →  max smooth: {smooth.max():.4f}")


# ---------------------------------------------------------------------------
# Gradient verification (single step)
# ---------------------------------------------------------------------------

def numerical_gradient(model, center_idx, context_idx, neg_indices, eps=1e-5):
    """Central difference on W_in[center_idx] first component."""
    orig = model.W_in[center_idx, 0]

    def loss_fn():
        v_c = model.W_in[center_idx]
        u_o = model.W_out[context_idx]
        u_k = model.W_out[neg_indices]
        sig = lambda x: np.where(x >= 0,
                                 1.0 / (1.0 + np.exp(-x)),
                                 np.exp(x) / (1.0 + np.exp(x)))
        s_pos = sig(u_o @ v_c)
        s_neg = sig(u_k @ v_c)
        return -math.log(float(s_pos) + 1e-9) - float(np.sum(np.log(1.0 - s_neg + 1e-9)))

    model.W_in[center_idx, 0] = orig + eps
    lp = loss_fn()
    model.W_in[center_idx, 0] = orig - eps
    lm = loss_fn()
    model.W_in[center_idx, 0] = orig
    return (lp - lm) / (2 * eps)


def main():
    rng = np.random.default_rng(42)

    section("VOCABULARY AND CORPUS STATS")
    w2i, i2w, freq = build_vocab(CORPUS)
    V = len(w2i)
    pairs = build_skipgram_pairs(CORPUS, w2i, window=2)
    print(f"  Vocabulary size : {V}")
    print(f"  Training pairs  : {len(pairs)}")
    print(f"  Sample pairs    : {[(i2w[c], i2w[o]) for c, o in pairs[:6]]}")

    demo_noise_dist(freq, i2w)

    section("NEGATIVE SAMPLING GRADIENT CHECK")
    noise_dist = noise_distribution(freq)
    mini_model = SkipGramNS(V, embed_dim=8, rng=rng)
    ci, oi = w2i["king"], w2i["queen"]
    neg = np.array([w2i["dog"], w2i["cat"]])
    analytic_grad = (mini_model._sigmoid(
        mini_model.W_out[oi] @ mini_model.W_in[ci]) - 1.0) * mini_model.W_out[oi, 0] + (
        mini_model._sigmoid(mini_model.W_out[neg] @ mini_model.W_in[ci]) @ mini_model.W_out[neg, 0])
    numeric_grad = numerical_gradient(mini_model, ci, oi, neg)
    rel_err = abs(analytic_grad - numeric_grad) / (abs(numeric_grad) + 1e-9)
    print(f"  Analytic grad  : {analytic_grad:.6f}")
    print(f"  Numerical grad : {numeric_grad:.6f}")
    print(f"  Relative error : {rel_err:.2e}  {'PASS' if rel_err < 1e-3 else 'FAIL'}")

    section("TRAINING — SKIP-GRAM WITH NEGATIVE SAMPLING")
    EMBED_DIM = 32
    K = 5
    EPOCHS = 30
    LR = 0.025
    model = SkipGramNS(V, EMBED_DIM, rng)
    print(f"  vocab={V}, embed_dim={EMBED_DIM}, K={K}, epochs={EPOCHS}, lr={LR}")
    train(model, pairs, noise_dist, K=K, epochs=EPOCHS, lr=LR, rng=rng)

    section("NEAREST NEIGHBOURS (COSINE SIMILARITY)")
    for query in ["king", "learning", "cat"]:
        if query not in w2i:
            continue
        neighbours = most_similar(query, model, w2i, i2w, topn=4)
        print(f"\n  most_similar('{query}'):")
        for sim, word in neighbours:
            print(f"    {word:<18} sim={sim:+.4f}")

    section("ANALOGY: a:b :: c:?   (vec(b) - vec(a) + vec(c))")
    tests = [
        ("man", "king", "woman"),
        ("france", "paris", "germany"),
    ]
    for a, b, c in tests:
        result = analogy(a, b, c, model, w2i, i2w, topn=3)
        if isinstance(result, str):
            print(f"  {a}:{b}::{c}:?  →  {result}")
        else:
            top = [(w, f"{s:+.4f}") for s, w in result]
            print(f"  {a}:{b}::{c}:?  →  {top}")
    print("\n  Note: small corpus limits analogy quality.")
    print("  On Wikipedia-scale text (3B tokens), Word2Vec achieves >70% analogy accuracy.")

    section("EMBEDDING PROPERTIES")
    print("  Embedding matrix W_in shape:", model.W_in.shape)
    norms = np.linalg.norm(model.W_in, axis=1)
    print(f"  Embedding norm — mean: {norms.mean():.4f}  std: {norms.std():.4f}")
    print("\n  CBOW vs Skip-Gram comparison:")
    print("  ┌──────────────────────┬──────────────────────┐")
    print("  │ CBOW                 │ Skip-Gram            │")
    print("  ├──────────────────────┼──────────────────────┤")
    print("  │ Faster training      │ Slower (more pairs)  │")
    print("  │ Better frequent wds  │ Better rare words    │")
    print("  │ Average context vecs │ Predict each context │")
    print("  └──────────────────────┴──────────────────────┘")

    section("FASTTEXT SUBWORD REPRESENTATION (DEMO)")
    word = "running"
    ngrams = []
    padded = f"<{word}>"
    for n in range(3, 7):
        for i in range(len(padded) - n + 1):
            ngrams.append(padded[i:i+n])
    ngrams.append(f"<{word}>")
    print(f"  FastText n-grams for '{word}' (n=3..6):")
    print(f"  {ngrams}")
    print(f"  The word embedding = mean of {len(ngrams)} n-gram vectors.")
    print("  Enables OOV handling: 'unrunnable' → average of its n-gram vectors.")


if __name__ == "__main__":
    main()
