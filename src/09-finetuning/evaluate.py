"""
Fine-tuning evaluation: perplexity, BLEU, ROUGE-L from scratch (pure numpy/stdlib).
pip install numpy
"""

import math
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np

RNG = np.random.default_rng(42)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Tokenisation (simple whitespace + punctuation split)
# ---------------------------------------------------------------------------

def tokenise(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def perplexity_from_logprobs(logprobs: List[float]) -> float:
    """
    PPL = exp(-1/T * sum(log p(w_t | w_{<t})))
    logprobs: list of log p(w_t | w_{<t}) for each token t
    """
    T = len(logprobs)
    if T == 0:
        return float("inf")
    avg_nll = -sum(logprobs) / T
    return math.exp(avg_nll)


def perplexity_from_cross_entropy(ce_per_token: List[float]) -> float:
    """
    ce_per_token: list of -log p(w_t | w_{<t}) values (positive, per token).
    PPL = exp(mean(ce_per_token))
    """
    if not ce_per_token:
        return float("inf")
    return math.exp(sum(ce_per_token) / len(ce_per_token))


def simulate_perplexity(vocab_size: int = 1000, seq_len: int = 50,
                        model_quality: float = 0.8) -> Dict:
    """
    Simulate token-level log probabilities for a model with given quality.
    model_quality in [0, 1]: 1.0 = perfect (PPL=1), 0.0 = uniform (PPL=vocab_size).
    """
    # Simulate: mix of peaked (good) and uniform (uncertain) distributions
    logprobs = []
    for _ in range(seq_len):
        if RNG.random() < model_quality:
            # Model is confident: high prob on correct token
            p_correct = RNG.uniform(0.5, 0.95)
        else:
            # Model is uncertain: near-uniform
            p_correct = 1.0 / vocab_size
        logprobs.append(math.log(p_correct + 1e-10))

    ppl = perplexity_from_logprobs(logprobs)
    avg_nll = -sum(logprobs) / len(logprobs)

    return {
        "perplexity": ppl,
        "avg_nll": avg_nll,
        "seq_len": seq_len,
        "model_quality": model_quality,
    }


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def modified_ngram_precision(hypothesis: List[str], reference: List[str],
                              n: int) -> Tuple[int, int]:
    """
    Modified n-gram precision: clips hypothesis n-gram counts to reference counts.
    Returns (clipped_count, total_hypothesis_ngrams).
    """
    hyp_ngrams = _ngrams(hypothesis, n)
    ref_ngrams = _ngrams(reference, n)

    clipped = 0
    for ngram, count in hyp_ngrams.items():
        clipped += min(count, ref_ngrams.get(ngram, 0))

    total = max(1, sum(hyp_ngrams.values()))
    return clipped, total


def brevity_penalty(hypothesis_len: int, reference_len: int) -> float:
    """
    BP = 1 if c > r, else exp(1 - r/c)
    c = hypothesis length, r = closest reference length.
    """
    c = hypothesis_len
    r = reference_len
    if c == 0:
        return 0.0
    if c > r:
        return 1.0
    return math.exp(1 - r / c)


def bleu_score(hypothesis: str, reference: str,
               max_n: int = 4,
               weights: Optional[List[float]] = None) -> Dict:
    """
    Compute BLEU-N score.
    BLEU = BP * exp(sum_n w_n * log p_n)
    Returns individual p_n values plus final BLEU.
    """
    hyp_tokens = tokenise(hypothesis)
    ref_tokens = tokenise(reference)

    if weights is None:
        weights = [1.0 / max_n] * max_n

    precisions = []
    log_precisions = []

    for n in range(1, max_n + 1):
        if len(hyp_tokens) < n:
            precisions.append(0.0)
            log_precisions.append(float("-inf"))
            continue
        clipped, total = modified_ngram_precision(hyp_tokens, ref_tokens, n)
        p_n = clipped / total if total > 0 else 0.0
        precisions.append(p_n)
        log_precisions.append(math.log(p_n) if p_n > 0 else float("-inf"))

    bp = brevity_penalty(len(hyp_tokens), len(ref_tokens))

    # Geometric mean of precisions weighted
    if any(math.isinf(lp) for lp in log_precisions):
        bleu = 0.0
    else:
        log_bleu = sum(w * lp for w, lp in zip(weights, log_precisions))
        bleu = bp * math.exp(log_bleu)

    return {
        "bleu": bleu,
        "bp": bp,
        "precisions": precisions,
        "hyp_len": len(hyp_tokens),
        "ref_len": len(ref_tokens),
    }


def corpus_bleu(hypotheses: List[str], references: List[str],
                max_n: int = 4) -> Dict:
    """
    Corpus-level BLEU: aggregate clipped counts across all sentences,
    then compute single precision and BP at corpus level.
    """
    clipped_totals = [0] * max_n
    hyp_totals = [0] * max_n
    total_hyp_len = 0
    total_ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = tokenise(hyp)
        ref_tokens = tokenise(ref)
        total_hyp_len += len(hyp_tokens)
        total_ref_len += len(ref_tokens)

        for n in range(1, max_n + 1):
            if len(hyp_tokens) < n:
                continue
            clipped, total = modified_ngram_precision(hyp_tokens, ref_tokens, n)
            clipped_totals[n-1] += clipped
            hyp_totals[n-1] += total

    bp = brevity_penalty(total_hyp_len, total_ref_len)
    precisions = []
    log_ps = []
    for n in range(max_n):
        p = clipped_totals[n] / hyp_totals[n] if hyp_totals[n] > 0 else 0.0
        precisions.append(p)
        log_ps.append(math.log(p) if p > 0 else float("-inf"))

    if any(math.isinf(lp) for lp in log_ps):
        bleu = 0.0
    else:
        bleu = bp * math.exp(sum(lp / max_n for lp in log_ps))

    return {
        "corpus_bleu": bleu,
        "bp": bp,
        "precisions": precisions,
        "hyp_len": total_hyp_len,
        "ref_len": total_ref_len,
    }


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def lcs_length(x: List[str], y: List[str]) -> int:
    """
    Classic LCS dynamic programming: O(|x||y|) time and space.
    L[i][j] = LCS length of x[:i] and y[:j].
    """
    m, n = len(x), len(y)
    # Use 1D rolling array for space efficiency: O(n) space
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def rouge_l(hypothesis: str, reference: str, beta: float = 1.0) -> Dict:
    """
    ROUGE-L = F1 based on LCS.
    P_lcs = LCS / |hyp|
    R_lcs = LCS / |ref|
    F_lcs = (1+beta^2) * P * R / (R + beta^2 * P)
    """
    hyp_tokens = tokenise(hypothesis)
    ref_tokens = tokenise(reference)

    if not hyp_tokens or not ref_tokens:
        return {"rouge_l": 0.0, "precision": 0.0, "recall": 0.0, "lcs": 0}

    lcs = lcs_length(hyp_tokens, ref_tokens)
    p = lcs / len(hyp_tokens)
    r = lcs / len(ref_tokens)

    if p + r == 0:
        f = 0.0
    else:
        f = (1 + beta**2) * p * r / (r + beta**2 * p)

    return {"rouge_l": f, "precision": p, "recall": r, "lcs": lcs}


def rouge_1(hypothesis: str, reference: str) -> Dict:
    """ROUGE-1: unigram F1."""
    hyp = tokenise(hypothesis)
    ref = tokenise(reference)
    if not hyp or not ref:
        return {"rouge_1": 0.0, "precision": 0.0, "recall": 0.0}
    hyp_c = Counter(hyp)
    ref_c = Counter(ref)
    overlap = sum((hyp_c & ref_c).values())
    p = overlap / len(hyp)
    r = overlap / len(ref)
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"rouge_1": f, "precision": p, "recall": r}


def rouge_2(hypothesis: str, reference: str) -> Dict:
    """ROUGE-2: bigram F1."""
    hyp = tokenise(hypothesis)
    ref = tokenise(reference)
    if len(hyp) < 2 or len(ref) < 2:
        return {"rouge_2": 0.0, "precision": 0.0, "recall": 0.0}
    hyp_c = _ngrams(hyp, 2)
    ref_c = _ngrams(ref, 2)
    overlap = sum((hyp_c & ref_c).values())
    p = overlap / max(sum(hyp_c.values()), 1)
    r = overlap / max(sum(ref_c.values()), 1)
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"rouge_2": f, "precision": p, "recall": r}


# ---------------------------------------------------------------------------
# Evaluation suite
# ---------------------------------------------------------------------------

EVAL_PAIRS = [
    {
        "question": "What is gradient descent?",
        "reference": (
            "Gradient descent iteratively updates parameters by moving opposite "
            "to the gradient of the loss: theta = theta - lr * grad(L)."
        ),
        "before_ft": (
            "Gradient descent is an optimisation algorithm used in machine learning "
            "to minimise a function by iteratively moving towards the steepest descent."
        ),
        "after_ft": (
            "Gradient descent updates parameters theta by: theta = theta - lr * grad(L). "
            "SGD uses one sample, mini-batch uses a subset, full-batch uses all data."
        ),
    },
    {
        "question": "What is attention in transformers?",
        "reference": (
            "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V. "
            "The scaling by sqrt(d_k) prevents vanishing gradients in high dimensions."
        ),
        "before_ft": (
            "Attention is a mechanism that allows the model to focus on "
            "different parts of the input sequence."
        ),
        "after_ft": (
            "Attention computes: softmax(QK^T / sqrt(d_k)) V. "
            "Scaling by sqrt(d_k) keeps dot products from growing too large in high dimensions."
        ),
    },
    {
        "question": "Explain overfitting.",
        "reference": (
            "Overfitting: model learns training data noise, fails on unseen data. "
            "Caused by high variance and low bias. Prevented by regularisation and dropout."
        ),
        "before_ft": (
            "Overfitting occurs when a model memorises training data "
            "and cannot generalise to new examples."
        ),
        "after_ft": (
            "Overfitting: model memorises training noise (high variance, low bias). "
            "Prevention: L1/L2 regularisation, dropout, early stopping, data augmentation."
        ),
    },
]


def evaluate_pair(pair: Dict, model_key: str) -> Dict:
    hyp = pair[model_key]
    ref = pair["reference"]
    bleu = bleu_score(hyp, ref, max_n=4)
    rl = rouge_l(hyp, ref)
    r1 = rouge_1(hyp, ref)
    r2 = rouge_2(hyp, ref)
    return {
        "bleu4": bleu["bleu"],
        "bp": bleu["bp"],
        "rouge_l": rl["rouge_l"],
        "rouge_1": r1["rouge_1"],
        "rouge_2": r2["rouge_2"],
        "lcs": rl["lcs"],
    }


def aggregate(results: List[Dict]) -> Dict:
    keys = [k for k in results[0] if isinstance(results[0][k], float)]
    return {k: sum(r[k] for r in results) / len(results) for k in keys}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    section("PERPLEXITY — FORMULA AND SIMULATION")
    print("""
  PPL = exp(-1/T * sum_{t=1}^{T} log p(w_t | w_{<t}))
      = exp(mean cross-entropy per token)

  PPL = 1:     perfect model — always assigns prob 1 to correct token
  PPL = k:     as uncertain as uniform over k choices per token
  PPL = vocab: equivalent to random guess
    """)

    print(f"  {'model_quality':>15} {'PPL':>10} {'Avg NLL':>10}")
    print(f"  {'-'*40}")
    for q in [0.95, 0.80, 0.60, 0.40, 0.20, 0.05]:
        result = simulate_perplexity(vocab_size=1000, seq_len=100, model_quality=q)
        print(f"  {q:>15.2f} {result['perplexity']:>10.2f} {result['avg_nll']:>10.4f}")

    section("BLEU — MODIFIED N-GRAM PRECISION")
    print("\n  Example computation:")
    hyp = "The cat sat on the mat"
    ref = "The cat is sitting on the mat"
    result = bleu_score(hyp, ref, max_n=4)
    print(f"  Hypothesis: {hyp!r}")
    print(f"  Reference:  {ref!r}")
    print(f"\n  {'n':>4}  {'p_n':>10}")
    for i, p in enumerate(result["precisions"], 1):
        print(f"  {i:>4}  {p:>10.4f}")
    print(f"\n  BP = {result['bp']:.4f}")
    print(f"  BLEU-4 = {result['bleu']:.4f}")

    section("BREVITY PENALTY ANALYSIS")
    print(f"\n  {'hyp_len':>8}  {'ref_len':>8}  {'BP':>8}")
    print(f"  {'-'*30}")
    ref_len = 20
    for hyp_len in [5, 10, 15, 18, 20, 25, 30]:
        bp = brevity_penalty(hyp_len, ref_len)
        print(f"  {hyp_len:>8}  {ref_len:>8}  {bp:>8.4f}")

    section("ROUGE-L — LCS DYNAMIC PROGRAMMING")
    print(f"""
  LCS recurrence:
    L[i][j] = L[i-1][j-1] + 1       if X[i] == Y[j]
             = max(L[i-1][j], L[i][j-1])  otherwise

  ROUGE-L precision = LCS / |hypothesis|
  ROUGE-L recall    = LCS / |reference|
  ROUGE-L F1        = 2 * P * R / (P + R)   (beta=1)
    """)

    hyp = "the quick brown fox"
    ref = "the brown fox jumped"
    lcs = lcs_length(tokenise(hyp), tokenise(ref))
    rl = rouge_l(hyp, ref)
    print(f"  Hypothesis: {hyp!r}")
    print(f"  Reference:  {ref!r}")
    print(f"  LCS length: {lcs}")
    print(f"  ROUGE-L P={rl['precision']:.4f}  R={rl['recall']:.4f}  F1={rl['rouge_l']:.4f}")

    section("BEFORE vs AFTER FINE-TUNING — METRIC COMPARISON")
    before_results = [evaluate_pair(p, "before_ft") for p in EVAL_PAIRS]
    after_results = [evaluate_pair(p, "after_ft") for p in EVAL_PAIRS]
    before_avg = aggregate(before_results)
    after_avg = aggregate(after_results)

    print(f"\n  {'Metric':>12}  {'Before FT':>12}  {'After FT':>12}  {'Delta':>10}")
    print(f"  {'-'*52}")
    for key in ("bleu4", "rouge_l", "rouge_1", "rouge_2"):
        b = before_avg[key]
        a = after_avg[key]
        delta = a - b
        sign = "+" if delta >= 0 else ""
        print(f"  {key:>12}  {b:>12.4f}  {a:>12.4f}  {sign}{delta:>9.4f}")

    section("PER-EXAMPLE BREAKDOWN")
    print(f"\n  {'Ex':>4}  {'Before BLEU':>12}  {'After BLEU':>12}  "
          f"{'Before ROUGE-L':>15}  {'After ROUGE-L':>14}")
    print(f"  {'-'*65}")
    for i, (b, a) in enumerate(zip(before_results, after_results)):
        print(f"  {i+1:>4}  {b['bleu4']:>12.4f}  {a['bleu4']:>12.4f}  "
              f"{b['rouge_l']:>15.4f}  {a['rouge_l']:>14.4f}")

    section("METRIC PROPERTIES TABLE")
    print(f"""
  Metric       | Range  | Reference | Measures            | Limit
  ─────────────┼────────┼───────────┼─────────────────────┼─────────────────────
  Perplexity   | [1,∞)  | No        | Token probability   | Tokeniser-dependent
  BLEU-4       | [0,1]  | Yes       | 4-gram precision    | Ignores semantics
  ROUGE-1      | [0,1]  | Yes       | Unigram overlap     | Word order ignored
  ROUGE-2      | [0,1]  | Yes       | Bigram overlap      | Phrase coverage
  ROUGE-L      | [0,1]  | Yes       | LCS subsequence     | Flexible ordering
  BERTScore    | ~[0,1] | Yes       | Semantic similarity | Slow, needs BERT
  Human eval   | [1,5]  | Yes       | Overall quality     | Expensive
    """)

    section("CORPUS BLEU EXAMPLE")
    hyps = [p["after_ft"] for p in EVAL_PAIRS]
    refs = [p["reference"] for p in EVAL_PAIRS]
    cb = corpus_bleu(hyps, refs, max_n=4)
    print(f"\n  Corpus BLEU-4: {cb['corpus_bleu']:.4f}")
    print(f"  Brevity penalty: {cb['bp']:.4f}")
    print(f"  Per-n precision: {[round(p, 4) for p in cb['precisions']]}")
    print(f"\n  Note: corpus BLEU > avg(sentence BLEU) because clipping is less strict")
    print(f"  at corpus level — aggregating counts reduces zero-precision n-grams.")


if __name__ == "__main__":
    main()
