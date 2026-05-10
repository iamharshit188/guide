"""
RAG evaluation metrics: faithfulness, answer relevancy, context precision/recall.
Covers: from-scratch metric implementations, RAGAS approximations without LLM judge,
        token overlap F1, embedding cosine similarity, BM25-based precision.
pip install numpy
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Token overlap F1 (approximate faithfulness / recall)
# ---------------------------------------------------------------------------

def token_overlap_f1(prediction: str, reference: str) -> float:
    """
    F1 between token sets of prediction and reference.
    Measures lexical overlap — used as proxy for content alignment.

    F1 = 2 * precision * recall / (precision + recall)
    precision = |pred ∩ ref| / |pred|
    recall    = |pred ∩ ref| / |ref|
    """
    pred_tokens = Counter(tokenize(prediction))
    ref_tokens = Counter(tokenize(reference))

    common = sum((pred_tokens & ref_tokens).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred_tokens.values())
    recall = common / sum(ref_tokens.values())
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, reference: str) -> float:
    """What fraction of reference tokens appear in prediction."""
    pred_tokens = set(tokenize(prediction))
    ref_tokens = Counter(tokenize(reference))
    hits = sum(cnt for tok, cnt in ref_tokens.items() if tok in pred_tokens)
    total = sum(ref_tokens.values())
    return hits / total if total > 0 else 0.0


def token_precision(prediction: str, reference: str) -> float:
    """What fraction of prediction tokens appear in reference."""
    pred_tokens = Counter(tokenize(prediction))
    ref_tokens = set(tokenize(reference))
    hits = sum(cnt for tok, cnt in pred_tokens.items() if tok in ref_tokens)
    total = sum(pred_tokens.values())
    return hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Faithfulness (claim-level, no LLM judge)
# ---------------------------------------------------------------------------

def faithfulness_token_overlap(answer: str, context_chunks: List[str],
                                per_sentence: bool = True) -> Dict[str, Any]:
    """
    Approximate RAGAS Faithfulness without an LLM judge.

    Strategy: decompose answer into sentences (atomic claims proxy),
    score each against context union via token recall.
    Faithfulness = fraction of sentences with recall > threshold.

    Real RAGAS uses an LLM to:
    1. Extract atomic claims from the answer.
    2. For each claim, ask: "Can this be inferred from the context?"
    """
    context_union = " ".join(context_chunks)

    if per_sentence:
        sentences = split_sentences(answer)
        if not sentences:
            return {"faithfulness": 0.0, "sentence_scores": []}
        sentence_scores = []
        for sent in sentences:
            recall = token_recall(sent, context_union)
            sentence_scores.append({"sentence": sent[:80], "recall": round(recall, 4)})
        # Sentence is "supported" if recall > 0.2 (at least some overlap)
        supported = sum(1 for s in sentence_scores if s["recall"] > 0.2)
        faithfulness = supported / len(sentences)
        return {
            "faithfulness": round(faithfulness, 4),
            "sentence_scores": sentence_scores,
            "supported": supported,
            "total_sentences": len(sentences),
        }
    else:
        recall = token_recall(answer, context_union)
        return {"faithfulness": round(recall, 4)}


# ---------------------------------------------------------------------------
# Answer Relevancy (embedding cosine similarity)
# ---------------------------------------------------------------------------

def _simple_embed(text: str, dim: int = 64) -> List[float]:
    """
    Character bigram embedding for answer relevancy proxy.
    Real RAGAS uses a sentence encoder (e.g., sentence-transformers).
    """
    tokens = tokenize(text)
    vec = [0.0] * dim
    for token in tokens:
        for i in range(len(token) - 1):
            h = (ord(token[i]) * 31 + ord(token[i+1])) % dim
            vec[h] += 1.0
    n = len(tokens) or 1
    vec = [v / n for v in vec]
    norm = math.sqrt(sum(v*v for v in vec))
    return [v / norm for v in vec] if norm > 1e-9 else vec


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb + 1e-9)


def answer_relevancy_sim(answer: str, question: str,
                          n_generated: int = 3) -> Dict[str, Any]:
    """
    Approximate RAGAS Answer Relevancy without an LLM judge.

    Real RAGAS:
    1. LLM generates n hypothetical questions from the answer.
    2. Score = mean cosine similarity of generated Q embeddings to original Q.

    Approximation: paraphrase the original question slightly and measure
    embedding similarity of answer to question.

    AnswerRelevancy = (1/n) Σ cos(embed(q_gen_i), embed(q_orig))
    """
    q_emb = _simple_embed(question)
    a_emb = _simple_embed(answer)

    # Direct similarity: answer embedding vs question embedding
    direct_sim = cosine_sim(a_emb, q_emb)

    # Sentence-level: find the sentence in the answer most similar to the question
    sentences = split_sentences(answer)
    sent_sims = []
    for sent in sentences:
        s_emb = _simple_embed(sent)
        sent_sims.append(cosine_sim(s_emb, q_emb))

    max_sim = max(sent_sims) if sent_sims else 0.0

    # Combined relevancy: geometric mean of direct and max-sentence similarity
    relevancy = math.sqrt(direct_sim * max_sim) if direct_sim > 0 and max_sim > 0 else 0.0

    return {
        "answer_relevancy": round(relevancy, 4),
        "direct_sim": round(direct_sim, 4),
        "max_sentence_sim": round(max_sim, 4),
        "sentence_sims": [round(s, 4) for s in sent_sims],
    }


# ---------------------------------------------------------------------------
# Context Precision
# ---------------------------------------------------------------------------

def context_precision_bm25(retrieved_chunks: List[str], query: str,
                            relevance_threshold: float = 0.3) -> Dict[str, Any]:
    """
    Approximate RAGAS Context Precision.

    Real RAGAS: LLM judges each retrieved chunk as relevant/irrelevant to the query.
    Approximation: BM25-style overlap between chunk and query.

    ContextPrecision = # relevant retrieved / # retrieved
    """
    q_tokens = set(tokenize(query))
    chunk_scores = []
    for i, chunk in enumerate(retrieved_chunks):
        c_tokens = set(tokenize(chunk))
        overlap = len(q_tokens & c_tokens) / (len(q_tokens | c_tokens) + 1e-9)
        is_relevant = overlap >= relevance_threshold
        chunk_scores.append({
            "chunk_index": i,
            "overlap": round(overlap, 4),
            "relevant": is_relevant,
            "preview": chunk[:60] + "..." if len(chunk) > 60 else chunk,
        })

    n_relevant = sum(1 for c in chunk_scores if c["relevant"])
    precision = n_relevant / len(retrieved_chunks) if retrieved_chunks else 0.0

    # Average Precision (AP): penalises irrelevant chunks earlier in the list
    ap_sum, hit_count = 0.0, 0
    for i, c in enumerate(chunk_scores):
        if c["relevant"]:
            hit_count += 1
            ap_sum += hit_count / (i + 1)
    avg_precision = ap_sum / n_relevant if n_relevant > 0 else 0.0

    return {
        "context_precision": round(precision, 4),
        "average_precision": round(avg_precision, 4),
        "n_relevant": n_relevant,
        "n_retrieved": len(retrieved_chunks),
        "chunk_scores": chunk_scores,
    }


# ---------------------------------------------------------------------------
# Context Recall
# ---------------------------------------------------------------------------

def context_recall_token_overlap(retrieved_chunks: List[str],
                                  ground_truth_answer: str) -> Dict[str, Any]:
    """
    Approximate RAGAS Context Recall.

    Real RAGAS: LLM checks if each sentence of the ground-truth answer
    can be attributed to the retrieved context.
    Approximation: token recall of GT answer against context union.

    ContextRecall = # GT claims supported by context / # GT claims
    """
    context_union = " ".join(retrieved_chunks)
    gt_sentences = split_sentences(ground_truth_answer)

    sentence_scores = []
    for sent in gt_sentences:
        recall = token_recall(sent, context_union)
        supported = recall > 0.25
        sentence_scores.append({
            "sentence": sent[:80],
            "recall": round(recall, 4),
            "supported": supported,
        })

    n_supported = sum(1 for s in sentence_scores if s["supported"])
    context_recall = n_supported / len(gt_sentences) if gt_sentences else 0.0
    overall_recall = token_recall(ground_truth_answer, context_union)

    return {
        "context_recall": round(context_recall, 4),
        "token_recall_overall": round(overall_recall, 4),
        "n_supported": n_supported,
        "n_gt_sentences": len(gt_sentences),
        "sentence_scores": sentence_scores,
    }


# ---------------------------------------------------------------------------
# RAGAS-style aggregate report
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Evaluates a RAG system over a test set.
    Each example: {"query", "answer", "context_chunks", "ground_truth" (optional)}
    """

    def evaluate(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        faithfulness_scores = []
        relevancy_scores = []
        precision_scores = []
        recall_scores = []

        per_example = []

        for ex in examples:
            query = ex["query"]
            answer = ex["answer"]
            chunks = ex["context_chunks"]
            gt = ex.get("ground_truth", "")

            faith = faithfulness_token_overlap(answer, chunks)
            relev = answer_relevancy_sim(answer, query)
            prec = context_precision_bm25(chunks, query)
            rec = context_recall_token_overlap(chunks, gt) if gt else None

            faithfulness_scores.append(faith["faithfulness"])
            relevancy_scores.append(relev["answer_relevancy"])
            precision_scores.append(prec["context_precision"])
            if rec:
                recall_scores.append(rec["context_recall"])

            per_example.append({
                "query": query[:60],
                "faithfulness": faith["faithfulness"],
                "answer_relevancy": relev["answer_relevancy"],
                "context_precision": prec["context_precision"],
                "context_recall": rec["context_recall"] if rec else None,
            })

        def mean(lst): return sum(lst) / len(lst) if lst else 0.0

        return {
            "aggregate": {
                "faithfulness": round(mean(faithfulness_scores), 4),
                "answer_relevancy": round(mean(relevancy_scores), 4),
                "context_precision": round(mean(precision_scores), 4),
                "context_recall": round(mean(recall_scores), 4) if recall_scores else None,
                "rag_score": round(mean(faithfulness_scores + relevancy_scores +
                                        precision_scores), 4),
            },
            "per_example": per_example,
            "n_examples": len(examples),
        }


# ---------------------------------------------------------------------------
# Demo dataset
# ---------------------------------------------------------------------------

TEST_EXAMPLES = [
    {
        "query": "How does scaled dot-product attention work?",
        "answer": (
            "Scaled dot-product attention computes queries, keys, and values. "
            "The dot products of queries and keys are divided by sqrt(d_k) to prevent "
            "vanishing gradients in the softmax. The result weights the values."
        ),
        "context_chunks": [
            "Scaled dot-product attention divides scores by sqrt(d_k) to stabilise gradients.",
            "The attention mechanism computes weighted sums of values using query-key similarities.",
            "Multi-head attention runs h parallel attention heads and concatenates their outputs.",
        ],
        "ground_truth": (
            "Attention computes queries dot keys divided by sqrt(d_k), applies softmax, "
            "then weights the values."
        ),
    },
    {
        "query": "What is BM25 and why is it useful for retrieval?",
        "answer": (
            "BM25 is a probabilistic ranking function based on term frequency. "
            "It uses inverse document frequency to weight rare terms higher. "
            "The k1 parameter controls TF saturation and b controls length normalisation."
        ),
        "context_chunks": [
            "BM25 ranks documents using term frequency and inverse document frequency.",
            "The k1 hyperparameter controls term frequency saturation in BM25.",
            "Dense retrieval embeds queries and documents into shared vector spaces.",
        ],
        "ground_truth": (
            "BM25 is a sparse retrieval function using TF-IDF statistics with "
            "document length normalisation."
        ),
    },
    {
        "query": "How does RAG prevent hallucination?",
        "answer": (
            "RAG retrieves relevant context from external documents and injects it into the prompt. "
            "The language model generates answers grounded in the retrieved evidence, "
            "reducing reliance on potentially incorrect memorised knowledge."
        ),
        "context_chunks": [
            "RAG injects retrieved context into the LLM prompt, grounding answers in evidence.",
            "RAG reduces hallucination by basing generation on retrieved documents.",
            "Source attribution is easy with RAG because chunk metadata tracks document origins.",
        ],
        "ground_truth": (
            "RAG grounds generation in retrieved documents, preventing the model from "
            "fabricating information not in the context."
        ),
    },
    {
        "query": "What is the capital of Mars?",
        "answer": "I don't know based on the provided context.",
        "context_chunks": [
            "Dense retrieval uses cosine similarity between embeddings.",
            "BM25 is a lexical retrieval function.",
        ],
        "ground_truth": "",  # no ground truth — testing "I don't know" case
    },
]


def main():
    section("FAITHFULNESS — TOKEN OVERLAP")
    for ex in TEST_EXAMPLES[:3]:
        result = faithfulness_token_overlap(ex["answer"], ex["context_chunks"])
        print(f"\n  Q: {ex['query'][:60]}...")
        print(f"  Faithfulness: {result['faithfulness']:.4f}  "
              f"({result['supported']}/{result['total_sentences']} sentences supported)")
        for s in result["sentence_scores"]:
            status = "✓" if s["recall"] > 0.2 else "✗"
            print(f"    {status} recall={s['recall']:.4f}  {s['sentence'][:70]}...")

    section("ANSWER RELEVANCY — EMBEDDING SIM")
    for ex in TEST_EXAMPLES:
        result = answer_relevancy_sim(ex["answer"], ex["query"])
        print(f"\n  Q: {ex['query'][:60]}...")
        print(f"  AnswerRelevancy: {result['answer_relevancy']:.4f}  "
              f"(direct={result['direct_sim']:.4f}, max_sent={result['max_sentence_sim']:.4f})")

    section("CONTEXT PRECISION — BM25 OVERLAP")
    for ex in TEST_EXAMPLES[:3]:
        result = context_precision_bm25(ex["context_chunks"], ex["query"])
        print(f"\n  Q: {ex['query'][:60]}...")
        print(f"  ContextPrecision: {result['context_precision']:.4f}  "
              f"AP={result['average_precision']:.4f}  "
              f"({result['n_relevant']}/{result['n_retrieved']} chunks relevant)")
        for c in result["chunk_scores"]:
            status = "✓" if c["relevant"] else "✗"
            print(f"    {status} overlap={c['overlap']:.4f}  {c['preview']}")

    section("CONTEXT RECALL — TOKEN OVERLAP")
    for ex in TEST_EXAMPLES[:3]:
        if not ex.get("ground_truth"):
            continue
        result = context_recall_token_overlap(ex["context_chunks"], ex["ground_truth"])
        print(f"\n  Q: {ex['query'][:60]}...")
        print(f"  Ground truth: {ex['ground_truth'][:80]}...")
        print(f"  ContextRecall: {result['context_recall']:.4f}  "
              f"(token_recall={result['token_recall_overall']:.4f}, "
              f"{result['n_supported']}/{result['n_gt_sentences']} GT sentences supported)")

    section("FULL RAGAS EVALUATION")
    evaluator = RAGEvaluator()
    report = evaluator.evaluate(TEST_EXAMPLES)

    agg = report["aggregate"]
    print(f"\n  N examples: {report['n_examples']}")
    print(f"\n  {'Metric':<25} {'Score':>8}")
    print(f"  {'-'*35}")
    print(f"  {'Faithfulness':<25} {agg['faithfulness']:>8.4f}")
    print(f"  {'Answer Relevancy':<25} {agg['answer_relevancy']:>8.4f}")
    print(f"  {'Context Precision':<25} {agg['context_precision']:>8.4f}")
    if agg['context_recall'] is not None:
        print(f"  {'Context Recall':<25} {agg['context_recall']:>8.4f}")
    print(f"  {'RAG Score (avg)':<25} {agg['rag_score']:>8.4f}")

    print(f"\n  Per-example breakdown:")
    print(f"  {'Query':<40} {'Faith':>6} {'Relev':>6} {'Prec':>6} {'Rec':>6}")
    print(f"  {'-'*65}")
    for ex in report["per_example"]:
        rec = f"{ex['context_recall']:.4f}" if ex['context_recall'] is not None else "  N/A"
        print(f"  {ex['query'][:40]:<40} "
              f"{ex['faithfulness']:>6.4f} "
              f"{ex['answer_relevancy']:>6.4f} "
              f"{ex['context_precision']:>6.4f} "
              f"{rec:>6}")

    section("METRIC COMPARISON TABLE")
    print("""
  ┌─────────────────────┬─────────────────────────────────────┬──────────────┐
  │ Metric              │ What it measures                    │ Needs GT?    │
  ├─────────────────────┼─────────────────────────────────────┼──────────────┤
  │ Faithfulness        │ Answer grounded in context          │ No           │
  │ Answer Relevancy    │ Answer addresses the question       │ No           │
  │ Context Precision   │ Retrieved chunks are relevant       │ No           │
  │ Context Recall      │ GT claims found in context          │ Yes          │
  └─────────────────────┴─────────────────────────────────────┴──────────────┘
""")

    section("TOKEN F1 vs EMBEDDING SIM — COMPARISON")
    pairs = [
        ("The attention scores are divided by sqrt(d_k).",
         "Scaled attention divides scores by the square root of d_k."),
        ("BM25 uses term frequency and inverse document frequency.",
         "Dense retrieval embeds documents into vector spaces."),
        ("RAG reduces hallucination by grounding answers.",
         "RAG prevents the model from making up information."),
    ]
    print(f"\n  {'Pair':<5} {'TokenF1':>8} {'EmbSim':>8} {'Interpretation'}")
    print(f"  {'-'*70}")
    for i, (a, b) in enumerate(pairs):
        f1 = token_overlap_f1(a, b)
        sim = cosine_sim(_simple_embed(a), _simple_embed(b))
        interp = "same fact" if f1 > 0.3 or sim > 0.6 else "different"
        print(f"  {i+1:<5} {f1:>8.4f} {sim:>8.4f}  {interp}")
        print(f"         A: {a[:60]}")
        print(f"         B: {b[:60]}")


if __name__ == "__main__":
    main()
