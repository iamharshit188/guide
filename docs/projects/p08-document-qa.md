# Project 08 — Personal Document Q&A System

> **Difficulty:** Advanced · **Module:** 08 — RAG Chatbot
> **Skills:** Recursive chunking, TF-IDF + random projection, all four retrieval strategies, RAGAS evaluation, Flask API

---

## What You'll Build

Use the 9 module guides in `docs/` as your corpus. Recursive chunk, embed with TF-IDF + random projection, build four retrieval strategies (BM25, dense, hybrid RRF, MMR), run 20 test questions with ground-truth answers, compute all 4 RAGAS metrics, and print a strategy comparison table. Serve via Flask API.

---

## Skills Exercised

- Recursive character splitting: prefer paragraph → sentence → char splits; 512-token chunks, 50-token overlap
- TF-IDF from scratch + random projection to 64 dims
- BM25 ($k_1=1.5$, $b=0.75$)
- Hybrid RRF: $\text{score}(d) = \frac{1}{60+r_{\text{dense}}} + \frac{1}{60+r_{\text{BM25}}}$
- MMR: $\text{MMR}(d) = \lambda\cdot\text{sim}(d,q) - (1-\lambda)\cdot\max_{s\in S}\text{sim}(d,s)$
- RAGAS metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Flask API: `POST /ingest`, `POST /query`, `GET /collection`, `DELETE /collection`

---

## Approach

### Phase 1 — Ingest corpus
```
files = glob.glob("docs/*.md")   # 9 module guides
for file in files:
    text = open(file).read()
    chunks = recursive_split(text, chunk_size=512, overlap=50, separators=["\n\n", "\n", ". ", " "])
    for chunk in chunks:
        doc_store.append({"text": chunk, "source": file, "chunk_id": uuid4()})
print(f"Total chunks: {len(doc_store)}")  # expect 200–500
```

### Phase 2 — Embed + index
```
# TF-IDF
tokenize all chunks → build vocab
compute TF-IDF matrix: (n_chunks × vocab_size)
random projection to 64 dims: R = rng.normal(0, 1/sqrt(64), (vocab_size, 64))
embeddings = tfidf_matrix @ R           # (n_chunks, 64)
normalize rows

# BM25
inverted_index: dict[term → list[(chunk_id, freq)]]
precompute avgdl, doc_lengths
```

### Phase 3 — Four retrieval functions
```
dense_retrieve(query, top_k):
    q_vec = embed(query)
    sims = embeddings @ q_vec
    return top_k chunks by cosine sim

bm25_retrieve(query, top_k):
    tokens = tokenize(query)
    score each chunk via BM25 formula
    return top_k chunks

hybrid_rrf(query, top_k):
    d_results = dense_retrieve(query, top_k=20)
    b_results = bm25_retrieve(query, top_k=20)
    rrf_score each doc, return top_k

mmr_retrieve(query, top_k, lambda_=0.5):
    candidates = dense_retrieve(query, top_k=20)
    selected = []
    while len(selected) < top_k and candidates:
        if not selected:
            best = candidates[0]
        else:
            best = argmax(lambda_*sim(c,q) - (1-lambda_)*max(sim(c,s) for s in selected))
        selected.append(best); candidates.remove(best)
    return selected
```

### Phase 4 — 20 test Q&A pairs
```
Create 20 questions from the module guides with known ground-truth answers:
Q: "What is the transformer LR schedule formula?"
A: "d^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})"
source: docs/07-transformers.md

Q: "What does NF4 stand for in QLoRA?"
A: "Normal Float 4-bit, quantized using quantiles of the standard normal distribution"
source: docs/09-finetuning.md
... (create 20 total)
```

### Phase 5 — RAGAS evaluation
```
for strategy in [dense, bm25, hybrid_rrf, mmr]:
    precisions, recalls, faithfulnesses, relevancies = [], [], [], []
    for q, answer, ground_truth in test_set:
        retrieved = strategy(q, top_k=5)
        context = "\n".join(c["text"] for c in retrieved)
        generated = mock_llm(q, context)  # or openai (graceful skip)

        # Context Precision: fraction of retrieved chunks containing ground_truth terms
        # Context Recall: fraction of ground_truth terms covered by retrieved chunks
        # Faithfulness: fraction of generated answer claims entailed by context (keyword overlap)
        # Answer Relevancy: cosine similarity of answer embedding to question embedding
        precisions.append(context_precision(retrieved, ground_truth))
        recalls.append(context_recall(retrieved, ground_truth))
        faithfulnesses.append(faithfulness(generated, context))
        relevancies.append(answer_relevancy(generated, q))

    print(f"{strategy.__name__}: precision={mean(precisions):.3f} "
          f"recall={mean(recalls):.3f} "
          f"faithfulness={mean(faithfulnesses):.3f} "
          f"relevancy={mean(relevancies):.3f} "
          f"latency={mean_latency:.1f}ms")
```

### Phase 6 — Flask API
```
POST /ingest          → chunk + embed docs/
POST /query           → {"query": "...", "strategy": "hybrid", "top_k": 5}
                        → {"answer": "...", "sources": [...], "latency_ms": ...}
GET  /collection      → {"n_chunks": ..., "sources": [...]}
DELETE /collection    → clear doc_store + embeddings
```

---

## Checkpoints

| Phase | What correct output looks like |
|-------|-------------------------------|
| 1 | 200–500 chunks; each chunk < 512 tokens; overlap evident in adjacent chunks |
| 2 | `embeddings.shape == (n_chunks, 64)`; normalized (row norms ≈ 1.0) |
| 3 | Dense returns relevant chunks for "attention mechanism"; BM25 beats dense on exact keyword match |
| 4 | Ground-truth JSON file with 20 Q&A pairs + source chunk references |
| 5 | Hybrid RRF should score highest on Context Recall; MMR should score highest on diversity |
| 6 | `POST /query` responds in < 200ms on localhost |

---

## Extensions

1. **Cross-encoder reranker** — after retrieval, score top-20 candidates with a cross-encoder (question + chunk → relevance score); keep top-5; measure precision improvement.
2. **Query decomposition** — for multi-hop questions ("What is LoRA and how does it relate to QLoRA?"), decompose into 2 sub-queries, retrieve independently, merge results.
3. **Persistent ChromaDB** — replace in-memory doc_store + embeddings with a ChromaDB persistent collection; measure cold-start ingest time vs. in-memory.

---

## Hints

<details><summary>Hint 1 — Recursive splitting implementation</summary>
Try each separator in order. Split by first separator that produces chunks small enough. For too-large remaining pieces, recurse with the next separator. This is exactly LangChain's RecursiveCharacterTextSplitter logic.
</details>

<details><summary>Hint 2 — Context Precision from scratch</summary>
For each retrieved chunk, check if any ground-truth answer keyword appears in the chunk text. Precision = fraction of retrieved chunks that are "relevant." Recall = fraction of ground-truth keywords found across all retrieved chunks.
</details>

<details><summary>Hint 3 — Mock LLM for testing</summary>
<code>def mock_llm(q, context): return context[:200]</code> — just return the first 200 chars of context. This lets you test Faithfulness and Relevancy metrics without an actual LLM.
</details>

<details><summary>Hint 4 — MMR lambda tradeoff</summary>
λ=1.0 → pure relevance (same as dense). λ=0.0 → pure diversity (avoids redundancy entirely). λ=0.5 is the standard balance. Test both extremes to verify your implementation.
</details>

<details><summary>Hint 5 — Module guides are your corpus</summary>
The 9 docs/*.md files total ~300KB of dense ML content. "Who is the user?" (Harshit) or "What is the live site?" won't be in the Q&A; focus questions on module content: math formulas, algorithm names, code patterns, architectural choices.
</details>

---

*Back to [Module 08 — RAG Chatbot](../08-rag.md)*
