# Module 08 — RAG Chatbot

```bash
cd src/08-rag
python3.14 ingest.py
python3.14 embed_store.py
python3.14 retriever.py
python3.14 generator.py
python3.14 app.py
python3.14 evaluate.py
```

> **Run:** `python3.14 src/08-rag/ingest.py`
> **Run:** `python3.14 src/08-rag/embed_store.py`
> **Run:** `python3.14 src/08-rag/retriever.py`
> **Run:** `python3.14 src/08-rag/generator.py`
> **Run:** `python3.14 src/08-rag/app.py`
> **Run:** `python3.14 src/08-rag/evaluate.py`

---

## Prerequisites & Overview

**Prerequisites:** Module 03 (vector databases, cosine similarity, FAISS), Module 06 (embeddings, attention). Flask knowledge from Module 04 helps for the API section. No LLM API key needed — a mock LLM is built from scratch.
**Estimated time:** 10–14 hours (6 scripts, each covering a distinct pipeline stage)

### Why This Module Matters
RAG is the dominant production pattern for giving LLMs access to external knowledge without retraining. Every AI assistant that retrieves from documents, search results, or a knowledge base uses RAG. Understanding the full pipeline — chunking, embedding, retrieval strategies, generation, and evaluation — is essential for AI/ML backend roles.

### The RAG Pipeline (End-to-End)

```
Documents (PDF/TXT)
        ↓ ingest.py       — chunk into overlapping segments
        ↓ embed_store.py  — embed chunks → vector store
        ↓ retriever.py    — BM25 + dense + hybrid + MMR
        ↓ generator.py    — inject retrieved chunks into prompt → LLM
        ↓ evaluate.py     — Faithfulness, Relevancy, Precision, Recall
        ↓ app.py          — Flask API: /ingest, /query, /collection
```

### Before You Start
- Know what cosine similarity is (Module 01)
- Know how FAISS flat search works (Module 03)
- Know what TF-IDF is (Module 02 or Module 03)
- Understand what an LLM prompt is and what "context window" means

### When to Use RAG vs Fine-Tuning

| Criterion | RAG | Fine-Tuning |
|-----------|-----|-------------|
| Knowledge updates frequently | ✅ Update document store | ❌ Retrain required |
| Need source citations | ✅ Built-in via retrieved chunks | ❌ Model cannot cite |
| Domain-specific style/format | ❌ Doesn't change model style | ✅ Adapts generation style |
| Small labeled dataset | ✅ No training data needed | ❌ Needs labeled pairs |
| Latency-critical inference | ❌ Retrieval adds ~50–200ms | ✅ No retrieval overhead |

---

## 1. RAG Architecture Overview

Retrieval-Augmented Generation (RAG) separates *parametric knowledge* (LLM weights) from *non-parametric knowledge* (external corpus). At inference, relevant documents are retrieved and injected into the prompt so the LLM answers from evidence rather than memorised associations.

**Core pipeline:**

```
Query
  │
  ▼
[Embed query] ──► [Vector Store] ──► top-k chunks
                                          │
  ┌───────────────────────────────────────┘
  │
  ▼
[Prompt: system + context + query] ──► [LLM] ──► Answer
```

**Why RAG over full fine-tuning?**

| Criterion | RAG | Full Fine-Tune |
|-----------|-----|----------------|
| Knowledge update cost | Re-ingest documents | Full retraining run |
| Hallucination surface | Bounded by retrieved context | Unbounded (memorised noise) |
| Source attribution | Trivial (chunk metadata) | Impossible |
| Compute at inference | Embedding + retrieval + LLM | LLM only |
| Staleness | Near-real-time | Snapshot at training |

---

## 2. Document Ingestion & Chunking

### 2.1 Why Chunking Matters

LLMs have fixed context windows ($C$ tokens). A 100-page PDF contains $\sim$80 000 tokens — far beyond any practical context. Chunking splits documents into segments small enough to fit in the context while large enough to be semantically coherent.

**Chunk size tradeoff:**
- Too small → low information density per chunk; many chunks needed; retrieval noise
- Too large → semantic dilution; less precise retrieval; context window pressure

### 2.2 Fixed-Size Chunking

Split on token or character count with overlap $o$:

$$\text{chunks} = \bigl[\,d[i : i + w]\;\bigm|\; i = 0, w-o, 2(w-o), \ldots\bigr]$$

- $w$ = window size (e.g., 512 tokens)
- $o$ = overlap (e.g., 50 tokens) — preserves context across boundaries

### 2.3 Recursive Character Splitting

Priority-ordered separator list: `["\n\n", "\n", ". ", " ", ""]`

1. Try to split on `"\n\n"` (paragraph boundary)
2. If any piece > max\_chars, recurse with `"\n"`
3. Continue until all pieces ≤ max\_chars

This preserves paragraph → sentence → word hierarchy.

### 2.4 Semantic Chunking

Embed each sentence. Compute cosine similarity between adjacent sentence embeddings. Split where similarity drops below threshold $\tau$:

$$\text{split at } i \iff \cos(\mathbf{e}_i, \mathbf{e}_{i+1}) < \tau$$

Produces variable-length chunks aligned with topic shifts rather than character counts.

**Comparison table:**

| Strategy | Deterministic | Semantic Alignment | Complexity | Best For |
|----------|--------------|-------------------|-----------|---------|
| Fixed-size | ✓ | ✗ | $O(n)$ | Code, structured data |
| Recursive | ✓ | Partial | $O(n \log n)$ | General prose |
| Semantic | ✗ | ✓ | $O(n \cdot d)$ | Long-form documents |

---

## 3. Embedding & Vector Storage

### 3.1 Embedding Models

An embedding model $f_\theta : \text{text} \to \mathbb{R}^d$ maps variable-length text to a fixed-dimension dense vector. For retrieval, the key property is that semantically similar texts have high cosine similarity:

$$\text{sim}(q, d) = \frac{f_\theta(q)^\top f_\theta(d)}{\|f_\theta(q)\| \cdot \|f_\theta(d)\|}$$

**Common embedding models for RAG:**

| Model | Dim | Context | Notes |
|-------|-----|---------|-------|
| `all-MiniLM-L6-v2` | 384 | 256 tokens | Fast, small, good quality |
| `all-mpnet-base-v2` | 768 | 384 tokens | Higher quality, 5× slower |
| `text-embedding-3-small` | 1536 | 8192 tokens | OpenAI API |
| `text-embedding-3-large` | 3072 | 8192 tokens | OpenAI API, best quality |
| `bge-large-en-v1.5` | 1024 | 512 tokens | MTEB SOTA open-source |

### 3.2 ChromaDB Storage

ChromaDB stores documents with their embeddings and metadata in a persistent or in-memory store. At query time it runs ANN search (HNSW internally) and returns top-k chunks with distances.

**Document schema:**

```
Collection
  ├── id        : str (unique per document)
  ├── document  : str (raw text chunk)
  ├── embedding : List[float] (dim = embedding model output)
  └── metadata  : dict (source, page, chunk_index, ...)
```

---

## 4. Retrieval Strategies

### 4.1 Cosine Similarity (Dense)

Retrieve top-$k$ chunks maximising $\cos(f(q), f(d_i))$.

$$\text{top-}k = \underset{d_i \in \mathcal{D}}{\arg\text{top-}k}\; \cos(f(q), f(d_i))$$

Efficient via ANN (HNSW); handles semantic paraphrase and synonym matching well.

### 4.2 BM25 (Sparse)

BM25 is a probabilistic term-frequency ranking function:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

where:
- $f(t,d)$ = term frequency of $t$ in document $d$
- $\text{IDF}(t) = \log\frac{N - n(t) + 0.5}{n(t) + 0.5}$, $N$ = corpus size, $n(t)$ = docs containing $t$
- $k_1 \in [1.2, 2.0]$ controls TF saturation
- $b \in [0, 1]$ controls document length normalisation (typically $b = 0.75$)

BM25 excels at exact keyword matching, rare terms, and proper nouns — cases where dense retrieval fails.

### 4.3 Hybrid Retrieval (BM25 + Dense)

Fuse BM25 scores $s_{\text{BM25}}$ and dense scores $s_{\text{dense}}$ via reciprocal rank fusion (RRF) or linear interpolation:

**Linear interpolation:**
$$s_{\text{hybrid}}(d) = \alpha \cdot s_{\text{dense}}(d) + (1 - \alpha) \cdot s_{\text{BM25}}(d)$$

**Reciprocal Rank Fusion (RRF):**
$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + r(d)}$$

with $k = 60$ (standard value). RRF is robust: immune to score scale mismatch, no normalisation needed.

### 4.4 Maximum Marginal Relevance (MMR)

MMR trades off relevance against redundancy to select a diverse set of $k$ chunks:

$$\text{MMR}_i = \underset{d \in \mathcal{R} \setminus S}{\arg\max}\;\left[\lambda \cdot \text{sim}(q, d) - (1-\lambda) \cdot \max_{s \in S} \text{sim}(d, s)\right]$$

where:
- $\mathcal{R}$ = retrieved candidate set (top-$n$, $n \gg k$)
- $S$ = already-selected set
- $\lambda \in [0,1]$: $\lambda=1$ → pure relevance; $\lambda=0$ → pure diversity

**Algorithm:**
1. Retrieve top-$n$ candidates by cosine similarity
2. Greedily select: each step picks the candidate maximising the MMR objective
3. Repeat until $|S| = k$

| Strategy | Semantic Match | Exact Match | Diversity | Latency |
|----------|---------------|-------------|-----------|---------|
| Dense | ✓ | ✗ | Low | Fast |
| BM25 | ✗ | ✓ | Low | Fast |
| Hybrid | ✓ | ✓ | Low | Fast |
| MMR | ✓ | ✗ | High | O(n·k) |

---

## 5. Prompt Construction & Generation

### 5.1 Context Injection

The standard RAG prompt template:

```
System: You are a helpful assistant. Answer using only the provided context.
        If the answer is not in the context, say "I don't know."

Context:
[CHUNK 1]
Source: {source}, Page: {page}
{chunk_text}

[CHUNK 2]
Source: {source}, Page: {page}
{chunk_text}

...

Question: {user_query}

Answer:
```

**Key design decisions:**
- Place context before the question (LLMs attend better to recent tokens in long prompts)
- Include source metadata for attribution and verifiability
- Explicit "I don't know" instruction reduces hallucination
- Separate chunks with clear delimiters to prevent context blending

### 5.2 Context Window Budget

With a 4096-token context window:
- System prompt: ~100 tokens
- User query: ~50 tokens
- Each chunk: ~200 tokens
- Available for $k$ chunks: $\approx 4096 - 150 - k \cdot 200$

For $k=10$ chunks: $4096 - 150 - 2000 = 1946$ tokens remaining — sufficient for a detailed answer.

### 5.3 LLM API (Mock + Real)

Real calls use OpenAI-compatible endpoints:

```python
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ],
    temperature=0.1,   # low for factual RAG
    max_tokens=512,
)
answer = response.choices[0].message.content
```

Temperature 0.0–0.2 for factual retrieval-grounded answers; higher for creative summarisation.

---

## 6. End-to-End RAG Pipeline (Flask API)

### 6.1 API Design

```
POST /ingest          — load and chunk a document, embed, store
POST /query           — run full RAG pipeline, return answer + sources
GET  /health          — liveness check
DELETE /collection    — clear vector store
```

**Request/Response:**

```json
POST /query
{
  "query": "What is the attention mechanism?",
  "top_k": 5,
  "strategy": "hybrid",   // "dense" | "bm25" | "hybrid" | "mmr"
  "lambda_mmr": 0.7
}

Response:
{
  "answer": "The attention mechanism computes ...",
  "sources": [
    {"source": "attention_paper.pdf", "page": 3, "chunk_index": 12, "score": 0.92},
    ...
  ],
  "retrieval_ms": 14,
  "generation_ms": 820
}
```

### 6.2 Pipeline Flow

```python
def rag_query(query, top_k=5, strategy="hybrid"):
    # 1. Embed query
    q_emb = embed([query])[0]
    # 2. Retrieve chunks
    chunks = retriever.retrieve(q_emb, query, top_k, strategy)
    # 3. Build prompt
    context = format_context(chunks)
    prompt = build_prompt(context, query)
    # 4. Generate
    answer = llm.generate(prompt)
    return answer, chunks
```

---

## 7. Evaluation with RAGAS

RAGAS evaluates RAG pipelines across four dimensions without ground-truth labels (except Context Recall):

### 7.1 Faithfulness

Measures whether each claim in the answer is supported by the retrieved context.

$$\text{Faithfulness} = \frac{\text{# claims in answer supported by context}}{\text{# claims in answer}}$$

**Process:**
1. LLM decomposes the answer into atomic claims
2. For each claim, LLM checks if it can be inferred from the context
3. Score = supported / total

### 7.2 Answer Relevancy

Measures whether the answer addresses the question (ignoring faithfulness).

$$\text{AnswerRelevancy} = \frac{1}{n}\sum_{i=1}^{n} \cos(f(q_{\text{gen},i}), f(q_{\text{orig}}))$$

LLM generates $n$ hypothetical questions from the answer, then computes cosine similarity of their embeddings to the original question. High score → answer is on-topic.

### 7.3 Context Precision

Measures what fraction of the retrieved context chunks are actually relevant.

$$\text{ContextPrecision} = \frac{\text{# relevant retrieved chunks}}{\text{# retrieved chunks}}$$

### 7.4 Context Recall

Measures what fraction of ground-truth answer claims can be attributed to the retrieved context.

$$\text{ContextRecall} = \frac{\text{# GT claims supported by context}}{\text{# GT claims}}$$

Requires ground-truth answers.

### 7.5 From-Scratch Approximations

Without an LLM evaluator, approximate each metric:

| Metric | Approximation |
|--------|--------------|
| Faithfulness | Check if answer tokens appear in context (token overlap F1) |
| Answer Relevancy | Embed question + answer, compute cosine similarity |
| Context Precision | BM25 score of retrieved chunks against query |
| Context Recall | Answer token overlap with context union |

---

## 8. Interview Q&A

**Q: What is the difference between RAG and fine-tuning for domain adaptation?**
Fine-tuning bakes knowledge into weights — expensive to update, can catastrophically forget. RAG externalises knowledge — update the document store without touching model weights. RAG provides source attribution; fine-tuned models cannot cite evidence. Use RAG for frequently-updated knowledge bases; use fine-tuning for style/format adaptation or domain-specific reasoning patterns.

**Q: Why does chunk size matter for retrieval quality?**
Small chunks (< 100 tokens) have high precision but low recall — each chunk is highly specific but a single chunk may not contain enough context to answer a question. Large chunks (> 1000 tokens) have high recall but low precision — the chunk contains the answer but also a lot of noise that dilutes the embedding signal. Empirically, 256–512 tokens with 50-token overlap is a strong default.

**Q: Explain the MMR algorithm and when to use it.**
MMR iteratively selects chunks that maximise relevance to the query while minimising redundancy with already-selected chunks. The $\lambda$ parameter interpolates. Use MMR when the top-$k$ dense results are near-duplicate paragraphs from the same source — common in repetitive documents or large corpora with many similar pages. It trades latency ($O(nk)$ similarity comparisons) for diversity.

**Q: What is the difference between BM25 and dense retrieval? When does each fail?**
BM25 is a sparse TF-IDF variant that counts exact term occurrences — fails on synonyms and paraphrases ("automobile" vs "car"). Dense retrieval embeds semantic meaning — fails on exact rare terms, proper nouns, and entity-heavy queries where the model hasn't seen the term. Hybrid search combines both: BM25 for lexical precision, dense for semantic recall.

**Q: How do you measure RAG quality without labeled data?**
Use LLM-as-judge metrics: Faithfulness (does the answer contradict the context?) and Answer Relevancy (does the answer address the question?) can both be computed without labels. Context Precision requires only the retrieved chunks and query, no ground truth. Only Context Recall strictly requires labeled reference answers.

**Q: What is the context window budget problem and how do you solve it?**
Each retrieved chunk consumes prompt tokens. With $k$ chunks of 512 tokens and a 4096-token window, you have $\approx 1000$ tokens for the question and answer. Solutions: (1) reduce chunk size, (2) reduce $k$, (3) use a reranker to filter the top-$m$ chunks from a larger candidate set (retrieve 20, rerank, keep 5), (4) use a model with a larger context window.

---

## Resources

### Papers (Essential)
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** — Lewis et al. (2020): `arxiv.org/abs/2005.11401`. The original RAG paper from Facebook AI.
- **REALM: Retrieval-Augmented Language Model Pre-Training** — Guu et al. (2020): `arxiv.org/abs/2002.08909`. Pre-training with retrieval.
- **Improving Language Models by Retrieving from Trillions of Tokens (RETRO)** — Borgeaud et al. (2021): `arxiv.org/abs/2112.04426`. Chunked cross-attention for trillion-token retrieval.
- **RAGAS: Automated Evaluation of Retrieval Augmented Generation** — Es et al. (2023): `arxiv.org/abs/2309.15217`. The evaluation framework implemented from scratch in `evaluate.py`.

### Chunking & Retrieval
- **BM25 Okapi** — Robertson & Zaragoza (2009): "The Probabilistic Relevance Framework: BM25 and Beyond". Foundation for the BM25 retriever in this module.
- **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction** — `arxiv.org/abs/2004.12832`. Advanced dense retrieval that outperforms single-vector embedding.

### Frameworks
- **LlamaIndex** (`llamaindex.ai/open-source`): Production RAG framework. After this module, its abstractions (Node, Retriever, QueryEngine) will make sense.
- **LangChain** (`python.langchain.com/docs`): Alternative framework with rich retriever ecosystem. `RecursiveCharacterTextSplitter` matches `ingest.py`'s recursive strategy.
- **Haystack** (`haystack.deepset.ai`): Enterprise RAG framework with built-in evaluation and pipelines.

### Video
- **"Building RAG from Scratch"** — various Andrej Karpathy/LlamaIndex tutorials on YouTube: Search "RAG from scratch tutorial" for multiple good walkthroughs.
- **DeepLearning.AI — "Building and Evaluating Advanced RAG"** (short course, free): Covers reranking, hybrid search, and evaluation — direct extensions of this module.

---

*Next: [Module 09 — Fine-Tuning (LoRA/QLoRA)](09-finetuning.md)*
