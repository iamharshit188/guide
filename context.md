# PROJECT CONTEXT — AI/ML Learning Platform
> Read this file at the start of every session before writing any code or guide.
> This is the single source of truth for tone, style, structure, and progress.

---

## WHO THIS IS FOR

- **User:** Harshit Tiwari — building this for placement preparation (AI/ML/GenAI/Backend roles).
- **Goal:** A local, interactive, self-paced learning platform covering the full AI/ML stack — from math foundations to fine-tuning open-source LLMs.
- **Platform runs at:** `python server.py` → `http://localhost:3000`

---

## REPOSITORY LOCATION

```
/Users/iamharshit188/Desktop/placement/prepration/guide/
```

Git repo: initialized, branch `main`. All commits signed by `Harshit Tiwari`.

---

## MONOREPO STRUCTURE

> **NOTE:** After the GitHub Pages migration (commit `09a9613`), frontend files moved to **repo root** (not `frontend/`).

```
guide/                          ← repo root = GitHub Pages root
├── context.md                  ← THIS FILE (read first every session)
├── index.html                  ← Neo-Brutalism UI (marked.js + highlight.js + MathJax)
├── style.css
├── app.js                      ← localStorage progress, module nav, markdown renderer
├── server.py                   ← Flask dev server for LOCAL use only (port 3000)
├── .nojekyll                   ← prevents GitHub Pages Jekyll processing
├── .gitignore
├── docs/
│   ├── list.md                 ← MASTER CURRICULUM ROADMAP
│   ├── 01-math.md              ← [COMPLETE] — includes Prerequisites + Resources sections
│   ├── 02-ml-basics.md         ← [COMPLETE] — includes Prerequisites + Resources sections
│   ├── 03-databases.md         ← [COMPLETE] — includes Prerequisites + Resources sections
│   ├── 04-backend.md           ← [COMPLETE] — includes Prerequisites + Resources sections
│   ├── 05-deep-learning.md     ← [COMPLETE] — includes Prerequisites + Resources sections
│   ├── 06-genai-core.md        ← [COMPLETE] — includes Prerequisites + Resources sections
│   ├── 07-transformers.md      ← [COMPLETE] — includes Prerequisites + Resources sections
│   ├── 08-rag.md               ← [COMPLETE] — includes Prerequisites + Resources sections
│   ├── 09-finetuning.md        ← [COMPLETE] — includes Prerequisites + Resources sections
│   └── projects/               ← [TODO] one .md per module project (see TODO section)
│       ├── p01-pca-compressor.md
│       ├── p02-titanic-pipeline.md
│       ├── p03-semantic-search.md
│       ├── p04-ml-api.md
│       ├── p05-training-dashboard.md
│       ├── p06-word-analogy.md
│       ├── p07-gpt-shakespeare.md
│       ├── p08-document-qa.md
│       └── p09-domain-tuner.md
└── src/
    ├── 01-math/                ← vectors.py, matrix_ops.py, calculus_demo.py, probability.py
    ├── 02-ml/                  ← linear_regression.py, logistic_regression.py, evaluation.py,
    │                              clustering.py, pca.py, random_forest.py, svm.py,
    │                              gradient_boosting.py
    ├── 03-databases/           ← sql_basics.py, nosql_patterns.py, chroma_demo.py,
    │                              pinecone_demo.py, faiss_demo.py
    ├── 04-backend/             ← app.py, ml_serving.py, middleware.py, async_tasks.py
    ├── 05-deep-learning/       ← nn_numpy.py, optimizers.py, mlflow_demo.py, monitoring.py
    ├── 06-genai/               ← word2vec.py, attention.py, multihead_attention.py,
    │                              positional_encoding.py, kv_cache.py
    ├── 07-transformer/         ← tokenizer.py, model.py, model_numpy.py, model.cpp, train.py
    ├── 08-rag/                 ← ingest.py, embed_store.py, retriever.py, generator.py,
    │                              app.py, evaluate.py
    └── 09-finetuning/          ← lora_theory.py, prepare_dataset.py, train_lora.py,
                                   train_qlora.py, evaluate.py, merge_push.py
```

---

## CURRICULUM PROGRESS

| # | Module | Status | Guide | Code |
|---|--------|--------|-------|------|
| 01 | Math for ML | ✅ COMPLETE | `docs/01-math.md` | `src/01-math/` |
| 02 | ML Basics to Advanced | ✅ COMPLETE | `docs/02-ml-basics.md` | `src/02-ml/` |
| 03 | Databases & Vector DBs | ✅ COMPLETE | `docs/03-databases.md` | `src/03-databases/` |
| 04 | Backend with Flask | ✅ COMPLETE | `docs/04-backend.md` | `src/04-backend/` |
| 05 | Deep Learning & MLOps | ✅ COMPLETE | `docs/05-deep-learning.md` | `src/05-deep-learning/` |
| 06 | GenAI Core | ✅ COMPLETE | `docs/06-genai-core.md` | `src/06-genai/` |
| 07 | Transformers from Scratch | ✅ COMPLETE | `docs/07-transformers.md` | `src/07-transformer/` |
| 08 | RAG Chatbot | ✅ COMPLETE | `docs/08-rag.md` | `src/08-rag/` |
| 09 | Fine-Tuning (LoRA/QLoRA) | ✅ COMPLETE | `docs/09-finetuning.md` | `src/09-finetuning/` |

**All 9 modules complete.**

---

## WHAT EACH MODULE DELIVERS

Every module produces exactly two things:

### 1. `/docs/NN-topic.md`
- Dense, technical Markdown guide — NO fluff, NO generic intros
- Full mathematical derivations with LaTeX (`$...$` inline, `$$...$$` block) rendered via MathJax in the frontend
- Reference tables, algorithm pseudocode, comparison tables
- **Explicit runnable references** at top: `> Run: python src/NN-topic/script.py`
- Ends with a "Next module" link
- Opening codeblock shows all `cd src/NN-topic && python X.py` commands

### 2. `/src/NN-topic/*.py` (and `.cpp` where applicable)
- **Fully runnable, self-contained** scripts — no missing setup, no TODOs
- From-scratch implementations first (numpy/pure Python), then sklearn/library comparison
- Educational `print()` output labelled with section headers using `='*'60` separators
- `section("TITLE")` helper function in every file
- Dependencies handled gracefully: `try: import X; except ImportError: print("pip install X")`
- No `matplotlib.show()` — output is always terminal-printable (no GUI required)

---

## TONE & WRITING STYLE

- **Zero fluff.** No "Great question!", no "In this section we will explore...". Start with the content.
- **Dense and technical.** Every sentence carries information. Treat the reader as someone preparing for a senior ML engineering interview.
- **Math-first.** Derive before implementing. Show the formula, then show its code equivalent.
- **Tables over prose** for comparisons (algorithms, hyperparameters, trade-offs).
- **Blockquotes** for runnable references: `> **Run:** python src/...`
- Markdown headers follow this hierarchy: `#` module title, `##` major section, `###` subsection, `####` sub-subsection.

---

## CODE STYLE RULES

1. **No AI attribution** — no "Generated by Claude", no Anthropic references, no comments attributing authorship.
2. **No file-header docstrings claiming authorship.**
3. **Module-level docstring** (triple-quoted, 3–5 lines): states what the file covers and its `pip install` requirements. Nothing else.
4. **No `matplotlib.show()`** — all output is printed to terminal. Plots are described numerically.
5. **`section()` helper** — every file has:
   ```python
   def section(title):
       print(f"\n{'='*60}")
       print(f"  {title}")
       print('='*60)
   ```
6. **From-scratch first, library second** — implement the algorithm in numpy/pure Python, then validate against the sklearn/library equivalent. Show they match.
7. **Graceful import guards:**
   ```python
   try:
       import chromadb
       CHROMA_AVAILABLE = True
   except ImportError:
       CHROMA_AVAILABLE = False
       print("pip install chromadb")
   ```
8. **Fixed random seeds** — `np.random.default_rng(42)` or `random.Random(42)`. Results must be reproducible.
9. **No hardcoded file paths** — use `os.path` or `tempfile` for any file I/O.
10. **`if __name__ == "__main__": main()`** — every file ends this way.

---

## GIT COMMIT PROTOCOL

After every module (doc + all src files + list.md update), commit with:

```
git add docs/NN-topic.md docs/list.md src/NN-topic/
git commit -m "feat: complete <topic> module (NN)"
```

Commit message body (HEREDOC) must include:
- What the doc covers (topics in the markdown)
- What each script covers (one line per file)
- list.md status change (e.g., "Module 07 → completed, Module 08 → in progress")

**Never use `--amend`.** Always new commits.
**Never `--no-verify`.**
**Never commit `.env` or `*.pkl` or `*.pt` files** (covered by `.gitignore`).

---

## list.md UPDATE PROTOCOL

Every time a module is completed:
1. Change completed module status: `` `[~]` In Progress `` → `` `[x]` Completed ``
2. Change next module status: `` `[ ]` Not Started `` → `` `[~]` In Progress ``
3. Update the last line: `*Last updated: Module NN complete — <Topic>*`

---

## FRONTEND ARCHITECTURE

### Production (GitHub Pages)
- Deployed at `https://harshittiwari.me/guide/` from `main` branch root
- **Static only** — no server, no API. Frontend fetches markdown directly from the `docs/` folder via relative URLs.
- Fetch URL pattern in `app.js`: `fetch('docs/${file}')` → resolves to `https://harshittiwari.me/guide/docs/02-ml-basics.md`
- `.nojekyll` in root prevents GitHub Pages from processing with Jekyll

### Local Dev
- `python server.py` → `http://localhost:3000`
- Flask serves `index.html` from root, `docs/*.md` as `text/plain`

### Frontend Files (all at repo root after GitHub Pages migration)
- `index.html`, `style.css`, `app.js`
  - Design: **Neo-Brutalism** — `border: 3px solid black`, `box-shadow: 4px 4px 0 black`, cream `#FFFDE7`, yellow `#FFD600`
  - marked.js: Markdown → HTML
  - highlight.js: syntax highlighting (Python, Bash, C++, SQL)
  - MathJax: LaTeX rendering (`$...$` and `$$...$$`)
  - localStorage: progress tracking (key: `aiml_platform_progress`)
  - "Mark Complete" button: `Cmd+Enter` shortcut
  - Progress bar in sidebar
  - `.md` link intercept in `renderMarkdown()`: clicks on relative markdown links (e.g. "Next module") call `openModule()` instead of navigating — **prevents GitHub Pages 404s**

---

## COMPLETED MODULES — DETAILED SUMMARIES

### Module 01 — Math for ML (`docs/01-math.md`, `src/01-math/`)
- **Linear Algebra:** Scalars/vectors/matrices/tensors, L1/L2/L∞ norms, dot product, cosine similarity, projection, matrix multiply, transpose, inverse, determinant, eigendecomposition ($A = V\Lambda V^{-1}$), SVD ($A = U\Sigma V^T$), pseudoinverse, PCA via eigendecomp
- **Calculus:** Partial derivatives, gradient vector, chain rule (crucial for backprop), Jacobian, Hessian, gradient descent update rule, numerical gradient check (central difference), LR sensitivity
- **Probability:** Bayes theorem, conditional probability, Gaussian/Bernoulli/Categorical/Poisson distributions, MLE derivation, cross-entropy, KL divergence, Shannon entropy, covariance matrix, Mahalanobis distance
- **Scripts:** `vectors.py`, `matrix_ops.py`, `calculus_demo.py`, `probability.py`

### Module 02 — ML Basics to Advanced (`docs/02-ml-basics.md`, `src/02-ml/`)
- **Linear Regression:** OLS normal equation derivation, GD, Ridge closed-form, Lasso coordinate descent + soft-thresholding, multicollinearity + condition number
- **Logistic Regression:** BCE from MLE, gradient form, softmax for multiclass, threshold analysis
- **Evaluation:** Bias-variance decomposition (bootstrap), K-fold CV from scratch, ROC/PR curves from scratch, confusion matrix, F-β, calibration, learning curves, imbalanced classes
- **Clustering:** K-Means++ from scratch (Lloyd's), elbow method, silhouette from scratch, DBSCAN outlier detection
- **PCA:** Full eigendecomp from scratch, SVD equivalence proof, EVR, reconstruction error, digits dataset
- **Random Forest:** Gini/entropy from scratch, minimal Decision Tree + RF from scratch, MDI + permutation importance, OOB error convergence
- **SVM:** Kernel PSD check (Mercer), linear SVM subgradient descent from scratch, kernel comparison, hinge vs log-loss
- **Gradient Boosting:** GBR/GBC from scratch (residual fitting), staged prediction, early stopping, XGBoost 2nd-order Taylor math
- **Scripts:** `linear_regression.py`, `logistic_regression.py`, `evaluation.py`, `clustering.py`, `pca.py`, `random_forest.py`, `svm.py`, `gradient_boosting.py`

### Module 03 — Databases & Vector DBs (`docs/03-databases.md`, `src/03-databases/`)
- **SQL:** B+ tree index internals, all join types + algorithms, composite index left-prefix rule, EXPLAIN QUERY PLAN, window functions (RANK/LAG/running total), CTEs, recursive CTEs, correlated subqueries
- **NoSQL:** CAP theorem, PACELC, 5 NoSQL types with ML use cases, MongoDB aggregation pipeline ($match/$group/$project/$sort/$unwind)
- **ANN Math:** Cosine/L2/IP metric relationships for normalized vectors
- **HNSW:** Multi-layer graph build algorithm, greedy descent query, M/efConstruction/efSearch parameters
- **IVF:** Voronoi cells, K-Means training, nprobe recall/speed tradeoff
- **PQ/IVFPQ:** Sub-quantizer compression, ADC, billion-scale use case
- **ChromaDB:** in-memory client, all metadata operators ($eq/$gte/$in/$and/$or/where_document), get/update/upsert/delete, distance metrics, batch benchmark
- **Pinecone:** Full API mock (FAISS-backed) — upsert/query/fetch/update/delete, namespaces, hybrid sparse+dense search (alpha parameter), metadata filtering
- **FAISS:** FlatL2/FlatIP → IVFFlat → HNSWFlat → IndexPQ → IndexIVFPQ — full recall@10 vs QPS sweep, IndexIDMap, serialization, selection guide
- **Scripts:** `sql_basics.py` (sqlite3, zero-install), `nosql_patterns.py` (pure Python, zero-install), `chroma_demo.py`, `pinecone_demo.py` (mock mode, zero-install), `faiss_demo.py`

### Module 04 — Backend with Flask (`docs/04-backend.md`, `src/04-backend/`)
- **Flask Internals:** WSGI interface, request context lifecycle, thread-local `g` / `request` / `current_app`, `before_request` / `after_request` / `teardown_request` execution order
- **App Factory:** `create_app(env)` pattern, `Config` class hierarchy (Base/Dev/Test/Prod), `from_object`, multi-environment isolation for testing
- **Blueprints:** URL-scoped routing, per-blueprint error handlers and `before_request`, versioned APIs (`/api/v1/` and `/api/v2/`)
- **REST Design:** Status code table (200/201/204/400/401/403/404/409/422/429/500/503), consistent error response schema, URL-path versioning
- **ML Serving:** Model registry (thread-safe with `RLock`), startup model loading, sklearn Pipeline pickle, PyTorch `state_dict` + TorchScript serialization, single + batch `/predict` endpoints, input validation (NaN/Inf/type checks), throughput benchmarking
- **Middleware:** `hmac.compare_digest` timing-safe auth, Token Bucket rate limiting (capacity/rate/thread-safe), per-client `PerClientRateLimiter`, structured JSON request logging, `X-Request-ID` propagation
- **Async Tasks:** Celery config (`ContextTask`, `task_acks_late`, `worker_prefetch_multiplier=1`), task state machine (PENDING→STARTED→SUCCESS/FAILURE/RETRY), `bind=True` retry pattern, 202/poll pattern, worker pool comparison (prefork/gevent/solo), in-process simulation (no Redis needed for demo)
- **Scripts:** `app.py` (factory, blueprints, test client demo), `ml_serving.py` (registry, sklearn + torch endpoints, serialization), `middleware.py` (auth, rate limit, logging demos), `async_tasks.py` (simulated broker, task types, Flask poll API)

### Module 05 — Deep Learning & MLOps (`docs/05-deep-learning.md`, `src/05-deep-learning/`)
- **Backpropagation:** Full derivation — $\delta^{(L)} = \hat{y} - y$ (BCE+sigmoid cancel), hidden $\delta^{(l)} = (W^{(l+1)T}\delta^{(l+1)}) \odot g'$, parameter gradients $\nabla W = \delta \mathbf{a}^T / m$
- **Initialisation:** Zero symmetry failure, vanishing/exploding gradient causes, Xavier derivation ($\text{Var}[w] = 2/(n_{in}+n_{out})$), He ($2/n_{in}$), empirical stability comparison over 10 layers
- **Activations:** ReLU/Sigmoid/Tanh/Softmax with gradients; numerically stable sigmoid (no overflow) and softmax implementations
- **Optimizers:** SGD, Momentum (heavy ball), Nesterov, RMSProp (per-parameter adaptive LR), Adam (1st+2nd moment + bias correction derivation), AdamW (decoupled weight decay); convergence benchmark on ill-conditioned quadratic + Rosenbrock
- **LR Scheduling:** StepDecay, CosineAnnealing, WarmupCosine (transformer schedule), ReduceOnPlateau
- **Regularisation:** Dropout (inverted dropout, inference mode), BatchNorm (batch-wise normalisation, running stats, $\gamma/\beta$ learnable), LayerNorm (sample-wise, no batch dependency — used in transformers)
- **MLflow:** Experiment/Run/Param/Metric/Artifact/Model hierarchy; `log_params`, `log_metric(step=)`, `log_artifact`, `log_model`; autolog; model registry lifecycle (None→Staging→Production→Archived); local filesystem backend (zero server, temp dir)
- **Drift Detection:** KS test (empirical CDF, Kolmogorov distribution p-value), Chi-squared (categorical), PSI ($\sum(A-E)\ln(A/E)$, rule of thumb: >0.2 = major), MMD (RBF kernel, unbiased estimator), JS divergence; `DriftDetector` class; prediction drift as concept drift proxy; monitoring pipeline design
- **Scripts:** `nn_numpy.py` (TwoLayerNN + DeepNN + numerical gradient check + mini-batch training loop), `optimizers.py` (6 optimizers + 4 schedulers + convergence benchmarks), `mlflow_demo.py` (4-model experiment + registry + autolog + artifact logging), `monitoring.py` (all 5 drift tests + DriftDetector + prediction drift + scipy validation)

### Module 06 — GenAI Core (`docs/06-genai-core.md`, `src/06-genai/`)
- **Word2Vec skip-gram:** Skip-gram objective, negative sampling loss/gradients ($\mathcal{L}_{\text{NEG}}$), noise distribution $P_n \propto f^{3/4}$, numerical gradient check, cosine similarity nearest neighbours, analogy via $\mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c$, FastText subword $n$-gram decomposition
- **Sentence Transformers:** Mean-pool SBERT, cosine similarity = L2 distance for unit-normalised vectors, use cases (semantic search, clustering, deduplication, cross-lingual)
- **Scaled dot-product attention:** $\text{Attention}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d_k})V$, variance scaling derivation, numerically stable softmax, causal mask ($-\infty$ additive), cross-attention, $O(n^2 d)$ complexity table
- **Multi-head attention:** $W^Q, W^K, W^V, W^O$ projection matrices, head split/concat, GPT-2 parameter count worked example, MQA (shared K/V, $h\times$ cache reduction), GQA ($h/g\times$ reduction, used in LLaMA 2/3)
- **Positional encoding:** Sinusoidal formula (sin/cos at $\omega_i = 1/10000^{2i/d}$), frequency spectrum, relative position property ($PE_{\text{pos}}^\top PE_{\text{pos}+k} \approx f(k)$), learned PE table, RoPE 2D rotation demo (relative offset preserved in dot product)
- **KV cache:** Autoregressive generation loop with/without cache, FLOP comparison ($O(T^2 d)$ vs $O(Td)$ projection), memory formula $2 \times B \times T \times L \times h \times d_k \times 2$ bytes, GPT-3 ≈ 9.66 GB at $T=2048$, MQA/GQA/quantisation reduction table, PagedAttention virtual memory analogy
- **Scripts:** `word2vec.py` (skip-gram NS + gradient check + analogy), `attention.py` (step-by-step + causal + cross), `multihead_attention.py` (MHA + MQA + GQA classes), `positional_encoding.py` (sinusoidal + learned + RoPE), `kv_cache.py` (timing benchmark + memory math + cache growth chart)

### Module 07 — Transformers from Scratch (`docs/07-transformers.md`, `src/07-transformer/`)
- **BPE Tokenizer:** Corpus→word→char decomposition, pair counting, merge loop, encode/decode with stored rules, OOV via char fallback, step-by-step trace on `['low', 'lower', 'lowest']`, Tiktoken/SentencePiece comparison
- **Architecture:** Encoder (Pre-LN + MHA + FFN) × $N_e$, Decoder (Masked-SA + Cross-Attn + FFN) × $N_d$, LayerNorm formula, Pre-LN vs Post-LN table, FFN ($d_{\text{ff}} = 4d$, ReLU/SwiGLU variants), residual connections
- **Parameter count:** $12Ld^2$ per block derivation; GPT-2 small worked example — 85M transformer + 38.6M embed = 117M ✓
- **Training:** Teacher forcing + exposure bias, transformer LR schedule ($d^{-0.5} \min(t^{-0.5}, t \cdot t_w^{-1.5})$), gradient clipping (global $\ell_2$ norm ≤ 1.0), label smoothing ($\epsilon=0.1$)
- **Inference:** Greedy, beam search (length penalty $\alpha$), temperature/top-k/top-p table
- **Variants table:** BERT (encoder, MLM), GPT (decoder, causal LM), T5 (enc-dec), LLaMA (RoPE + SwiGLU + GQA + RMSNorm), Mistral (GQA + sliding window)
- **Scripts:** `tokenizer.py` (BPE from scratch, merge trace, round-trip), `model.py` (PyTorch enc-dec, Pre-LN, Xavier init, greedy decode — graceful skip without torch), `model_numpy.py` (pure NumPy encoder, PyTorch equivalence check < 1e-4), `model.cpp` (C++17 no deps, matmul/MHA/LN/FFN/causal mask, compiles with `g++ -O2 -std=c++17`), `train.py` (seq-reversal task, transformer LR schedule, label smoothing, gradient clipping, beam search, checkpoint save/load)

### Module 09 — Fine-Tuning (LoRA/QLoRA) (`docs/09-finetuning.md`, `src/09-finetuning/`)
- **Full FT vs PEFT:** Parameter count comparison (full $mn$ vs LoRA $r(m+n)$), memory comparison table across FP32/BF16/LoRA/QLoRA for LLaMA-2-7B
- **LoRA:** $W' = W_0 + \frac{\alpha}{r}BA$, $B=0$ init so $\Delta W=0$ at start, $\alpha/r$ scaling decouples rank from effective LR, rank sensitivity sweep, layer-wise param savings (LLaMA-2-7B q/v projections), adapter merging ($W_{\text{merged}} = W_0 + \frac{\alpha}{r}BA$, lossless, zero inference overhead), SVD verification that $\Delta W$ has rank $= r$
- **QLoRA:** NF4 construction (16 quantiles of $\mathcal{N}(0,1)$, renorm to $[-1,1]$, denser near mode → less quantisation error for Gaussian weights), INT4 vs NF4 RMSE comparison, double quantisation (8-bit absmax → $0.127$ bits/weight overhead), paged Adam (UVM offload), full memory breakdown (QLoRA ≈ 8 GB for 7B)
- **Dataset prep:** Alpaca/ChatML/LLaMA-3 templates, label masking ($-100$ for instruction tokens), data collator, train/val split, tokeniser simulation
- **Training loop:** Numpy gradient flow simulation (A, B only; W0 frozen), Adam with bias correction, dropout, hyperparameter sweep (r/alpha/lr sensitivity), PEFT SFTTrainer config (graceful skip)
- **Evaluation:** Perplexity ($\exp(\text{mean CE})$), BLEU-4 (modified n-gram precision + brevity penalty), ROUGE-1/2/L (unigram/bigram/LCS F1), corpus BLEU, before/after FT metric comparison, per-example breakdown
- **Merge & push:** Weight merge math, multi-adapter composition (weighted sum), adapter inspection (config JSON, param count), PEFT merge_and_unload simulation, Hub push commands, inference overhead comparison
- **Scripts:** `lora_theory.py` (pure numpy, fully runnable), `prepare_dataset.py` (pure stdlib+numpy), `train_lora.py` (numpy gradient sim + PEFT graceful skip), `train_qlora.py` (NF4 from scratch + BitsAndBytes graceful skip), `evaluate.py` (pure numpy/stdlib PPL+BLEU+ROUGE), `merge_push.py` (numpy merge + PEFT graceful skip)

### Module 08 — RAG Chatbot (`docs/08-rag.md`, `src/08-rag/`)
- **Architecture:** RAG pipeline (retrieve → inject → generate), parametric vs non-parametric knowledge, RAG vs fine-tuning comparison table
- **Chunking:** Fixed-size (sliding window, $w=512$, $o=50$), recursive character splitting (paragraph→sentence→word fallback), semantic chunking (cosine similarity between adjacent sentence embeddings, split at $\text{sim} < \tau$)
- **Embedding:** TF-IDF from scratch (vocab fit, IDF = $\log((N+1)/(df+1))+1$, random projection to dense), ChromaDB ingestion (graceful skip), batch upsert, cosine similarity matrix
- **Retrieval:** Dense (brute-force cosine), BM25 from scratch ($k_1=1.5$, $b=0.75$, IDF Robertson variant), hybrid RRF ($\text{RRF}(d) = \sum 1/(k+r)$, $k=60$), hybrid linear ($\alpha$ interpolation with min-max normalisation), MMR (greedy $\lambda$-blend of relevance and anti-redundancy)
- **Generation:** Context window budget analysis (~4 chars/token), prompt template (context-first, explicit "I don't know"), mock LLM (token overlap sentence extraction), OpenAI-compatible wrapper (graceful skip), citation formatting with source metadata
- **Flask API:** `POST /ingest`, `POST /query` (strategy parameter), `GET /collection`, `DELETE /collection`; strategy selector (dense/bm25/hybrid/mmr); in-process pipeline state; `--serve` flag to start server
- **Evaluation:** Faithfulness (sentence-level recall > 0.2 threshold), Answer Relevancy (embedding cosine similarity), Context Precision (BM25 overlap + Average Precision), Context Recall (GT sentence support), RAGEvaluator aggregate over test sets, metric comparison table (which need GT labels)
- **Scripts:** `ingest.py` (3 chunking strategies + TXT/PDF loader + stats + strategy comparison), `embed_store.py` (TFIDFEmbedder + InMemoryVectorStore + ChromaStore + batch benchmark), `retriever.py` (BM25 + DenseRetriever + HybridRetriever + MMR + latency benchmark), `generator.py` (prompt templates + budget analysis + MockLLM + OpenAI wrapper + citation formatter), `app.py` (full pipeline + Flask routes + CLI demo + latency benchmark), `evaluate.py` (all 4 RAGAS metrics from scratch + RAGEvaluator + token F1 vs embedding sim comparison)

---

## UPCOMING MODULES

All 9 core modules are complete. See TODO section below for what to build next.

---

## TODO — PLANNED WORK

> Priority order: Projects section first (highest user value), then additional modules, then content polish.

---

### TODO 1 — Projects Section (UI + Content) `[ ]`

**What it is:** A new "PROJECTS" view in the app — one project per module, each opening a dedicated guided page. Projects give **approach and checkpoints, not full code** — they bridge the gap between reading the module and building something real.

#### Frontend Changes Required
- Add "PROJECTS" button to the sidebar (between MODULES list and ROADMAP button)
- New view in `app.js`: `openProject(file, label)` fetches from `docs/projects/`
- Projects grid on the welcome screen: 9 cards, one per module, styled like module cards
- `MODULE_META` gets a `project` field: `{ file: "p01-pca-compressor.md", label: "PCA Image Compressor" }`
- Fetch path: `docs/projects/${file}`

#### Project Content Format (for each `docs/projects/pNN-name.md`)
Every project file must follow this structure:
1. **Project title + difficulty badge** (Beginner / Intermediate / Advanced)
2. **What you'll build** — 2–3 sentence description of the end product
3. **Skills exercised** — bullet list linking back to the module
4. **Approach** — numbered phases with what to implement at each step (NO full code; pseudocode or code skeletons only for non-obvious parts)
5. **Checkpoints** — what correct output looks like at each phase so the learner can self-verify
6. **Extensions** — 2–3 harder variants to try after completing the base project
7. **Hints** — 3–5 targeted hints for the hardest parts (spoiler-style, not full solutions)

#### The 9 Projects

---

**P01 — PCA Image Compressor** `[Module 01 — Math for ML]` `Beginner`
- **What:** Compress grayscale images using PCA from scratch. Show reconstruction error vs. number of retained principal components. Print compression ratio and RMSE at ranks 5, 10, 20, 50.
- **Why it's good:** Forces full eigendecomposition pipeline — center data, compute covariance, eigendecomp, project, reconstruct. No sklearn allowed in the core implementation.
- **Phases:**
  1. Load any grayscale image as a NumPy matrix (use `PIL` or read a raw PPM; no matplotlib)
  2. Center the data (subtract column means)
  3. Compute covariance matrix $C = \frac{1}{n-1}X^TX$
  4. Eigendecompose $C$ using `np.linalg.eigh`; sort eigenvalues descending
  5. Project to rank-$k$ subspace; reconstruct; compute RMSE and explained variance ratio
  6. Loop over $k \in \{5, 10, 20, 50\}$ and print a comparison table
- **Extensions:** Apply to a dataset of face images and find the minimum rank where faces are recognizable; implement incremental PCA for images that don't fit in RAM.

---

**P02 — Titanic Survival Predictor** `[Module 02 — ML Basics to Advanced]` `Beginner`
- **What:** End-to-end ML pipeline on the Titanic dataset. Feature engineering → cross-validation → compare Ridge Logistic Regression vs Random Forest vs XGBoost → ROC/PR curves → feature importance.
- **Why it's good:** Covers the full supervised learning workflow: missing value imputation, categorical encoding, scaling, CV, model comparison, evaluation — all topics from Module 02.
- **Phases:**
  1. Load `titanic.csv` (include in `src/projects/` as a small CSV or generate synthetic equivalent)
  2. Feature engineering: extract title from Name, bin Age, create FamilySize = SibSp + Parch + 1
  3. Impute missing values (median for Age, mode for Embarked); one-hot encode categoricals
  4. K-fold CV (from scratch or sklearn) for Logistic Regression + Random Forest + XGBoost; compare mean accuracy ± std
  5. Plot ROC/PR curves using the from-scratch implementations from Module 02 (print AUC values)
  6. Print feature importance table (permutation importance for RF; gain for XGBoost)
- **Extensions:** Build a calibration curve (Brier score); implement SMOTE from scratch for the class imbalance.

---

**P03 — Semantic Code Search Engine** `[Module 03 — Databases & Vector DBs]` `Intermediate`
- **What:** Index a corpus of Python function docstrings using TF-IDF + FAISS. Answer natural language queries with BM25 + dense hybrid retrieval. Return top-5 results with relevance scores.
- **Why it's good:** Covers the full vector DB pipeline: embedding, indexing, hybrid retrieval, metadata filtering — directly applying FAISS and BM25 from Module 03.
- **Phases:**
  1. Parse ~200 Python stdlib docstrings from `help()` output (or a manually curated JSON file in `src/projects/`)
  2. Build a TF-IDF embedder (from Module 03's `embed_store.py`); embed all docstrings
  3. Store in a FAISS IVFFlat index; also build a BM25 index from scratch
  4. Implement hybrid retrieval with RRF fusion; query with 10 test questions and print ranked results
  5. Add metadata filtering: filter by module (e.g., only `os` or `string` functions)
  6. Measure latency: BM25 vs dense vs hybrid for 100 repeated queries; print results table
- **Extensions:** Add ChromaDB as an alternative backend; implement query expansion (add synonyms from a small dict).

---

**P04 — Production ML Serving API** `[Module 04 — Backend with Flask]` `Intermediate`
- **What:** Deploy a trained sklearn Pipeline (Titanic model from P02 or any classifier) behind a Flask REST API with: HMAC auth, token-bucket rate limiting, async batch prediction via a simulated task queue, structured JSON logging.
- **Why it's good:** Assembles every piece from Module 04 into one production-grade service. Interviewer-ready system design.
- **Phases:**
  1. Train and pickle a sklearn Pipeline (preprocessor + RandomForestClassifier) in a setup script
  2. Build Flask app with app factory + two blueprints: `/api/v1/` (sync predict) and `/api/v2/` (async batch)
  3. Wire HMAC auth middleware and token-bucket rate limiter (from Module 04)
  4. Implement `/api/v2/predict/batch` → returns a task ID; `/api/v2/tasks/<id>` → polls status
  5. Add structured JSON request logging with `X-Request-ID` propagation
  6. Load-test: submit 50 concurrent single-predict requests; print p50/p95/p99 latency from logs
- **Extensions:** Add model versioning (v1 model vs v2 model in registry); add `/health` and `/metrics` (uptime + request count) endpoints.

---

**P05 — Neural Network Training Dashboard** `[Module 05 — Deep Learning & MLOps]` `Intermediate`
- **What:** Train a 3-layer neural network on a synthetic multi-class dataset from scratch (NumPy). Log all metrics to MLflow. Implement early stopping + cosine LR schedule. Run KS drift detection on validation set activations each epoch.
- **Why it's good:** Connects the entire Module 05 pipeline: backprop → Adam optimizer → LR scheduling → MLflow tracking → drift monitoring.
- **Phases:**
  1. Generate a 4-class spiral dataset (pure NumPy, no sklearn); split 70/15/15
  2. Implement 3-layer network (from `nn_numpy.py`): ReLU hidden, softmax output, cross-entropy loss
  3. Train with Adam (from `optimizers.py`) + cosine-warmup LR schedule (from `optimizers.py`)
  4. Log per-epoch: loss, accuracy, LR, gradient norm → MLflow run
  5. Implement early stopping (patience=10 on val loss); save best weights
  6. At each epoch, extract hidden layer activations; run KS test against epoch-0 baseline; log drift p-value to MLflow
  7. Print final: training curve table + MLflow run URL + whether drift was detected
- **Extensions:** Compare Adam vs SGD-Momentum convergence on the same run (log both to one MLflow experiment); add permutation feature importance using the trained network.

---

**P06 — Word Analogy & Semantic Similarity Explorer** `[Module 06 — GenAI Core]` `Intermediate`
- **What:** Train Word2Vec (skip-gram + negative sampling) on a text corpus. Implement analogy solver ($\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$). Build a KV-cache-aware greedy text generator that produces token continuations.
- **Why it's good:** Combines the two hardest scripts from Module 06 (`word2vec.py` and `kv_cache.py`) into a single interactive demo.
- **Phases:**
  1. Tokenize a text file (~10K sentences; include a small sample in `src/projects/` or use any plain-text book)
  2. Train skip-gram with negative sampling (from `word2vec.py`): window=5, dim=100, epochs=10
  3. Implement cosine nearest-neighbors; print top-10 similar words for 5 test words
  4. Implement analogy solver: $\text{nearest}(\mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c)$; test on 10 analogy pairs; print accuracy
  5. Build a stub autoregressive generator: given a seed phrase, predict next-token via dot product with embedding table; use KV cache structure from Module 06
  6. Print: analogy accuracy table, top-5 similar words for 5 queries, 3 generated continuations
- **Extensions:** Train FastText (subword n-grams); compare OOV handling by querying rare/misspelled words.

---

**P07 — Shakespeare GPT (Mini Language Model)** `[Module 07 — Transformers from Scratch]` `Advanced`
- **What:** Train a small GPT-style decoder-only transformer on the Shakespeare corpus. Implement BPE tokenizer on the corpus. Train with transformer LR schedule + gradient clipping. Generate text with temperature sampling, top-k, and top-p.
- **Why it's good:** This is Andrej Karpathy's nanoGPT, reproduced from the building blocks in Module 07. The most impactful single project on the list.
- **Phases:**
  1. Download `shakespeare.txt` (include in `src/projects/` or fetch); build BPE tokenizer from `tokenizer.py` on the corpus; encode full corpus
  2. Build character-level or BPE-level GPT decoder: 4 layers, 4 heads, $d=128$, context length 128 (small enough to train on CPU in ~20 min)
  3. Implement training loop from `train.py`: transformer LR schedule, label smoothing, gradient clipping, checkpoint save every 500 steps
  4. Track train/val loss per step; print loss table every 100 steps
  5. After training, generate 5 samples with temperature 1.0, 0.7 (greedy), and top-p=0.9; print all
  6. Print parameter count breakdown by component (embedding, MHA, FFN, total)
- **Extensions:** Implement top-k + top-p sampling with repetition penalty; add beam search from `train.py`; scale to 6 layers and compare loss.

---

**P08 — Personal Document Q&A System** `[Module 08 — RAG Chatbot]` `Advanced`
- **What:** Build a full RAG pipeline over a folder of PDF/TXT files (research papers, textbook chapters). Serve it as a Flask API. Evaluate with all 4 RAGAS metrics. Show retrieval strategy comparison (BM25 vs dense vs hybrid vs MMR) on 20 test questions.
- **Why it's good:** This is the most interview-relevant end-to-end project — directly maps to what every AI startup builds in their first sprint.
- **Phases:**
  1. Ingest 5–10 documents (use this platform's own module guides as the corpus — they're already in `docs/`)
  2. Apply recursive character chunking (512 tokens, 50 overlap); embed with TF-IDF + random projection (from `embed_store.py`)
  3. Build all four retriever variants from `retriever.py`; store in FAISS + BM25 index
  4. Write 20 test questions with ground-truth answers about the corpus
  5. Run each retriever variant on all 20 questions; generate answers with mock LLM; compute all 4 RAGAS metrics
  6. Print: retrieval strategy comparison table (precision, recall, faithfulness, latency per strategy)
  7. Serve via Flask API (`/ingest`, `/query?strategy=hybrid`, `/evaluate`) from `app.py`
- **Extensions:** Add a reranker (cross-encoder dot-product scoring on top-20 candidates); implement query decomposition (split complex questions into sub-queries).

---

**P09 — Domain-Specific Instruction Tuner** `[Module 09 — Fine-Tuning]` `Advanced`
- **What:** Prepare a custom Alpaca-format dataset from the module guides (Q&A pairs extracted from the Q&A sections). LoRA fine-tune TinyLLaMA-1.1B or Phi-2 (if GPU available) using QLoRA (4-bit NF4). Evaluate before/after with perplexity + BLEU + ROUGE. Merge adapter and compare inference speed.
- **Why it's good:** This closes the full curriculum loop — the platform's own content becomes fine-tuning data. Also directly demonstrates the most in-demand skill: adapting open-source LLMs.
- **Phases:**
  1. Extract all Q&A blocks from `docs/` module guides as instruction-response pairs; target ~200–500 pairs; format as Alpaca JSON
  2. Tokenize with a real tokenizer (HuggingFace tokenizer via graceful skip); implement manual BPE tokenization fallback using Module 07's `tokenizer.py`
  3. Configure LoRA ($r=8$, $\alpha=16$, target q/v projections) and NF4 quantization; use `train_lora.py` + `train_qlora.py` from Module 09
  4. Run NumPy gradient simulation training (always works); attempt real PEFT training (graceful skip if no GPU)
  5. Evaluate before/after: PPL on held-out Q&A pairs, BLEU-4, ROUGE-L
  6. Merge adapter using `merge_push.py`; print parameter count before/after merge; inference latency comparison
  7. Print: dataset stats, training loss curve, before/after metrics table, merged model size
- **Extensions:** Add DPO (Direct Preference Optimization) loss on rejected/chosen pairs; experiment with different LoRA target modules (q/k/v/o projections vs q/v only).

---

### TODO 2 — Additional Modules `[ ]`

Ranked by interview relevance for AI/ML/GenAI/Backend roles:

| Priority | Module | Key Topics | Why Now |
|----------|--------|-----------|---------|
| 1 | **Module 10 — LLM Agents & Tool Use** | ReAct loop, function calling (OpenAI tools API), chain-of-thought, multi-step planning, tool schemas, agent memory, error recovery | Agents are in every AI job description in 2025 |
| 2 | **Module 11 — Deployment & Production ML** | Docker + docker-compose, ONNX export, TorchScript, quantization (INT8/FP16), Gunicorn + nginx, health checks, rolling deploys, A/B serving | Closes the gap between "trains a model" and "ships a model" |
| 3 | **Module 12 — RLHF & Alignment** | PPO from scratch, reward modeling, DPO (Direct Preference Optimization) — simpler and now dominant — KL penalty, constitutional AI overview | RLHF/DPO is asked in every LLM-adjacent role |
| 4 | **Module 13 — Multimodal Models** | CLIP (contrastive image-text pre-training), vision transformers (ViT patch embedding), image captioning pipeline, cross-modal attention | Vision-language models are the next wave |

Each module follows the same format: one `.md` guide + `src/NN-*/` scripts.

---

### TODO 3 — Content Polish `[ ]`

- [ ] **Interview Q&A bank** — add 10 more Q&A pairs to each module (currently Modules 05–09 have Q&As; Modules 01–04 need them)
- [ ] **Complexity / trade-off summary table** — each module should end with a "Cheat Sheet" table suitable for last-minute review before an interview
- [ ] **Module 02** — add Decision Trees as a standalone section (currently embedded only in Random Forest); interviewers often ask DT-specific questions (impurity, pruning, CART)
- [ ] **Module 03** — add Redis as a caching layer section (commonly paired with vector DBs in production)
- [ ] **Module 04** — add a gRPC section (protocol buffers, streaming, bidirectional streaming) as an alternative to REST for ML serving
- [ ] **Module 05** — add ONNX export walkthrough; add TorchScript tracing (both are commonly asked)
- [ ] **Module 08** — add reranking section (cross-encoders, ColBERT late interaction); this is the most common "what would you do to improve RAG" answer

---

### TODO 4 — Projects Section Frontend Implementation `[ ]`

When implementing the Projects view, follow these rules:
1. Add "PROJECTS" nav button to sidebar, below the module list
2. Projects have their own grid on the welcome screen (below module grid) or as a separate tab
3. `openProject(file, label)` function in `app.js`: fetches `docs/projects/${file}`, renders markdown, shows "Mark Complete" button
4. Project cards use a different accent color to distinguish from module cards (suggest: black background, white text, red-orange tag for difficulty badge)
5. Project pages render identically to module pages (marked.js + MathJax + hljs)
6. Progress tracking: separate localStorage key `aiml_platform_projects_progress`
7. **Do not** break existing module navigation when adding projects

---

## RULES FOR FUTURE SESSIONS (CRITICAL)

1. **Read `docs/list.md`** to verify current module status before touching anything.
2. **All 9 modules are complete.** No new modules to build unless user requests.
3. **Pause after each module** and wait for user approval before starting the next.
4. **No half-done modules** — every module must have BOTH the `.md` guide AND all `src/` scripts before committing.
5. **Commit after every module** — use the HEREDOC commit format described above.
6. **Update `context.md`** — after completing each module: move it from "UPCOMING" to "COMPLETED SUMMARIES", update the progress table, update the footer timestamps.
7. **No AI attribution anywhere** — no "Claude", no "Anthropic", no "AI-generated" in any file.
8. **No matplotlib GUI** — all script output is terminal-printable only.
9. **Math notation consistency** — use the same LaTeX conventions as existing modules:
   - Weight vectors: $\mathbf{w}$
   - Matrices: $A$, $W$, $\Sigma$
   - Loss: $\mathcal{L}$
   - Expectation: $\mathbb{E}$
   - Reals: $\mathbb{R}$
   - Distributions: $\mathcal{N}$, $\text{Uniform}$
10. **Frontend automatically picks up new `.md` files** from `/docs/` — no frontend changes needed for new modules.
11. **Python runtime:** System uses `python3.14` (no `python` alias). PyTorch has no wheel for 3.14 yet — all PyTorch scripts use graceful `try/except ImportError` and exit cleanly. Pure NumPy + stdlib scripts run fine.

---

## HOW TO START A NEW SESSION

```
1. Read this file (context.md) fully.
2. Run: git log --oneline -5   (verify recent commits)
3. Check docs/list.md for current module status.
4. Ask user which module to proceed with (or they will tell you).
5. Build the next module: doc first → all src scripts → list.md update → context.md update → commit.
```

---

*Last updated: 2026-05-11 — 404 fix (app.js link intercept) + Prerequisites & Resources sections added to all 9 modules + TODO list added (Projects, 4 new modules, content polish)*
*Modules complete: 01, 02, 03, 04, 05, 06, 07, 08, 09 — ALL COMPLETE*
*Open TODOs: Projects section (9 project guides + frontend), Modules 10–13, content polish*
