# PROJECT CONTEXT — AI/ML Learning Platform
> **READ THIS ENTIRE FILE BEFORE DOING ANYTHING.**
> This is the single source of truth. No prior conversation context needed — everything is here.

---

## COLD-START ORIENTATION (read this first, every session)

**What this is:** A self-hosted interactive learning platform for Harshit Tiwari's AI/ML placement preparation. 9 complete learning modules covering Math → ML → Databases → Backend → Deep Learning → GenAI → Transformers → RAG → Fine-Tuning. Each module has a dense Markdown guide + runnable Python scripts.

**Who the user is:** Harshit Tiwari. Targeting AI/ML/GenAI/Backend engineering roles. Treat every module as prep for a senior ML engineering interview.

**Where things live:**
- Local path: `/Users/iamharshit188/Desktop/placement/prepration/guide/`
- Live site: `https://harshittiwari.me/guide/` (GitHub Pages, static)
- Git remote: `git@github.com:iamharshit188/guide.git`, branch `main`
- Local dev: `python server.py` → `http://localhost:3000`

**Current status as of 2026-05-11:**
- ✅ All 9 modules COMPLETE (guides + scripts + prerequisites sections + resources sections)
- ✅ GitHub Pages 404 bug FIXED (`.md` link intercept in `app.js`)
- 🔲 Projects section NOT started (9 project guides + frontend changes needed)
- 🔲 Modules 10–13 NOT started (Agents, Deployment, RLHF, Multimodal)
- 🔲 Content polish NOT started (Q&A banks, cheat sheets, DT section, etc.)

**Session boot sequence:**
```
1. Read this file fully (you are doing this now)
2. git log --oneline -5             (verify you're on the right commit)
3. Ask the user what they want to work on today
4. Do NOT start building anything until the user confirms the task
```

---

## REPOSITORY — EXACT CURRENT STRUCTURE

> **IMPORTANT:** Frontend files moved to repo root in commit `09a9613` (GitHub Pages migration). They are NOT in a `frontend/` folder.

```
guide/                              ← repo root = GitHub Pages web root
├── context.md                      ← THIS FILE
├── index.html                      ← Neo-Brutalism SPA (marked.js + hljs + MathJax)
├── style.css
├── app.js                          ← All frontend logic: nav, fetch, markdown render
├── server.py                       ← Flask dev server (local only, port 3000)
├── .nojekyll                       ← Disables Jekyll on GitHub Pages
├── .gitignore
├── docs/                           ← All markdown content (fetched directly by app.js)
│   ├── list.md                     ← Curriculum roadmap (update when modules change)
│   ├── 01-math.md                  ← COMPLETE — Prerequisites + Resources sections added
│   ├── 02-ml-basics.md             ← COMPLETE — Prerequisites + Resources sections added
│   ├── 03-databases.md             ← COMPLETE — Prerequisites + Resources sections added
│   ├── 04-backend.md               ← COMPLETE — Prerequisites + Resources sections added
│   ├── 05-deep-learning.md         ← COMPLETE — Prerequisites + Resources sections added
│   ├── 06-genai-core.md            ← COMPLETE — Prerequisites + Resources sections added
│   ├── 07-transformers.md          ← COMPLETE — Prerequisites + Resources sections added
│   ├── 08-rag.md                   ← COMPLETE — Prerequisites + Resources sections added
│   ├── 09-finetuning.md            ← COMPLETE — Prerequisites + Resources sections added
│   └── projects/                   ← DOES NOT EXIST YET — create when building TODO 1
└── src/                            ← Runnable Python (and one C++) scripts
    ├── 01-math/                    ← vectors.py, matrix_ops.py, calculus_demo.py, probability.py
    ├── 02-ml/                      ← linear_regression.py, logistic_regression.py, evaluation.py,
    │                                  clustering.py, pca.py, random_forest.py, svm.py,
    │                                  gradient_boosting.py
    ├── 03-databases/               ← sql_basics.py, nosql_patterns.py, chroma_demo.py,
    │                                  pinecone_demo.py, faiss_demo.py
    ├── 04-backend/                 ← app.py, ml_serving.py, middleware.py, async_tasks.py
    ├── 05-deep-learning/           ← nn_numpy.py, optimizers.py, mlflow_demo.py, monitoring.py
    ├── 06-genai/                   ← word2vec.py, attention.py, multihead_attention.py,
    │                                  positional_encoding.py, kv_cache.py
    ├── 07-transformer/             ← tokenizer.py, model.py, model_numpy.py, model.cpp, train.py
    ├── 08-rag/                     ← ingest.py, embed_store.py, retriever.py, generator.py,
    │                                  app.py, evaluate.py
    └── 09-finetuning/              ← lora_theory.py, prepare_dataset.py, train_lora.py,
                                       train_qlora.py, evaluate.py, merge_push.py
```

---

## CURRICULUM — WHAT IS COMPLETE

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

Every module guide now contains:
- **Prerequisites & Overview** section at the top (time estimate, module map, before-you-start checklist)
- **Resources** section at the bottom (papers with arxiv links, books, videos, tooling)
- **Q&A section** (Modules 05–09) — dense interview-prep questions with full answers

---

## FRONTEND — HOW IT WORKS

### Production (GitHub Pages static)
- `app.js` fetches markdown via `fetch('docs/${file}')` — relative URL, no server needed
- GitHub Pages serves the `docs/` folder directly as static files
- `.md` links inside rendered markdown are **intercepted client-side** (added 2026-05-11):
  ```js
  // In renderMarkdown(), after innerHTML is set:
  $("doc-rendered").querySelectorAll("a[href]").forEach((a) => {
    const basename = a.getAttribute("href").split("/").pop();
    const mod = MODULE_META.find((m) => m.file === basename);
    if (mod) {
      a.addEventListener("click", (e) => {
        e.preventDefault();
        openModule(mod.file, mod.label);
      });
    }
  });
  ```
  This prevents "Next module" links from causing GitHub Pages 404s.

### Local Dev
- `python server.py` → Flask on port 3000
- Serves `index.html` from root; `docs/*.md` as `text/plain`

### Design System
- **Neo-Brutalism:** `border: 3px solid black`, `box-shadow: 4px 4px 0 black`
- Background: cream `#FFFDE7`, accent: yellow `#FFD600`
- marked.js (Markdown), highlight.js (Python/Bash/C++/SQL), MathJax (`$...$` and `$$...$$`)
- `localStorage` key `aiml_platform_progress` — progress persists in browser
- "Mark Complete" button: `Cmd+Enter` shortcut

---

## MODULE CONTENT STANDARD

Every guide (`docs/NN-topic.md`) contains in this order:
1. **Title + run commands** (blockquote with `cd src/NN && python X.py` for every script)
2. **Prerequisites & Overview** — time estimate, module map table, before-you-start checklist
3. **Main content** — math derivations → algorithm → code equivalent. Dense, no fluff.
4. **Q&A section** (senior interview questions with complete answers) — Modules 05–09 have these; 01–04 do not yet
5. **Resources** — books, papers (arxiv links), videos, tooling
6. **Next module link** — `*Next: [Module NN — Title](NN-topic.md)*`

Every script (`src/NN-topic/*.py`) follows:
- Module docstring (3–5 lines: what it covers + pip install deps)
- `section("TITLE")` helper at top of every file
- From-scratch implementation first, then library equivalent
- Graceful import guards: `try: import X except ImportError: ...`
- Fixed seeds: `np.random.default_rng(42)`
- No `matplotlib.show()` — all output terminal-printable
- Ends with `if __name__ == "__main__": main()`
- **No AI attribution** anywhere in any file

---

## WRITING & CODE STYLE (non-negotiable)

**Tone:** Zero fluff. No "In this section we will explore...". Every sentence carries information. Math-first: derive the formula, then show the code equivalent. Tables over prose for comparisons.

**LaTeX conventions:**
- Weight vectors: `$\mathbf{w}$` — Loss: `$\mathcal{L}$` — Expectation: `$\mathbb{E}$`
- Reals: `$\mathbb{R}$` — Distributions: `$\mathcal{N}$`, `$\text{Uniform}$` — Matrices: `$A$`, `$W$`

**Python runtime:** System is `python3.14`. No `python` alias exists. PyTorch has no 3.14 wheel — all PyTorch code uses `try: import torch except ImportError: sys.exit(0)`. Pure NumPy + stdlib scripts always run.

---

## GIT COMMIT PROTOCOL

```bash
git add <specific files>          # never git add -A or git add .
git commit -m "$(cat <<'EOF'
type: short summary

- bullet explaining what was built/fixed
- bullet per significant file changed

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- **Never `--amend`**, never `--no-verify`
- Never commit `.env`, `*.pkl`, `*.pt`, `*.pth` files
- After every completed module: `git add docs/NN.md docs/list.md src/NN-*/` then commit

---

## COMPLETED MODULES — WHAT EACH COVERS

### Module 01 — Math for ML
- **Linear Algebra:** L1/L2/L∞ norms, cosine similarity, projection, matrix ops, eigendecomposition ($A = V\Lambda V^{-1}$), SVD ($A = U\Sigma V^T$), pseudoinverse, PCA via eigendecomp
- **Calculus:** Partial derivatives, gradient, chain rule, Jacobian, Hessian, gradient descent, numerical gradient check
- **Probability:** Bayes, distributions (Gaussian/Bernoulli/Categorical/Poisson), MLE derivation, cross-entropy, KL divergence, covariance, Mahalanobis distance
- **Scripts:** `vectors.py`, `matrix_ops.py`, `calculus_demo.py`, `probability.py`

### Module 02 — ML Basics to Advanced
- **Linear Regression:** OLS normal equation, GD, Ridge/Lasso derivations, multicollinearity
- **Logistic Regression:** BCE from MLE, softmax for multiclass, threshold analysis
- **Evaluation:** Bias-variance (bootstrap), K-fold CV, ROC/PR curves, confusion matrix, F-β, calibration, imbalanced classes
- **Clustering:** K-Means++, elbow, silhouette, DBSCAN — all from scratch
- **PCA:** Full eigendecomp, SVD equivalence, EVR, reconstruction error
- **Random Forest:** Gini/entropy, decision tree + RF from scratch, MDI + permutation importance, OOB error
- **SVM:** Mercer kernels, subgradient SVM from scratch, kernel comparison, hinge vs log-loss
- **Gradient Boosting:** GBR/GBC from scratch, early stopping, XGBoost 2nd-order Taylor math
- **Scripts:** `linear_regression.py`, `logistic_regression.py`, `evaluation.py`, `clustering.py`, `pca.py`, `random_forest.py`, `svm.py`, `gradient_boosting.py`

### Module 03 — Databases & Vector DBs
- **SQL:** B+ tree internals, join types + algorithms, composite index left-prefix rule, EXPLAIN QUERY PLAN, window functions, CTEs, recursive CTEs
- **NoSQL:** CAP theorem, PACELC, 5 NoSQL types with ML use cases, MongoDB aggregation pipeline
- **ANN Math:** Cosine/L2/IP metric relationships for normalized vectors
- **HNSW:** Multi-layer graph, greedy descent query, M/efConstruction/efSearch parameters
- **IVF + PQ:** Voronoi cells, K-Means training, nprobe tradeoff, sub-quantizer compression
- **ChromaDB / Pinecone (mock) / FAISS:** Full APIs — FlatL2 → IVFFlat → HNSWFlat → PQ → IVFPQ, recall@10 vs QPS sweep
- **Scripts:** `sql_basics.py`, `nosql_patterns.py`, `chroma_demo.py`, `pinecone_demo.py` (mock), `faiss_demo.py`

### Module 04 — Backend with Flask
- **Flask Internals:** WSGI interface, request context lifecycle, thread-local `g`/`request`/`current_app`
- **App Factory:** `create_app(env)`, Config class hierarchy (Base/Dev/Test/Prod), multi-env isolation
- **Blueprints:** URL-scoped routing, versioned APIs (`/api/v1/`, `/api/v2/`)
- **ML Serving:** Thread-safe model registry (`RLock`), sklearn pickle + PyTorch TorchScript, single + batch `/predict`, input validation
- **Middleware:** HMAC timing-safe auth, Token Bucket rate limiter, per-client limiter, structured JSON logging, `X-Request-ID`
- **Async Tasks:** Celery state machine (PENDING→STARTED→SUCCESS/FAILURE/RETRY), 202/poll pattern, in-process simulation
- **Scripts:** `app.py`, `ml_serving.py`, `middleware.py`, `async_tasks.py`

### Module 05 — Deep Learning & MLOps
- **Backprop:** Full derivation — $\delta^{(L)} = \hat{y} - y$, hidden deltas, parameter gradients
- **Init:** Xavier (Var=$2/(n_{in}+n_{out})$), He ($2/n_{in}$), zero-symmetry failure, empirical stability over 10 layers
- **Activations:** ReLU/Sigmoid/Tanh/Softmax with gradients; numerically stable implementations
- **Optimizers:** SGD, Momentum, Nesterov, RMSProp, Adam (bias-correction derivation), AdamW; convergence benchmarks on quadratic + Rosenbrock
- **LR Scheduling:** StepDecay, CosineAnnealing, WarmupCosine, ReduceOnPlateau
- **Regularisation:** Dropout (inverted), BatchNorm (running stats, γ/β), LayerNorm (sample-wise, used in transformers)
- **MLflow:** Experiment/Run/Param/Metric/Artifact hierarchy, model registry lifecycle, autolog
- **Drift Detection:** KS test, Chi-squared, PSI, MMD (RBF kernel), JS divergence; DriftDetector class
- **Scripts:** `nn_numpy.py`, `optimizers.py`, `mlflow_demo.py`, `monitoring.py`

### Module 06 — GenAI Core
- **Word2Vec:** Skip-gram + negative sampling ($\mathcal{L}_{\text{NEG}}$), noise distribution $P_n \propto f^{3/4}$, analogy via vector arithmetic, FastText subword n-grams
- **Attention:** $\text{softmax}(QK^T/\sqrt{d_k})V$, variance scaling derivation, causal mask, cross-attention, $O(n^2d)$ complexity
- **MHA:** $W^Q, W^K, W^V, W^O$ projections, GPT-2 parameter count worked example, MQA, GQA
- **Positional Encoding:** Sinusoidal ($\omega_i = 1/10000^{2i/d}$), learned PE, RoPE 2D rotation
- **KV Cache:** FLOP comparison ($O(T^2d)$ vs $O(Td)$), memory formula, GPT-3 ≈ 9.66 GB at T=2048, MQA/GQA/quantisation reduction, PagedAttention analogy
- **Scripts:** `word2vec.py`, `attention.py`, `multihead_attention.py`, `positional_encoding.py`, `kv_cache.py`

### Module 07 — Transformers from Scratch
- **BPE Tokenizer:** Merge loop, encode/decode, OOV via char fallback, Tiktoken/SentencePiece comparison
- **Architecture:** Encoder × $N_e$, Decoder × $N_d$, Pre-LN vs Post-LN table, FFN ($d_{ff}=4d$, SwiGLU), residual connections
- **Parameter count:** $12Ld^2$ per block; GPT-2 small: 85M transformer + 38.6M embed = 117M ✓
- **Training:** Teacher forcing, transformer LR schedule ($d^{-0.5}\min(t^{-0.5}, t \cdot t_w^{-1.5})$), gradient clipping, label smoothing
- **Inference:** Greedy, beam search, temperature/top-k/top-p
- **Variants:** BERT, GPT, T5, LLaMA (RoPE+SwiGLU+GQA+RMSNorm), Mistral
- **Scripts:** `tokenizer.py`, `model.py` (PyTorch, graceful skip), `model_numpy.py`, `model.cpp` (C++17, no deps), `train.py`

### Module 08 — RAG Chatbot
- **Architecture:** Retrieve → inject → generate; parametric vs non-parametric knowledge
- **Chunking:** Fixed-size (sliding window), recursive character splitting, semantic chunking
- **Embedding:** TF-IDF from scratch ($\text{IDF} = \log((N+1)/(df+1))+1$), random projection, ChromaDB
- **Retrieval:** Dense cosine, BM25 ($k_1=1.5$, $b=0.75$), hybrid RRF, hybrid linear, MMR
- **Generation:** Context window budget, mock LLM, OpenAI wrapper (graceful skip), citation formatting
- **Flask API:** `POST /ingest`, `POST /query`, `GET /collection`, `DELETE /collection`
- **RAGAS Evaluation:** Faithfulness, Answer Relevancy, Context Precision, Context Recall — all from scratch
- **Scripts:** `ingest.py`, `embed_store.py`, `retriever.py`, `generator.py`, `app.py`, `evaluate.py`

### Module 09 — Fine-Tuning (LoRA/QLoRA)
- **LoRA:** $W' = W_0 + \frac{\alpha}{r}BA$, $B=0$ init, $\alpha/r$ scaling, rank sensitivity, adapter merging, SVD rank verification
- **QLoRA:** NF4 construction (quantiles of $\mathcal{N}(0,1)$, renorm to $[-1,1]$), INT4 vs NF4 RMSE, double quantisation (0.127 bits/weight overhead), paged Adam, memory breakdown (≈8 GB for 7B)
- **Dataset prep:** Alpaca/ChatML/LLaMA-3 templates, label masking ($-100$ for instruction tokens), data collator
- **Evaluation:** PPL ($\exp(\text{mean CE})$), BLEU-4, ROUGE-1/2/L — all from scratch
- **Merge:** $W_{\text{merged}} = W_0 + \frac{\alpha}{r}BA$, multi-adapter composition
- **Scripts:** `lora_theory.py`, `prepare_dataset.py`, `train_lora.py`, `train_qlora.py`, `evaluate.py`, `merge_push.py`

---

## TODO — WHAT TO BUILD NEXT

> These are ordered by priority. Confirm with user before starting any item.

---

### TODO 1 — Projects Section `[ ] NOT STARTED` ← **DO THIS FIRST**

**What:** Add 9 guided project pages to the platform (one per module). Each project opens in its own view, gives approach + phases + checkpoints + hints — **not full code**. They bridge reading a module and building something real.

**Frontend work needed in `app.js` and `index.html`:**
- Add `PROJECT_META` array (9 entries with `file`, `label`, `module`, `difficulty`)
- Add `openProject(file, label)` function: fetches `docs/projects/${file}`, renders same as `openModule()`
- Add "PROJECTS" button to sidebar nav (between module list and ROADMAP button)
- Add projects grid to welcome screen (below module grid)
- Separate localStorage key: `aiml_platform_projects_progress`
- Project cards: different accent color from module cards; difficulty badge (Beginner/Intermediate/Advanced)
- Do NOT break existing module navigation

**Content to create — `docs/projects/` folder (create it):**

Each project file structure:
1. Title + difficulty badge
2. What you'll build (2–3 sentences, end result)
3. Skills exercised (bullet list → links to module section)
4. Approach: numbered phases (pseudocode/skeletons only, no full solutions)
5. Checkpoints: what correct output looks like per phase
6. Extensions: 2–3 harder variants
7. Hints: 3–5 spoiler-style targeted hints

| File | Project | Module | Difficulty |
|------|---------|--------|-----------|
| `p01-pca-compressor.md` | PCA Image Compressor | 01 | Beginner |
| `p02-titanic-pipeline.md` | Titanic Survival Predictor | 02 | Beginner |
| `p03-semantic-search.md` | Semantic Code Search Engine | 03 | Intermediate |
| `p04-ml-api.md` | Production ML Serving API | 04 | Intermediate |
| `p05-training-dashboard.md` | Neural Network Training Dashboard | 05 | Intermediate |
| `p06-word-analogy.md` | Word Analogy & Similarity Explorer | 06 | Intermediate |
| `p07-gpt-shakespeare.md` | Shakespeare GPT | 07 | Advanced |
| `p08-document-qa.md` | Personal Document Q&A System | 08 | Advanced |
| `p09-domain-tuner.md` | Domain-Specific Instruction Tuner | 09 | Advanced |

**Project specs (exact what to build for each):**

**P01 — PCA Image Compressor** (Module 01, Beginner)
Load grayscale image as NumPy matrix → center data → compute covariance $C = \frac{1}{n-1}X^TX$ → eigendecompose with `np.linalg.eigh` → project + reconstruct at ranks {5, 10, 20, 50} → print RMSE + explained variance table. No sklearn in core. Extensions: face dataset, incremental PCA.

**P02 — Titanic Survival Predictor** (Module 02, Beginner)
Load titanic.csv → feature engineering (extract title, bin Age, FamilySize) → impute + encode → K-fold CV comparing Logistic Regression vs Random Forest vs XGBoost → ROC/PR curves (from-scratch implementations) → feature importance table. Extensions: calibration curve, SMOTE from scratch.

**P03 — Semantic Code Search Engine** (Module 03, Intermediate)
Parse ~200 Python stdlib docstrings → TF-IDF embed → FAISS IVFFlat index + BM25 index → hybrid retrieval with RRF → metadata filtering by module → latency table (BM25 vs dense vs hybrid for 100 queries). Extensions: ChromaDB backend, query expansion.

**P04 — Production ML Serving API** (Module 04, Intermediate)
Pickle a sklearn Pipeline → Flask app factory + two blueprints (`/api/v1/` sync, `/api/v2/` async batch) → HMAC auth + token-bucket rate limiter → `/predict/batch` returns task ID → `/tasks/<id>` polls status → structured JSON logging → p50/p95/p99 latency print from 50 concurrent requests. Extensions: model versioning, `/health` + `/metrics` endpoints.

**P05 — Neural Network Training Dashboard** (Module 05, Intermediate)
Generate 4-class spiral dataset (pure NumPy) → 3-layer network with Adam + cosine-warmup schedule → early stopping (patience=10) → MLflow logging (loss, accuracy, LR, gradient norm per epoch) → KS drift detection on hidden activations each epoch → print training table + drift log. Extensions: Adam vs SGD-Momentum comparison, permutation feature importance.

**P06 — Word Analogy & Similarity Explorer** (Module 06, Intermediate)
Tokenize 10K-sentence text corpus → train Word2Vec skip-gram + NS (window=5, dim=100, epochs=10) → cosine nearest-neighbors for 5 test words → analogy solver $\text{nearest}(\mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_c)$ on 10 pairs → stub autoregressive generator with KV cache structure → print analogy accuracy table + top-5 similar words + 3 generated continuations. Extensions: FastText subword n-grams, OOV comparison.

**P07 — Shakespeare GPT** (Module 07, Advanced)
BPE tokenizer on shakespeare.txt corpus → GPT decoder (4 layers, 4 heads, d=128, ctx=128, trains on CPU in ~20 min) → transformer LR schedule + gradient clipping + label smoothing → train/val loss table every 100 steps → checkpoint every 500 steps → generate 5 samples at temperature {1.0, 0.7} and top-p=0.9 → print parameter breakdown by component. Extensions: repetition penalty, beam search, scale to 6 layers.

**P08 — Personal Document Q&A System** (Module 08, Advanced)
Use `docs/` module guides as corpus → recursive chunking (512 tokens, 50 overlap) → TF-IDF + random projection embed → all four retriever variants (BM25, dense, hybrid RRF, MMR) → 20 test questions with ground-truth → run all strategies, compute all 4 RAGAS metrics → print strategy comparison table (precision, recall, faithfulness, latency) → serve via Flask API. Extensions: reranker on top-20 candidates, query decomposition.

**P09 — Domain-Specific Instruction Tuner** (Module 09, Advanced)
Extract Q&A pairs from `docs/` guides → Alpaca JSON format → BPE tokenize (fallback to Module 07 tokenizer) → LoRA ($r=8$, $\alpha=16$, q/v projections) + NF4 QLoRA → NumPy gradient simulation (always runs) + real PEFT (graceful skip if no GPU) → PPL + BLEU-4 + ROUGE-L before/after → merge adapter → latency comparison → print dataset stats, training curve, metrics table, merged model size. Extensions: DPO loss, different target modules.

---

### TODO 2 — Additional Modules `[ ] NOT STARTED`

Build in this order. Each follows the same structure as existing modules: `.md` guide + `src/NN-*/` scripts.

| # | Module | Key Topics | Why |
|---|--------|-----------|-----|
| 10 | **LLM Agents & Tool Use** | ReAct loop, function calling (OpenAI tools API format), chain-of-thought, multi-step planning, tool schemas (JSON), agent memory (buffer + summary), error recovery, agent evaluation | In every AI job description in 2025–2026 |
| 11 | **Deployment & Production ML** | Docker + docker-compose, ONNX export, TorchScript tracing, INT8/FP16 quantization for serving, Gunicorn + nginx config, health checks, rolling deploys, A/B model serving, CI/CD hooks | Closes "trains a model" → "ships a model" gap |
| 12 | **RLHF & Alignment** | Reward modeling, PPO from scratch (clip objective, value function), DPO (Direct Preference Optimization, now dominant over PPO), KL penalty, constitutional AI overview, evaluation (win-rate, MT-Bench) | Asked in every LLM-adjacent role; DPO is now production standard |
| 13 | **Multimodal Models** | CLIP (contrastive image-text pre-training, InfoNCE loss), ViT patch embedding, image captioning pipeline (vision encoder + language decoder), cross-modal attention, zero-shot image classification | Vision-language is the next mainstream wave |

---

### TODO 3 — Content Polish `[ ] NOT STARTED`

Fix these in any order when time permits:

- `[ ]` **Interview Q&A banks for Modules 01–04** — currently only Modules 05–09 have Q&A sections; add 8–10 dense senior-level Q&As to each of 01–04
- `[ ]` **Interview Cheat Sheet table** — add a 1-page "Last 30 minutes before the interview" summary table at the bottom of each module; key formulas, key trade-offs, key code patterns
- `[ ]` **Module 02 — Decision Trees standalone section** — currently DT is embedded inside Random Forest; interviewers ask DT-specific questions (CART, impurity, pruning, max_depth); needs its own `###` section and a `decision_tree.py` script
- `[ ]` **Module 03 — Redis caching section** — add Redis as a caching layer (LRU, TTL, pipeline batching, connection pooling); commonly asked alongside vector DB questions in ML infra interviews
- `[ ]` **Module 04 — gRPC section** — add protocol buffers + gRPC streaming as alternative to REST for ML serving; unary, server-streaming, and bidirectional patterns
- `[ ]` **Module 05 — ONNX export walkthrough** — add `onnx_export.py` showing sklearn Pipeline → ONNX and NumPy NN → ONNX; inference with `onnxruntime`; commonly asked in production ML interviews
- `[ ]` **Module 08 — Reranking section** — add cross-encoder reranking (dot-product scoring on top-20 candidates → keep top-5), ColBERT late interaction overview; the most common "how would you improve RAG" interview answer

---

## RULES (apply to every session, no exceptions)

1. **Read `docs/list.md`** before touching module content — it's the canonical status tracker.
2. **Confirm task with user** before building anything. Do not assume.
3. **No half-done modules** — guide AND all scripts must be complete before committing.
4. **No AI attribution** — no "Claude", "Anthropic", "AI-generated" anywhere in any file.
5. **No `matplotlib.show()`** — all output terminal-printable.
6. **Never `--amend`**, never `--no-verify`**, never `git add -A`**.
7. **Update `context.md`** after completing any TODO item: update the `[ ]` to `[x]`, update the footer.
8. **Commit after every completed unit** (module, project file, or frontend feature) — never leave uncommitted changes.
9. **Python runtime is `python3.14`** — no `python` alias. PyTorch has no 3.14 wheel; all torch code uses graceful skip.
10. **Frontend: do not break existing module navigation** when adding new features (projects, new modules).
11. **Projects: approach only, no full solutions** — pseudocode and skeletons are fine; complete working implementations defeat the purpose.

---

## HOW TO HANDLE COMMON USER REQUESTS

| User says | What to do |
|-----------|-----------|
| "Build the projects section" | Start with TODO 1: frontend changes first (`app.js` + `index.html`), then create `docs/projects/` and write all 9 `.md` files, then commit |
| "Add Module 10 / Agents" | Follow module standard: `docs/10-agents.md` + `src/10-agents/*.py` + update `docs/list.md` + update this file |
| "Fix X in a module" | Read the specific module file first, make targeted edits, commit |
| "Push to GitHub" | `git push origin main` — confirm with user first since it affects the live site |
| "Add content to Module NN" | Read `docs/NN-*.md` first, then edit; commit immediately after |
| "What's left to do?" | Point to TODO 1, 2, 3 sections above in order |

---

*Last updated: 2026-05-11*
*Modules complete: 01–09 (ALL). Each has Prerequisites + Resources sections.*
*Recent: 404 fix in app.js (link intercept), Prerequisites + Resources added to all 9 modules*
*Open: TODO 1 (Projects), TODO 2 (Modules 10–13), TODO 3 (content polish)*
