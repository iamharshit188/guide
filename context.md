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
- ✅ All 13 modules COMPLETE (guides + scripts + prerequisites sections + resources sections)
- ✅ GitHub Pages 404 bug FIXED (`.md` link intercept in `app.js`)
- ✅ Projects section COMPLETE (9 project guides in `docs/projects/` + frontend: `app.js`, `index.html`, `style.css`)
- ✅ Modules 10–13 COMPLETE (Agents, Deployment, RLHF, Multimodal)
- ✅ Content polish COMPLETE (Q&A banks 01–04, cheat sheets all modules, DT section, Redis, gRPC, reranking, ONNX export)
- ✅ UI revamp COMPLETE (loading screen, greeting bar, smooth transitions, mobile sidebar, footer, accessibility)
- ✅ Dark Mode Neo-Brutalism COMPLETE (Space Grotesk/Outfit/JetBrains Mono fonts, dark palette, tabbed sidebar)

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
│   ├── 10-agents.md                ← COMPLETE — ReAct, tool schemas, CoT, memory, eval
│   ├── 11-deployment.md            ← COMPLETE — Docker, ONNX, quantization, A/B serving
│   ├── 12-rlhf.md                  ← COMPLETE — Reward model, PPO, DPO, alignment eval
│   ├── 13-multimodal.md            ← COMPLETE — CLIP, ViT, captioning, zero-shot
│   └── projects/                   ← COMPLETE — 9 project guides (p01–p09)
└── src/                            ← Runnable Python (and one C++) scripts
    ├── 01-math/                    ← vectors.py, matrix_ops.py, calculus_demo.py, probability.py
    ├── 02-ml/                      ← linear_regression.py, logistic_regression.py, evaluation.py,
    │                                  clustering.py, pca.py, random_forest.py, svm.py,
    │                                  gradient_boosting.py, decision_tree.py
    ├── 03-databases/               ← sql_basics.py, nosql_patterns.py, chroma_demo.py,
    │                                  pinecone_demo.py, faiss_demo.py
    ├── 04-backend/                 ← app.py, ml_serving.py, middleware.py, async_tasks.py
    ├── 05-deep-learning/           ← nn_numpy.py, optimizers.py, mlflow_demo.py,
    │                                  monitoring.py, onnx_export.py
    ├── 06-genai/                   ← word2vec.py, attention.py, multihead_attention.py,
    │                                  positional_encoding.py, kv_cache.py
    ├── 07-transformer/             ← tokenizer.py, model.py, model_numpy.py, model.cpp, train.py
    ├── 08-rag/                     ← ingest.py, embed_store.py, retriever.py, generator.py,
    │                                  app.py, evaluate.py
    ├── 09-finetuning/              ← lora_theory.py, prepare_dataset.py, train_lora.py,
    │                                  train_qlora.py, evaluate.py, merge_push.py
    ├── 10-agents/                  ← react_agent.py, tool_calling.py, agent_memory.py,
    │                                  agent_eval.py
    ├── 11-deployment/              ← onnx_export.py, quantize.py, ab_serving.py,
    │                                  health_check.py
    ├── 12-rlhf/                    ← reward_model.py, ppo_scratch.py, dpo.py,
    │                                  evaluate_alignment.py
    └── 13-multimodal/              ← clip_scratch.py, vit_patch.py, captioning.py,
                                       zero_shot.py
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
| 10 | LLM Agents & Tool Use | ✅ COMPLETE | `docs/10-agents.md` | `src/10-agents/` |
| 11 | Deployment & Production ML | ✅ COMPLETE | `docs/11-deployment.md` | `src/11-deployment/` |
| 12 | RLHF & Alignment | ✅ COMPLETE | `docs/12-rlhf.md` | `src/12-rlhf/` |
| 13 | Multimodal Models | ✅ COMPLETE | `docs/13-multimodal.md` | `src/13-multimodal/` |

Every module guide now contains:
- **Prerequisites & Overview** section at the top (time estimate, module map, before-you-start checklist)
- **Resources** section at the bottom (papers with arxiv links, books, videos, tooling)
- **Q&A section** (all 13 modules) — dense interview-prep questions with full answers
- **Cheat Sheet table** (Modules 01–04 added in content polish) — key formulas, trade-offs, code patterns

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
- **Dark Neo-Brutalism:** `border: 3px solid var(--gray-line)` and spring UI transitions.
- Background: Pure black `#050505` & Soft black `#121212`, Accent: Electric yellow `#E6FF00`
- Typography: Space Grotesk (headers), Outfit (body), JetBrains Mono (code)
- marked.js (Markdown), highlight.js (Python/Bash/C++/SQL), MathJax (`$...$` and `$$...$$`)
- `localStorage` key `aiml_platform_progress` — module progress persists in browser
- `localStorage` key `aiml_platform_projects_progress` — project progress persists separately
- "Mark Complete" button: `Cmd+Enter` shortcut
- `MODULE_META`: 13 entries (modules 01–13)
- `PROJECT_META`: 9 entries (projects p01–p09), difficulty badges (Beginner/Intermediate/Advanced)
- Sidebar UI: Tabbed navigation (MODULES / PROJECTS) dynamically toggled via JS; projects grid on welcome screen

### UI Features (added 2026-05-11)
- **Loading screen:** Full-screen black overlay, pulsing `ML` logo, shimmer scan bar, blinking status text; minimum 1.4 s display (even if init is instant); fades out with CSS opacity transition
- **Greeting bar:** Time-aware greeting ("Good morning/afternoon/evening/Working late, Harshit") + live stats (`X/13 modules · Y/9 projects · Z% complete`); updates on every progress change; sits at top of welcome screen in black bar with yellow text
- **Smooth transitions:** `fadeContent()` wraps all view switches (module open, project open, roadmap return) with a 180 ms opacity fade; module/project grid cards stagger-animate in with 35 ms per-card delay (`wmc-enter` keyframe); welcome cards slide up on load
- **Mobile sidebar:** Hamburger button (animates ☰ → ✕) slides sidebar in via CSS `translateX`; dim overlay behind sidebar; tap overlay or press `Escape` to close; `--sidebar-w: 0` on mobile so main takes full width
- **Footer:** `Made with ❤️ by Harshit and AI` — fixed at bottom of `#main`, always visible; `flex-shrink: 0` below scrollable content area
- **Accessibility:** Skip-to-content link (keyboard-visible only), `tabindex="0"` + `Enter`/`Space` key handlers on all nav/grid items, ARIA roles (`navigation`, `progressbar`, `status`, `contentinfo`, `live`, `listitem`), `aria-expanded` on hamburger, all decorative elements `aria-hidden`
- **Color fixes:** `.wmc-9` (cyan), `.wmc-10` (pink), `.wmc-11` (teal), `.wmc-12` (indigo) — the missing left-border accent stripes for modules 09–13

---

## MODULE CONTENT STANDARD

Every guide (`docs/NN-topic.md`) contains in this order:
1. **Title + run commands** (blockquote with `cd src/NN && python X.py` for every script)
2. **Prerequisites & Overview** — time estimate, module map table, before-you-start checklist
3. **Main content** — math derivations → algorithm → code equivalent. Dense, no fluff.
4. **Q&A section** (senior interview questions with complete answers) — all 13 modules have these
4a. **Cheat Sheet table** — all modules 01–13 have a "Last 30 minutes before the interview" formula + trade-off table
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

### Module 10 — LLM Agents & Tool Use
- **ReAct loop:** Thought→Action→Observation formal definition, prompt template, `ReActAgent` class with step limit + error recovery
- **Tool schemas:** OpenAI function-calling JSON Schema format, `@tool` decorator registry, `dispatch()` and `dispatch_parallel()` (ThreadPoolExecutor)
- **Chain-of-thought:** Zero-shot, few-shot with examples, self-consistency (majority vote across K samples)
- **Multi-step planning:** Plan-then-execute, hierarchical planning, plan validation
- **Memory:** `BufferMemory` (deque), `SummaryMemory` (auto-compress at token limit), `EntityMemory` (rule-based extraction), `CombinedMemory`
- **Evaluation:** trajectory accuracy, tool precision/recall/F1, answer F1, step efficiency
- **Scripts:** `react_agent.py`, `tool_calling.py`, `agent_memory.py`, `agent_eval.py`

### Module 11 — Deployment & Production ML
- **Docker:** Multi-stage Dockerfile, docker-compose with nginx, health checks in compose
- **ONNX:** Graph structure (nodes/initializers/inputs/outputs), sklearn Pipeline → ONNX via skl2onnx, NumPy MLP manual export, onnxruntime inference + latency benchmark
- **Quantization:** FP32→FP16, INT8 symmetric/asymmetric/per-channel, INT4, NF4 from scratch; static vs dynamic; PTQ vs QAT
- **Gunicorn:** Worker models (sync/gevent/uvicorn), `2×CPU+1` rule, graceful reload
- **A/B serving:** `ABRouter` (weighted random), `CanaryController` (ramp-up + rollback triggers), `ShadowRouter` (fire-and-forget), `StickyRouter` (hash-based)
- **Health/circuit breaker:** `/health`, `/ready`, `/metrics` endpoints; `CircuitBreaker` (CLOSED→OPEN→HALF-OPEN); `GracefulShutdown` (in-flight counter)
- **Scripts:** `onnx_export.py`, `quantize.py`, `ab_serving.py`, `health_check.py`

### Module 12 — RLHF & Alignment
- **Reward modeling:** Bradley-Terry loss $P(y_w\succ y_l)=\sigma(r_w-r_l)$, margin loss, length bias, normalization; `LinearRewardModel` with analytical gradient
- **PPO:** GAE $A_t^{GAE}=\sum(\gamma\lambda)^k\delta_{t+k}$, clip objective $\min(r_tA_t, \text{clip}(r_t,1-\varepsilon,1+\varepsilon)A_t)$, tabular MDP demo (5-state grid)
- **DPO:** Full algebraic derivation — Z(x) cancels, $\mathcal{L}_{DPO}=-\mathbb{E}[\log\sigma(\beta(r_\theta(y_w)-r_\theta(y_l)))]$
- **DPO variants:** IPO (MSE regularization), KTO (binary labels), SimPO (no reference model)
- **Constitutional AI:** Critique-revision loop
- **Evaluation:** Win-rate with bootstrap CI, RM accuracy, MT-Bench (8-category simulation), reward hacking detection
- **Scripts:** `reward_model.py`, `ppo_scratch.py`, `dpo.py`, `evaluate_alignment.py`

### Module 13 — Multimodal Models
- **CLIP:** InfoNCE loss derivation ($L=-\frac{1}{B}\sum\log\frac{\exp(S_{ii}/\tau)}{\sum_j\exp(S_{ij}/\tau)}$), temperature $\tau$ effect, contrastive training simulation, retrieval R@k evaluation
- **ViT:** Patch extraction formula $N=(H/P)(W/P)$, linear projection (≡ Conv2d with stride=P), [CLS] token, 2D sinusoidal PE + learned PE, MHA, full ViT encoder forward pass
- **Cross-modal attention:** Q from text, K/V from image; attention weight analysis per head
- **Flamingo gated cross-attention:** `x = x + tanh(α)·CrossAttn(LN(x), visual)`; $\alpha=0$ init → visual signal gated out at start
- **Q-Former (BLIP-2):** 32 learned queries cross-attend to image tokens → compressed visual prompt for frozen LLM
- **Captioning:** Teacher forcing loss, greedy decode, beam search; architecture comparison table
- **Zero-shot:** $\hat{y}=\arg\max_c\cos(\mathbf{v},\mathbf{t}_c)$, prompt ensemble (`L2_normalize(mean(embs))`), temperature sensitivity, confusion matrix
- **Scripts:** `clip_scratch.py`, `vit_patch.py`, `captioning.py`, `zero_shot.py`

### Content Polish (completed 2026-05-11)
- **Modules 01–04:** 8-question senior-level Q&A bank + cheat sheet table added to each guide
- **Module 02:** `decision_tree.py` — CART from scratch (Gini/entropy, best split, pruning, MDI feature importance, CART vs ID3/C4.5)
- **Module 03:** Redis section — LRU cache, TTL, pipeline batching, connection pooling, embedding cache pattern
- **Module 04:** gRPC section — protobuf schema, server/client implementation, streaming patterns, REST vs gRPC comparison table
- **Module 05:** `onnx_export.py` — ONNX graph anatomy, NumPy MLP → ONNX manual export, sklearn Pipeline → ONNX (with skl2onnx), onnxruntime inference + latency
- **Module 08:** Reranking section — bi-encoder vs cross-encoder, ColBERT late interaction (MaxSim), two-stage retrieval latency table

### Projects Section (completed 2026-05-11)
- **Frontend:** `PROJECT_META` (9 entries) in `app.js`, `openProject()`, sidebar tab navigation (Modules vs Projects), projects welcome card with grid, difficulty badges, orange active accent, separate `aiml_platform_projects_progress` localStorage key
- **Content:** 9 project guides in `docs/projects/` — each with: what-you-build, skills, phased approach (pseudocode only), checkpoints, extensions, hints

---

## COMPLETED TODOS

All three TODOs are complete as of 2026-05-11. Preserved here for reference.

### TODO 1 — Projects Section ✅ COMPLETE

9 project guides in `docs/projects/` (p01–p09). Each: what-you-build, skills, phased approach (pseudocode/skeletons only), checkpoints, extensions, hints. Frontend: `PROJECT_META`, `openProject()`, sidebar PROJECTS section, projects welcome card, difficulty badges.

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

### TODO 2 — Additional Modules ✅ COMPLETE

Modules 10–13 built: guides + scripts + Q&A + cheat sheets. See "COMPLETED MODULES" section above.

### TODO 3 — Content Polish ✅ COMPLETE

- `[x]` **Q&A banks for Modules 01–04** — 8-question senior-level Q&A added to each
- `[x]` **Cheat Sheet tables** — added to all modules (key formulas, trade-offs, code patterns)
- `[x]` **Module 02 — `decision_tree.py`** — CART from scratch, Gini/entropy, pruning, MDI
- `[x]` **Module 03 — Redis caching section** — LRU, TTL, pipeline batching, connection pooling
- `[x]` **Module 04 — gRPC section** — protobuf schema, server/client, streaming, REST vs gRPC
- `[x]` **Module 05 — `onnx_export.py`** — ONNX graph anatomy, NumPy MLP + sklearn → ONNX
- `[x]` **Module 08 — Reranking section** — cross-encoder, ColBERT late interaction, two-stage table

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
| "Fix X in a module" | Read the specific module file first, make targeted edits, commit |
| "Push to GitHub" | `git push origin main` — confirm with user first since it affects the live site |
| "Add content to Module NN" | Read `docs/NN-*.md` first, then edit; commit immediately after |
| "What's left to do?" | Platform is fully built — all 13 modules, 9 projects, and content polish complete |
| "Add a new module" | Follow the standard: `docs/NN-topic.md` + `src/NN-*/` scripts + update `docs/list.md`, `app.js` MODULE_META, and this file |

---

*Last updated: 2026-05-11*
*ALL TODOs COMPLETE: 13 modules + 9 projects + content polish + UI revamp*
*Modules 01–13: guides + scripts + Q&A banks + cheat sheets*
*Projects: 9 guides in docs/projects/ + frontend (app.js, index.html, style.css)*
*Content polish: DT section (02), Redis (03), gRPC (04), ONNX export (05), reranking (08)*
*UI revamp: loading screen, greeting bar, smooth transitions, dark mode neo-brutalism, tabbed sidebar*
*Open: nothing — platform is fully built*
