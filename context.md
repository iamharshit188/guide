# PROJECT CONTEXT — AI/ML Learning Platform
> **SINGLE SOURCE OF TRUTH. Read entirely before every session.**

## 1. ORIENTATION & STATUS
**What:** Self-hosted AI/ML placement prep platform targeting senior roles.
**Where:** Local: `/Users/iamharshit188/Desktop/placement/prepration/guide/` | Live: `https://harshittiwari.me/guide/`
**Status:** Fully built! 13 Modules, 9+ Projects, Code Area, Dark Neo-Brutalism UI, Content Polish (Q&A/Cheat sheets/Beginner Basics).
**Boot Sequence:** 1. Read this file. 2. `git log --oneline -5`. 3. Ask user for task. 4. Wait for confirmation.

## 2. REPOSITORY STRUCTURE
- `index.html`, `style.css`, `app.js`: Frontend SPA (Dark Neo-brutalism, static GH Pages compatible).
- `server.py`: Local Flask dev server (port 3000).
- `docs/`: Markdown guides (`01-math.md` to `13-multimodal.md`), `list.md` (roadmap), `projects/` (p01 to p09).
- `src/`: Runnable Python/C++ scripts matching modules (`src/01-math/` to `src/13-multimodal/`).

## 3. CURRICULUM & MODULE DETAILED SPECS
*Strategy: Start with intuitive basics and real-world examples for beginners, then escalate to rigorous math/derivations.*
*Guides: Math-first derivations, Code equivalents, Q&A prep, Cheat sheets, Resources. Scripts: `if __name__ == "__main__": main()`, no `matplotlib.show()`, `try/except` for heavy deps.*
- **01 Math:** `src/01-math/`. LinAlg (L1/L2, cosine, $A = V\Lambda V^{-1}$, SVD PCA), Calculus (Jacobian, Hessian, GD), Prob (MLE, cross-entropy, KL divergence).
- **02 ML Basics:** `src/02-ml/`. Lin/Log Reg (OLS, BCE), Eval (Bias-Var, ROC, CV), Clustering, PCA, RF (Gini/Entropy, MDI, OOB), SVM (Mercer, hinge), Boosting (XGBoost 2nd-order).
- **03 Databases:** `src/03-databases/`. SQL (B+ tree, EXPLAIN, CTEs), NoSQL (PACELC), Vector DBs (HNSW, IVF+PQ, Chroma/FAISS), Redis cache (LRU, pipeline).
- **04 Backend:** `src/04-backend/`. Flask (WSGI, thread-local), App Factory, Blueprints, Thread-safe ML Serving, Middleware (Rate limits, HMAC), Celery async, gRPC.
- **05 Deep Learning:** `src/05-deep-learning/`. Backprop ($\delta$), Init (Xavier Var=$2/(n_{in}+n_{out})$, He), Optimizers (AdamW bias-correction), LR Schedulers, BatchNorm/LayerNorm, MLflow, Drift, ONNX graph anatomy.
- **06 GenAI Core:** `src/06-genai/`. Word2Vec (Skip-gram, Neg-sample), Attention ($	ext{softmax}(QK^T/\sqrt{d_k})V$), Positional Encoding (RoPE), KV Cache ($O(Td)$ size formula).
- **07 Transformers:** `src/07-transformer/`. BPE Tokenizer, Encode/Decode block ($12Ld^2$ tracking), Teacher forcing, Generation (Beam search, top-p/k).
- **08 RAG:** `src/08-rag/`. Chunking, TF-IDF from scratch, Retrieval (BM25, Hybrid RRF, MMR), Reranking (ColBERT late interaction), RAGAS Eval.
- **09 Fine-Tuning:** `src/09-finetuning/`. LoRA ($W'=W_0+rac{lpha}{r}BA$), QLoRA (NF4 double quant, 0.127 b/w overhead), Dataset masks, Merge adapter, PPL/ROUGE Eval.
- **10 Agents:** `src/10-agents/`. ReAct loop, JSON function tool schemas, CoT self-consistency, Hierarchical Memory (Buffer/Summary), Eval heuristics.
- **11 Deployment:** `src/11-deployment/`. Docker, ONNX export logic, Quantization (FP16/INT8/4), Gunicorn workers, A/B Serving controllers, Circuit breakers.
- **12 RLHF:** `src/12-rlhf/`. Bradley-Terry loss, PPO (GAE), DPO ($\mathcal{L}_{DPO}=-\mathbb{E}[\log\sigma(eta(r_w-r_l))]$ cancels partition function Z), IPO/KTO/SimPO, MT-Bench.
- **13 Multimodal:** `src/13-multimodal/`. CLIP (InfoNCE $L=-rac{1}{B}\sum\log\dots$), ViT patches ($N=(H/P)(W/P)$), Cross-modal attention, Q-Former, Zero-shot eval.

## 4. PROJECTS SECTION
9 guides in `docs/projects/` with frontend tracking (`PROJECT_META` in `app.js`). *Pseudocode/checkpoints only; no full solutions.*
- (p01) PCA Compressor, (p02) Titanic, (p03) Semantic Search, (p04) Production ML API, (p05) Training Dashboard, (p06) Word Analogy, (p07) Shakespeare GPT, (p08) Doc Q&A, (p09) Domain Tuner.

## 5. FRONTEND & UI (Dark Neo-Brutalism)
- **Tech:** marked.js, highlight.js, MathJax. `.md` link intercepts prevent GH Pages 404s.
- **Design:** Pure/Soft Blacks (`#050505`, `#121212`), Electric Yellow accents. `--gray-line` borders. Fonts: Space Grotesk, Outfit, JetBrains Mono.
- **State:** `localStorage` key `aiml_platform_progress` vs `aiml_platform_projects_progress`.
- **Features:** Loading screen, dynamic time/stat greeting, tabbed sidebar (MODULES/PROJECTS) via JS toggle, cubic-bezier spring transitions.

## 6. RULES & STYLE (NON-NEGOTIABLE)
1. **Check Status First:** Read `docs/list.md` before altering curriculum content.
2. **Confirm First:** Never start building/refactoring without user confirmation.
3. **Tone & Form:** Authoritative, math-first, tables > prose. Use LaTeX (`$\mathbf{w}$`, `$\mathcal{L}$`, `$\mathbb{R}$`).
4. **No AI Attribution:** Never mention AI generation, Claude, Anthropic, ChatGPT, etc.
5. **Python 3.14 Environment:** Scripts use graceful imports (`sys.exit(0)`) if missing wheels like PyTorch.
6. **Git Protocol:** Commit format: `type: short summary 
 - bullet`. Never `--amend` or `git add .` (add specific files).
7. **Complete Units Only:** Guides and corresponding scripts must be fully done and committed together.

## 7. USER REQUEST ACTIONS
- **"Fix X"**: Read the targeted file(s) first, edit carefully, commit directly.
- **"Push to GitHub"**: Execute `git push origin main` (Get user confirmation first!).
- **"Add new module"**: Scaffold `docs/NN-topic.md` + `src/NN-topic/`. Update `list.md`, `app.js` (META arrays), and this context file.
