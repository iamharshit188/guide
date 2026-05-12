# AI/ML Placement Preparation Guide

A self-hosted, comprehensive learning platform for AI and ML engineering roles. Covers the complete stack — from linear algebra through multimodal models — with every concept derived from first principles, every line of code annotated, and every module ending in a standalone mini-project.

Live at [harshittiwari.me/guide](https://harshittiwari.me/guide)

---

## Tech Stack

<p>
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python_3.14-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://numpy.org"><img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://flask.palletsprojects.com"><img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask"></a>
  <a href="https://react.dev"><img src="https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB" alt="React"></a>
  <a href="https://tailwindcss.com"><img src="https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white" alt="Tailwind CSS"></a>
  <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript"><img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black" alt="JavaScript"></a>
  <a href="https://www.markdownguide.org"><img src="https://img.shields.io/badge/Markdown-000000?style=flat-square&logo=markdown&logoColor=white" alt="Markdown"></a>
</p>

---

## Curriculum

14 modules. Each module follows the same structure: intuition, rigorous math derivations, annotated code from scratch, interview Q&A, cheat sheet, and a standalone mini-project.

### Module 01 — Math for ML

<a href="https://numpy.org"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" height="18" alt="NumPy"> NumPy</a>

Linear algebra (vectors, matrices, SVD, eigendecomposition), multivariate calculus (chain rule, Jacobians, gradient descent), and probability (MLE, KL divergence, Bayes theorem). Every identity proven before being applied.

**Mini-project:** Grade predictor via gradient descent from scratch.

---

### Module 02 — ML Basics to Advanced

<a href="https://scikit-learn.org"><img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" height="18" alt="scikit-learn"> scikit-learn</a>

Linear and logistic regression, decision trees (CART from scratch), random forests, SVM (kernel trick), gradient boosting (XGBoost math), K-Means, PCA. Bias-variance decomposition, cross-validation, ROC-AUC.

**Mini-project:** Titanic survival predictor — five algorithms benchmarked side by side.

---

### Module 03 — Databases

<a href="https://www.postgresql.org"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/postgresql/postgresql-original.svg" height="18" alt="PostgreSQL"> PostgreSQL</a> &nbsp;
<a href="https://redis.io"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/redis/redis-original.svg" height="18" alt="Redis"> Redis</a> &nbsp;
<a href="https://www.trychroma.com"><img src="https://img.shields.io/badge/ChromaDB-FF6B35?style=flat-square" alt="ChromaDB"> ChromaDB</a> &nbsp;
<a href="https://github.com/facebookresearch/faiss"><img src="https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white" alt="FAISS"> FAISS</a>

SQL (B+ trees, ACID, indexing strategies), NoSQL (CAP theorem, document stores, key-value stores), vector databases (HNSW graph traversal, IVF clustering, product quantization). ChromaDB and FAISS usage.

**Mini-project:** Movie recommender combining SQL metadata with FAISS vector search.

---

### Module 04 — Backend Engineering

<a href="https://flask.palletsprojects.com"><img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask"> Flask</a> &nbsp;
<a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI"> FastAPI</a> &nbsp;
<a href="https://jwt.io"><img src="https://img.shields.io/badge/JWT-000000?style=flat-square&logo=jsonwebtokens&logoColor=white" alt="JWT"> JWT</a>

REST API design (WSGI vs ASGI, blueprints, app factory), Pydantic validation, JWT authentication from scratch, rate limiting (token bucket algorithm), ML model serving patterns.

**Mini-project:** Production ML serving API with sentiment classifier, JWT auth, and rate limiting.

---

### Module 05 — Deep Learning

<a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"> PyTorch</a> &nbsp;
<a href="https://numpy.org"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" height="18" alt="NumPy"> NumPy</a>

Backpropagation derived via chain rule, Xavier/He initialization, SGD / Momentum / Adam / AdamW from scratch, BatchNorm and LayerNorm with full forward-backward passes, dropout, residual connections.

**Mini-project:** MNIST digit classifier — neural network built entirely from NumPy.

---

### Module 06 — GenAI Core

<a href="https://huggingface.co"><img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face"> Hugging Face</a>

Word2Vec skip-gram with negative sampling, scaled dot-product attention ($\text{softmax}(QK^\top/\sqrt{d_k})V$), multi-head attention, sinusoidal positional encoding, Rotary Position Embedding (RoPE), KV cache for inference.

**Mini-project:** Semantic text similarity engine.

---

### Module 07 — Transformers

<a href="https://huggingface.co"><img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face"> Hugging Face</a> &nbsp;
<a href="https://openai.com"><img src="https://img.shields.io/badge/GPT_architecture-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI"> GPT architecture</a>

BPE tokenizer from scratch (merge rules, encode/decode), encoder and decoder blocks with residual connections, full encoder-decoder Transformer, parameter counting for GPT-2 and BERT scale, warmup learning rate schedule.

**Mini-project:** Character-level language model with autoregressive generation.

---

### Module 08 — RAG

<a href="https://github.com/facebookresearch/faiss"><img src="https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white" alt="FAISS"> FAISS</a> &nbsp;
<a href="https://www.trychroma.com"><img src="https://img.shields.io/badge/ChromaDB-FF6B35?style=flat-square" alt="ChromaDB"> ChromaDB</a> &nbsp;
<a href="https://python.langchain.com"><img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white" alt="LangChain"> LangChain</a>

Fixed / sentence / semantic chunking strategies, TF-IDF and BM25 from scratch, LSH and flat vector indexes, hybrid retrieval via Reciprocal Rank Fusion, MMR diversity selection, prompt construction with citation, RAGAS evaluation (context precision, context recall).

**Mini-project:** Document Q&A system over an AI/ML knowledge base with self-evaluation.

---

### Module 09 — Fine-Tuning

<a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"> PyTorch</a> &nbsp;
<a href="https://huggingface.co/docs/peft"><img src="https://img.shields.io/badge/PEFT-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="PEFT"> PEFT / LoRA</a>

Full fine-tuning vs PEFT, LoRA rank decomposition ($\Delta W = BA$) with forward and backward pass, NF4 quantization block-wise from scratch, QLoRA memory math, instruction tuning with loss masking on response tokens, Alpaca and ChatML prompt templates.

**Mini-project:** LoRA fine-tuning simulator comparing frozen baseline / LoRA / full fine-tuning.

---

### Module 10 — LLM Agents

<a href="https://python.langchain.com"><img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white" alt="LangChain"> LangChain</a> &nbsp;
<a href="https://openai.com"><img src="https://img.shields.io/badge/Function_Calling-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI Function Calling"> Function Calling</a>

ReAct (Reason+Act) loop with Thought/Action/Observation parsing, structured JSON function calling with schema validation, sliding window and semantic long-term memory, multi-agent supervisor pattern, chain-of-thought, self-consistency voting.

**Mini-project:** Multi-turn research assistant with search, calculator, and memory.

---

### Module 11 — Deployment & Production ML

<a href="https://www.docker.com"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg" height="18" alt="Docker"> Docker</a> &nbsp;
<a href="https://onnx.ai"><img src="https://img.shields.io/badge/ONNX-005CED?style=flat-square&logo=onnx&logoColor=white" alt="ONNX"> ONNX</a>

Model serialization (JSON, binary, safetensors), dynamic batching with threading queue, INT8 quantization from scratch, MLServer with input validation and structured error handling, PSI-based data drift detection, A/B testing with z-test significance, pre-deployment validation checklist.

**Mini-project:** ProductionService integrating all components with live drift detection.

---

### Module 12 — RLHF & Alignment

<a href="https://openai.com"><img src="https://img.shields.io/badge/PPO-412991?style=flat-square&logo=openai&logoColor=white" alt="PPO"> PPO</a> &nbsp;
<a href="https://arxiv.org/abs/2305.18290"><img src="https://img.shields.io/badge/DPO-B31B1B?style=flat-square" alt="DPO"> DPO</a>

Reward model training via Bradley-Terry pairwise loss, PPO with KL-penalized RLHF objective and clipped surrogate, DPO margin derivation eliminating the separate reward model, Constitutional AI critique-and-revise loop, reward hacking and overoptimization analysis.

**Mini-project:** Preference learning pipeline — collect comparisons, train reward model, train policy via DPO, measure Spearman rank correlation.

---

### Module 13 — Multimodal Models

<a href="https://openai.com/research/clip"><img src="https://img.shields.io/badge/CLIP-412991?style=flat-square&logo=openai&logoColor=white" alt="CLIP"> CLIP</a> &nbsp;
<a href="https://arxiv.org/abs/2010.11929"><img src="https://img.shields.io/badge/ViT-4285F4?style=flat-square&logo=google&logoColor=white" alt="ViT"> ViT</a>

Patch embedding and positional encoding, Vision Transformer encoder (GELU, pre-norm blocks, CLS token), CLIP InfoNCE contrastive loss with temperature scaling, zero-shot image classification, text-to-image retrieval, cross-modal attention for VQA, greedy image captioning.

**Mini-project:** Semantic image search engine with tag-based recall evaluation and similarity matrix.

---

### Module 14 — Frontend Engineering

<a href="https://react.dev"><img src="https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB" alt="React"> React</a> &nbsp;
<a href="https://nextjs.org"><img src="https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=nextdotjs&logoColor=white" alt="Next.js"> Next.js</a> &nbsp;
<a href="https://tailwindcss.com"><img src="https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white" alt="Tailwind CSS"> Tailwind CSS</a> &nbsp;
<a href="https://www.typescriptlang.org"><img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white" alt="TypeScript"> TypeScript</a>

JavaScript fundamentals through advanced (closures, event loop, Promises), React hooks and lifecycle, state management (Zustand, Context API), Next.js App Router and Server Components, authentication, deployment, performance optimization.

**Mini-project:** Full-stack AI blog with React frontend, Flask backend, and Supabase auth.

---

## Projects

16 standalone project guides in `docs/projects/`, graduated by difficulty.

| Level | Projects |
|---|---|
| Starter | PCA Compressor, Titanic Pipeline |
| Intermediate | Semantic Search Engine, Production ML API, Training Dashboard, Word Analogy Explorer |
| Advanced | Shakespeare GPT, Document Q&A, A/B Testing Deploy API, Full-Stack AI Blog |
| Expert | Domain Tuner, Crypto Analyst Agent, Mini-RLHF Tuner, Image-Text Hybrid Search, Real-Time Voice Assistant, MLOps Feature Store |

---

## Running Locally

```bash
# Clone
git clone https://github.com/iamharshit188/placement-prepration-guide.git
cd placement-prepration-guide

# Start the local server (Python 3.14)
python3.14 server.py
# Opens at http://localhost:3000
```

The platform is a static SPA served by a minimal Flask dev server. No build step required. All content is Markdown rendered client-side via [marked.js](https://marked.js.org), math via [MathJax](https://www.mathjax.org), and syntax highlighting via [highlight.js](https://highlightjs.org).

---

## Repository Structure

```
.
├── index.html          # SPA shell — Dark Neo-Brutalism UI
├── style.css           # Space Grotesk + JetBrains Mono, electric yellow accents
├── app.js              # Module/project routing, progress tracking (localStorage)
├── server.py           # Flask dev server (port 3000)
├── docs/
│   ├── list.md         # Curriculum roadmap and completion status
│   ├── 01-math.md      # Module guides (01 – 14)
│   └── projects/       # p01 – p16 project guides
└── src/
    ├── 01-math/        # Runnable Python scripts per module
    └── ...
```

---

## Design

Dark Neo-Brutalism. Pure black backgrounds (`#050505`, `#121212`), electric yellow accents, sharp borders. Fonts: [Space Grotesk](https://fonts.google.com/specimen/Space+Grotesk) for headings, [Outfit](https://fonts.google.com/specimen/Outfit) for body, [JetBrains Mono](https://www.jetbrains.com/lp/mono/) for code.

---

## Writing Philosophy

Every module follows: **Intuition** (real-world analogy) → **Math** (derived from first principles) → **Code** (every line annotated) → **Interview Q&A** → **Cheat sheet** → **Mini-project**.

No fluff. No handwaving. If a formula is used, it is derived. If a class is written, every method is explained. Target reader: someone preparing for senior ML engineering interviews who wants to understand, not just memorize.
