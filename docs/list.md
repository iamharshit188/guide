# AI/ML Learning Platform — Curriculum Roadmap

## Status Legend
| Symbol | Meaning |
|--------|---------|
| `[ ]` | Not Started |
| `[~]` | In Progress |
| `[x]` | Completed |

---

## Module 01 — Math for ML
**Status:** `[x]` Completed
**Guide:** [docs/01-math.md](01-math.md)
**Code:** `src/01-math/`

| Script | Covers |
|--------|--------|
| `src/01-math/vectors.py` | Vector ops, norms, dot product, cosine similarity, projections |
| `src/01-math/matrix_ops.py` | Matrix multiply, inverse, rank, eigendecomp, SVD |
| `src/01-math/calculus_demo.py` | Partial derivatives, numerical gradients, gradient descent |
| `src/01-math/probability.py` | Distributions, Bayes, MLE, cross-entropy, KL divergence |

### Checklist
- [ ] Vectors, norms, dot product
- [ ] Matrix operations, transpose, inverse
- [ ] Eigenvalues & Eigenvectors
- [ ] SVD (Singular Value Decomposition)
- [ ] Partial derivatives, chain rule
- [ ] Gradient descent derivation
- [ ] Probability axioms, Bayes theorem
- [ ] Key distributions (Gaussian, Bernoulli, Categorical)
- [ ] MLE, cross-entropy, KL divergence

---

## Module 02 — ML Basics to Advanced
**Status:** `[x]` Completed
**Guide:** `docs/02-ml-basics.md`
**Code:** `src/02-ml/`

| Script | Covers |
|--------|--------|
| `src/02-ml/linear_regression.py` | OLS, gradient descent, regularization (L1/L2) |
| `src/02-ml/logistic_regression.py` | Sigmoid, binary cross-entropy, decision boundary |
| `src/02-ml/evaluation.py` | Bias-variance, cross-val, ROC-AUC, precision-recall |
| `src/02-ml/clustering.py` | K-Means, DBSCAN, elbow method |
| `src/02-ml/pca.py` | PCA from scratch via covariance eigendecomp |
| `src/02-ml/random_forest.py` | Bagging, feature importance, OOB error |
| `src/02-ml/svm.py` | Kernel trick, dual form, soft margin |
| `src/02-ml/gradient_boosting.py` | XGBoost math, residuals, shrinkage |
| `src/02-ml/decision_tree.py` | CART from scratch, Gini/entropy, pruning, feature importance |

### Checklist
- [ ] Linear & Logistic Regression
- [ ] Bias-Variance tradeoff, cross-validation
- [ ] K-Means, PCA
- [ ] Decision Trees (Gini, entropy, pruning)
- [ ] Random Forests (bagging, feature importance)
- [ ] SVMs (kernel trick, dual formulation)
- [ ] Gradient Boosting (XGBoost mathematical derivation)

---

## Module 03 — Databases & Vector DBs
**Status:** `[x]` Completed
**Guide:** `docs/03-databases.md`
**Code:** `src/03-databases/`

| Script | Covers |
|--------|--------|
| `src/03-databases/sql_basics.py` | SQLite queries, joins, indexes, EXPLAIN |
| `src/03-databases/nosql_patterns.py` | MongoDB-style document ops via pymongo |
| `src/03-databases/chroma_demo.py` | ChromaDB CRUD, metadata filtering, similarity search |
| `src/03-databases/pinecone_demo.py` | Pinecone upsert, namespaces, hybrid search |
| `src/03-databases/faiss_demo.py` | FAISS index types, HNSW, IVF, benchmarks |

### Checklist
- [ ] SQL joins, indexes, query optimization, EXPLAIN plans
- [ ] NoSQL patterns, CAP theorem
- [ ] Embedding fundamentals for retrieval
- [ ] ChromaDB: setup, CRUD, metadata filtering, distance functions
- [ ] Pinecone: namespaces, sparse-dense hybrid search
- [ ] HNSW & FAISS internals (graph traversal, quantization)

---

## Module 04 — Backend with Flask
**Status:** `[x]` Completed
**Guide:** `docs/04-backend.md`
**Code:** `src/04-backend/`

| Script | Covers |
|--------|--------|
| `src/04-backend/app.py` | Flask app factory, blueprints, error handlers |
| `src/04-backend/ml_serving.py` | Load sklearn/torch model, predict endpoint |
| `src/04-backend/middleware.py` | Auth, rate limiting, request logging |
| `src/04-backend/async_tasks.py` | Celery + Redis task queue |

### Checklist
- [ ] Flask routing, blueprints, app factory pattern
- [ ] RESTful API design (versioning, status codes, error formats)
- [ ] Serving ML models (sklearn pickle, torch scripted)
- [ ] Request validation (Marshmallow / Pydantic)
- [ ] Async task queue with Celery + Redis

---

## Module 05 — Deep Learning & MLOps
**Status:** `[x]` Completed
**Guide:** `docs/05-deep-learning.md`
**Code:** `src/05-deep-learning/`

| Script | Covers |
|--------|--------|
| `src/05-deep-learning/nn_numpy.py` | 2-layer NN, forward/backward pass, pure NumPy |
| `src/05-deep-learning/optimizers.py` | SGD, Momentum, Adam from scratch |
| `src/05-deep-learning/mlflow_demo.py` | Experiment tracking, model registry, artifact logging |
| `src/05-deep-learning/docker_serve/` | Dockerfile + Flask model server |
| `src/05-deep-learning/monitoring.py` | Data drift detection with Evidently |
| `src/05-deep-learning/onnx_export.py` | ONNX graph structure, sklearn Pipeline → ONNX, NumPy MLP → ONNX, onnxruntime inference |

### Checklist
- [ ] Neural networks from scratch (NumPy)
- [ ] Backpropagation full derivation
- [ ] Optimizers: SGD, Momentum, RMSProp, Adam
- [ ] MLflow: tracking, model registry, artifact storage
- [ ] Docker containerization for model serving
- [ ] Data drift & concept drift monitoring

---

## Module 06 — GenAI Core
**Status:** `[x]` Completed
**Guide:** `docs/06-genai-core.md`
**Code:** `src/06-genai/`

| Script | Covers |
|--------|--------|
| `src/06-genai/word2vec.py` | Skip-gram, negative sampling, training loop |
| `src/06-genai/sentence_transformers_demo.py` | Sentence embeddings, semantic search |
| `src/06-genai/attention.py` | Scaled dot-product attention from scratch |
| `src/06-genai/multihead_attention.py` | MHA with projection matrices, NumPy |
| `src/06-genai/positional_encoding.py` | Sinusoidal PE, learned PE |
| `src/06-genai/kv_cache.py` | KV cache simulation, memory analysis |

### Checklist
- [ ] Word embeddings: Word2Vec skip-gram, CBOW
- [ ] Sentence Transformers and semantic similarity
- [ ] Scaled dot-product attention (mathematical derivation)
- [ ] Multi-head attention
- [ ] Positional encoding (sinusoidal, learned)
- [ ] KV Cache mechanics and memory math

---

## Module 07 — Transformers from Scratch
**Status:** `[x]` Completed
**Guide:** `docs/07-transformers.md`
**Code:** `src/07-transformer/`

| Script | Covers |
|--------|--------|
| `src/07-transformer/tokenizer.py` | BPE tokenizer from scratch |
| `src/07-transformer/model.py` | Full Transformer encoder-decoder, PyTorch |
| `src/07-transformer/model_numpy.py` | Inference-only Transformer, pure NumPy |
| `src/07-transformer/model.cpp` | Transformer forward pass in C++ |
| `src/07-transformer/train.py` | Training loop, LR scheduling, checkpointing |

### Checklist
- [ ] BPE Tokenization
- [ ] Encoder block (MHA + FFN + LayerNorm + residuals)
- [ ] Decoder block (masked MHA + cross-attention)
- [ ] Full Transformer architecture (PyTorch)
- [ ] Transformer in NumPy (inference)
- [ ] Transformer forward pass in C++
- [ ] Training loop with LR warmup

---

## Module 08 — RAG Chatbot
**Status:** `[x]` Completed
**Guide:** `docs/08-rag.md`
**Code:** `src/08-rag/`

| Script | Covers |
|--------|--------|
| `src/08-rag/ingest.py` | PDF/text loader, chunking strategies |
| `src/08-rag/embed_store.py` | Embed chunks, store in ChromaDB |
| `src/08-rag/retriever.py` | Similarity + MMR + hybrid retrieval |
| `src/08-rag/generator.py` | Context injection, prompt template, LLM call |
| `src/08-rag/app.py` | Full RAG pipeline Flask API |
| `src/08-rag/evaluate.py` | RAGAS: faithfulness, answer relevancy, context recall |

### Checklist
- [ ] Document ingestion (PDF, TXT, web)
- [ ] Chunking: fixed-size, recursive, semantic
- [ ] Embedding + vector store ingestion
- [ ] Retrieval: cosine similarity, MMR, hybrid BM25+dense
- [ ] Prompt construction and context injection
- [ ] End-to-end RAG pipeline
- [ ] Evaluation with RAGAS

---

## Module 09 — Fine-Tuning
**Status:** `[x]` Completed
**Guide:** `docs/09-finetuning.md`
**Code:** `src/09-finetuning/`

| Script | Covers |
|--------|--------|
| `src/09-finetuning/lora_theory.py` | LoRA rank decomposition demo, parameter count math |
| `src/09-finetuning/prepare_dataset.py` | Instruction format, train/val split, tokenization |
| `src/09-finetuning/train_lora.py` | PEFT + SFTTrainer + LoRA config |
| `src/09-finetuning/train_qlora.py` | 4-bit BitsAndBytes + QLoRA full pipeline |
| `src/09-finetuning/evaluate.py` | Perplexity, BLEU, ROUGE evaluation |
| `src/09-finetuning/merge_push.py` | Merge LoRA adapters, push to Hub |

### Checklist
- [x] Full fine-tuning vs PEFT (parameter count analysis)
- [x] LoRA: mathematical derivation (W = W₀ + BA, rank selection)
- [x] QLoRA: NF4 quantization + double quantization
- [x] HuggingFace PEFT library configuration
- [x] Dataset preparation: instruction format, chat template
- [x] Training with trl SFTTrainer
- [x] Evaluation: perplexity, BLEU, ROUGE, human eval
- [x] Adapter merging and Hub deployment

---

## Module 10 — LLM Agents & Tool Use
**Status:** `[x]` Completed
**Guide:** `docs/10-agents.md`
**Code:** `src/10-agents/`

| Script | Covers |
|--------|--------|
| `src/10-agents/react_agent.py` | ReAct loop, tool dispatch, error recovery, self-consistency |
| `src/10-agents/tool_calling.py` | JSON Schema tool registry, parallel dispatch, multi-turn loop |
| `src/10-agents/agent_memory.py` | Buffer, summary, entity, combined memory systems |
| `src/10-agents/agent_eval.py` | Trajectory accuracy, tool F1, answer F1, benchmark suite |

### Checklist
- [x] ReAct: Thought/Action/Observation loop
- [x] Tool schemas (OpenAI function calling format)
- [x] Chain-of-thought (zero-shot, few-shot, self-consistency)
- [x] Multi-step planning (plan-then-execute, hierarchical)
- [x] Agent memory: buffer, summary, entity
- [x] Error recovery with observation injection
- [x] Agent evaluation: trajectory accuracy, tool F1, answer F1

---

## Module 11 — Deployment & Production ML
**Status:** `[x]` Completed
**Guide:** `docs/11-deployment.md`
**Code:** `src/11-deployment/`

| Script | Covers |
|--------|--------|
| `src/11-deployment/onnx_export.py` | sklearn→ONNX, NumPy MLP serialization, onnxruntime inference, FP16/INT8 quantization |
| `src/11-deployment/quantize.py` | FP16/INT8/INT4/NF4 quantization from scratch, per-channel, static vs dynamic |
| `src/11-deployment/ab_serving.py` | A/B router, canary controller, shadow mode, sticky sessions, rollback |
| `src/11-deployment/health_check.py` | /health /ready /metrics, circuit breaker, graceful shutdown, latency tracker |

### Checklist
- [x] Docker multi-stage build + docker-compose + nginx config
- [x] ONNX export (sklearn, NumPy) + onnxruntime inference
- [x] TorchScript trace vs. script
- [x] Quantization: FP16, INT8 (symmetric/asymmetric/per-channel), NF4
- [x] Gunicorn worker models and configuration
- [x] A/B serving: weighted routing, canary ramp-up, rollback triggers
- [x] Health/readiness/metrics endpoints, circuit breaker, graceful shutdown

---

## Module 12 — RLHF & Alignment
**Status:** `[x]` Completed
**Guide:** `docs/12-rlhf.md`
**Code:** `src/12-rlhf/`

| Script | Covers |
|--------|--------|
| `src/12-rlhf/reward_model.py` | Bradley-Terry loss, RM training, margin loss, length bias, normalization |
| `src/12-rlhf/ppo_scratch.py` | PPO clip objective, GAE, KL penalty, value function, tabular MDP demo |
| `src/12-rlhf/dpo.py` | DPO loss derivation, implicit reward, IPO/KTO/SimPO variants, training sim |
| `src/12-rlhf/evaluate_alignment.py` | Win-rate (bootstrap CI), RM accuracy, MT-Bench sim, CAI, reward hacking |

### Checklist
- [x] Reward modeling: Bradley-Terry, margin loss, length bias, normalization
- [x] PPO: clip objective, GAE, value function, KL penalty, entropy bonus
- [x] DPO: derivation from optimal RLHF solution, implicit reward, training
- [x] DPO variants: IPO, KTO, SimPO
- [x] Constitutional AI: critique-revision loop
- [x] Evaluation: win-rate, RM accuracy, MT-Bench, reward hacking detection

---

## Module 13 — Multimodal Models
**Status:** `[x]` Completed
**Guide:** `docs/13-multimodal.md`
**Code:** `src/13-multimodal/`

| Script | Covers |
|--------|--------|
| `src/13-multimodal/clip_scratch.py` | CLIP InfoNCE loss, temperature scaling, contrastive training, retrieval evaluation |
| `src/13-multimodal/vit_patch.py` | Patch extraction, linear projection, [CLS] token, positional encoding, ViT forward pass |
| `src/13-multimodal/captioning.py` | Cross-modal attention, gated cross-attention (Flamingo), Q-Former, teacher forcing, beam search |
| `src/13-multimodal/zero_shot.py` | Zero-shot classification, prompt engineering, ensemble embeddings, temperature sensitivity |

### Checklist
- [x] CLIP: InfoNCE loss derivation, temperature τ, contrastive training
- [x] ViT: patch extraction formula, [CLS] token, positional encoding (sinusoidal + learned)
- [x] Multi-head self-attention in ViT encoder
- [x] Cross-modal attention (Q from text, K/V from image)
- [x] Gated cross-attention (Flamingo, tanh(α) init=0)
- [x] Q-Former: compressed visual prompt for frozen LLM (BLIP-2)
- [x] Image captioning: teacher forcing + autoregressive inference + beam search
- [x] Zero-shot classification: ŷ = argmax_c cos(v, t_c)
- [x] Prompt engineering + ensemble text embeddings

---

*Last updated: Content polish complete — Q&A banks, cheat sheets, DT section, Redis, gRPC, reranking, ONNX export*
