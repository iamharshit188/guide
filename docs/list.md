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

### Checklist
- [ ] Neural networks from scratch (NumPy)
- [ ] Backpropagation full derivation
- [ ] Optimizers: SGD, Momentum, RMSProp, Adam
- [ ] MLflow: tracking, model registry, artifact storage
- [ ] Docker containerization for model serving
- [ ] Data drift & concept drift monitoring

---

## Module 06 — GenAI Core
**Status:** `[~]` In Progress
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
**Status:** `[ ]` Not Started
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
**Status:** `[ ]` Not Started
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
**Status:** `[ ]` Not Started
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
- [ ] Full fine-tuning vs PEFT (parameter count analysis)
- [ ] LoRA: mathematical derivation (W = W₀ + BA, rank selection)
- [ ] QLoRA: NF4 quantization + double quantization
- [ ] HuggingFace PEFT library configuration
- [ ] Dataset preparation: instruction format, chat template
- [ ] Training with trl SFTTrainer
- [ ] Evaluation: perplexity, BLEU, ROUGE, human eval
- [ ] Adapter merging and Hub deployment

---

*Last updated: Module 05 complete — Deep Learning & MLOps*
