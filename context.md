# PROJECT CONTEXT — AI/ML Learning Platform
> **SINGLE SOURCE OF TRUTH. Read entirely before every session.**

## 1. ORIENTATION & STATUS
**What:** Self-hosted AI/ML placement prep platform targeting senior roles.
**Where:** Local: `/Users/iamharshit188/Desktop/placement/prepration/guide/` | Live: `https://harshittiwari.me/guide/`
**Status:** Fully built! 14 Modules, 16 Projects, Code Area, Dark Neo-Brutalism UI, Content Polish (Q&A/Cheat sheets/Beginner Basics).
**Boot Sequence:** 1. Read this file. 2. `git log --oneline -5`. 3. Ask user for task. 4. Wait for confirmation.

## 2. REPOSITORY STRUCTURE
- `index.html`, `style.css`, `app.js`: Frontend SPA (Dark Neo-brutalism, static GH Pages compatible).
- `server.py`: Local Flask dev server (port 3000).
- `docs/`: Markdown guides (`01-math.md` to `13-multimodal.md`), `list.md` (roadmap), `projects/` (p01 to p09).
- `src/`: Runnable Python/C++ scripts matching modules (`src/01-math/` to `src/13-multimodal/`).

## 3. CURRICULUM & MODULE DETAILED SPECS
*Strategy: Start with intuitive basics and real-world examples for beginners, then escalate to rigorous math/derivations. Every concept must answer "Why do we need this?" before showing the math.*
*Guides: Intuition-first, Math-backed derivations, Code equivalents, Q&A prep. Scripts: `if __name__ == "__main__": main()`, no `matplotlib.show()`.*

- **01 Math:** Intuition (GPS, Tables, Mountains) -> Rigor (LinAlg, Calculus, Probability, $A = V\Lambda V^{-1}$, MLE).
- **02 ML Basics:** Intuition (Curve fitting, voting) -> Rigor (OLS, BCE, Bias-Var, ROC, CV, SVM, Boosting).
- **03 Databases:** Intuition (Filing cabinets, libraries) -> Rigor (B+ tree, HNSW, IVF+PQ, PACELC).
- **04 Backend:** Intuition (Restaurant kitchen) -> Rigor (Flask/FastAPI, WSGI, Thread-local, JWT/OAuth, Supabase/Firebase Auth Integration, REST vs gRPC, Rate limits, Celery).
- **05 Deep Learning:** Intuition (Brain circuits, trial & error) -> Rigor (Backprop $\delta$, Xavier/He Init, AdamW, ONNX).
- **06 GenAI Core:** Intuition (Word meaning by context) -> Rigor (Skip-gram, Attention $\text{softmax}(QK^T/\sqrt{d_k})V$, RoPE, KV Cache).
- **07 Transformers:** Intuition (Translators taking sequential notes) -> Rigor (BPE, $12Ld^2$ tracking, Beam search).
- **08 RAG:** Intuition (Open-book exams) -> Rigor (TF-IDF, BM25, Hybrid RRF, MMR, ColBERT).
- **09 Fine-Tuning:** Intuition (Specializing a generalist) -> Rigor (LoRA, QLoRA NF4, Dataset masks, PPL).
- **10 Agents:** Intuition (Delegating tasks to departments) -> Rigor (ReAct loop, JSON schemas, CoT, Hierarchical Memory).
- **11 Deployment:** Intuition (Shipping a product) -> Rigor (Docker, ONNX export, INT8 Quantization, A/B Serving).
- **12 RLHF:** Intuition (Training a dog with treats) -> Rigor (Bradley-Terry, PPO, DPO, MT-Bench).
- **13 Multimodal:** Intuition (Connecting eyes and ears) -> Rigor (CLIP InfoNCE, ViT patches, Cross-modal attention).
- **14 Frontend (React + Tailwind):** Intuition (Building UI blocks) -> Rigor (JS ES6 foundations, React Hooks, State Management, Tailwind utility classes).

## 4. PROJECTS SECTION
16 individual project guides mapped meticulously to difficulty levels.
- (p01) PCA Compressor (Starter)
- (p02) Titanic Pipeline (Starter)
- (p03) Semantic Search Engine (Intermediate)
- (p04) Production ML API (Intermediate)
- (p05) Training Dashboard (Intermediate)
- (p06) Word Analogy Explorer (Intermediate)
- (p07) Shakespeare GPT (Advanced)
- (p08) Document Q&A (Advanced)
- (p09) Domain Tuner (Expert)
- (p10) Crypto Analyst Agent (Expert)
- (p11) A/B Testing Deploy API (Advanced)
- (p12) Mini-RLHF Tuner (Expert)
- (p13) Image-Text Hybrid Search (Expert)
- (p14) Full-Stack AI Blog (React, Tailwind, Supabase, Firebase, Flask) (Advanced)
- (p15) Real-Time Voice Assistant (Expert)
- (p16) MLOps Feature Store (Expert)

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
