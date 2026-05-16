"use strict";

// ── Constants ──────────────────────────────────────────────────
const LS_KEY          = "aiml_platform_progress";
const LS_KEY_PROJECTS = "aiml_platform_projects_progress";
const LOAD_START      = Date.now();
const MIN_LOAD_MS     = 1400;

const MODULE_META = [
  { file: "01-math.md",           label: "Math for ML",               tag: "01", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>` },
  { file: "02-ml-basics.md",      label: "ML Basics → Advanced",      tag: "02", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><path d="M6 21V9a9 9 0 0 0 9 9"/></svg>` },
  { file: "03-databases.md",      label: "Databases & Vector DBs",    tag: "03", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>` },
  { file: "04-backend.md",        label: "Backend with Flask",        tag: "04", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8m-4-4v4"/><path d="M7 8h10M7 12h6"/></svg>` },
  { file: "05-deep-learning.md",  label: "Deep Learning",             tag: "05", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><circle cx="3" cy="6" r="2"/><circle cx="3" cy="18" r="2"/><circle cx="21" cy="6" r="2"/><circle cx="21" cy="18" r="2"/><path d="M5 6h4m6 0h4M5 18h4m6 0h4M10 10l-5-3m9 3l5-3M10 14l-5 3m9-3l5 3"/></svg>` },
  { file: "06-genai-core.md",     label: "GenAI Core",                tag: "06", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><path d="M8 10h8M8 14h5"/></svg>` },
  { file: "07-transformers.md",   label: "Transformers from Scratch", tag: "07", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="5" height="5" rx="1"/><rect x="16" y="3" width="5" height="5" rx="1"/><rect x="3" y="16" width="5" height="5" rx="1"/><rect x="16" y="16" width="5" height="5" rx="1"/><path d="M8 5.5h8M5.5 8v8M18.5 8v8M8 18.5h8"/></svg>` },
  { file: "08-rag.md",            label: "RAG Systems",               tag: "08", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>` },
  { file: "09-finetuning.md",     label: "Fine-Tuning & LoRA",        tag: "09", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>` },
  { file: "10-agents.md",         label: "LLM Agents & Tool Use",     tag: "10", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a10 10 0 1 0 10 10"/><path d="M12 8v4l3 3"/><circle cx="18" cy="5" r="3"/><path d="M18 2v3h3"/></svg>` },
  { file: "11-deployment.md",     label: "Deployment & Production",   tag: "11", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/></svg>` },
  { file: "12-rlhf.md",           label: "RLHF & Alignment",          tag: "12", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>` },
  { file: "13-multimodal.md",     label: "Multimodal Models",         tag: "13", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>` },
  { file: "14-frontend.md",       label: "Frontend (React+Tailwind)", tag: "14", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/><line x1="19" y1="12" x2="5" y2="12"/></svg>` },
];

const CODE_META = [
  { file: "01-math/calculus_demo.py",          label: "calculus_demo.py",          module: "01" },
  { file: "01-math/matrix_ops.py",             label: "matrix_ops.py",             module: "01" },
  { file: "01-math/probability.py",            label: "probability.py",            module: "01" },
  { file: "01-math/vectors.py",                label: "vectors.py",                module: "01" },
  { file: "02-ml/clustering.py",               label: "clustering.py",             module: "02" },
  { file: "02-ml/decision_tree.py",            label: "decision_tree.py",          module: "02" },
  { file: "02-ml/evaluation.py",               label: "evaluation.py",             module: "02" },
  { file: "02-ml/gradient_boosting.py",        label: "gradient_boosting.py",      module: "02" },
  { file: "02-ml/linear_regression.py",        label: "linear_regression.py",      module: "02" },
  { file: "02-ml/logistic_regression.py",      label: "logistic_regression.py",    module: "02" },
  { file: "02-ml/pca.py",                      label: "pca.py",                    module: "02" },
  { file: "02-ml/random_forest.py",            label: "random_forest.py",          module: "02" },
  { file: "02-ml/svm.py",                      label: "svm.py",                    module: "02" },
  { file: "03-databases/chroma_demo.py",       label: "chroma_demo.py",            module: "03" },
  { file: "03-databases/faiss_demo.py",        label: "faiss_demo.py",             module: "03" },
  { file: "03-databases/nosql_patterns.py",    label: "nosql_patterns.py",         module: "03" },
  { file: "03-databases/pinecone_demo.py",     label: "pinecone_demo.py",          module: "03" },
  { file: "03-databases/sql_basics.py",        label: "sql_basics.py",             module: "03" },
  { file: "04-backend/app.py",                 label: "app.py",                    module: "04" },
  { file: "04-backend/async_tasks.py",         label: "async_tasks.py",            module: "04" },
  { file: "04-backend/middleware.py",          label: "middleware.py",             module: "04" },
  { file: "04-backend/ml_serving.py",          label: "ml_serving.py",             module: "04" },
  { file: "05-deep-learning/mlflow_demo.py",   label: "mlflow_demo.py",            module: "05" },
  { file: "05-deep-learning/monitoring.py",    label: "monitoring.py",             module: "05" },
  { file: "05-deep-learning/nn_numpy.py",      label: "nn_numpy.py",               module: "05" },
  { file: "05-deep-learning/onnx_export.py",   label: "onnx_export.py",            module: "05" },
  { file: "05-deep-learning/optimizers.py",    label: "optimizers.py",             module: "05" },
  { file: "06-genai/attention.py",             label: "attention.py",              module: "06" },
  { file: "06-genai/kv_cache.py",              label: "kv_cache.py",               module: "06" },
  { file: "06-genai/multihead_attention.py",   label: "multihead_attention.py",    module: "06" },
  { file: "06-genai/positional_encoding.py",   label: "positional_encoding.py",    module: "06" },
  { file: "06-genai/word2vec.py",              label: "word2vec.py",               module: "06" },
  { file: "07-transformer/model.cpp",          label: "model.cpp",                 module: "07" },
  { file: "07-transformer/model.py",           label: "model.py",                  module: "07" },
  { file: "07-transformer/model_numpy.py",     label: "model_numpy.py",            module: "07" },
  { file: "07-transformer/tokenizer.py",       label: "tokenizer.py",              module: "07" },
  { file: "07-transformer/train.py",           label: "train.py",                  module: "07" },
  { file: "08-rag/app.py",                     label: "app.py",                    module: "08" },
  { file: "08-rag/embed_store.py",             label: "embed_store.py",            module: "08" },
  { file: "08-rag/evaluate.py",                label: "evaluate.py",               module: "08" },
  { file: "08-rag/generator.py",               label: "generator.py",              module: "08" },
  { file: "08-rag/ingest.py",                  label: "ingest.py",                 module: "08" },
  { file: "08-rag/retriever.py",               label: "retriever.py",              module: "08" },
  { file: "09-finetuning/evaluate.py",         label: "evaluate.py",               module: "09" },
  { file: "09-finetuning/lora_theory.py",      label: "lora_theory.py",            module: "09" },
  { file: "09-finetuning/merge_push.py",       label: "merge_push.py",             module: "09" },
  { file: "09-finetuning/prepare_dataset.py",  label: "prepare_dataset.py",        module: "09" },
  { file: "09-finetuning/train_lora.py",       label: "train_lora.py",             module: "09" },
  { file: "09-finetuning/train_qlora.py",      label: "train_qlora.py",            module: "09" },
  { file: "10-agents/agent_eval.py",           label: "agent_eval.py",             module: "10" },
  { file: "10-agents/agent_memory.py",         label: "agent_memory.py",           module: "10" },
  { file: "10-agents/react_agent.py",          label: "react_agent.py",            module: "10" },
  { file: "10-agents/tool_calling.py",         label: "tool_calling.py",           module: "10" },
  { file: "11-deployment/ab_serving.py",       label: "ab_serving.py",             module: "11" },
  { file: "11-deployment/health_check.py",     label: "health_check.py",           module: "11" },
  { file: "11-deployment/onnx_export.py",      label: "onnx_export.py",            module: "11" },
  { file: "11-deployment/quantize.py",         label: "quantize.py",               module: "11" },
  { file: "12-rlhf/dpo.py",                    label: "dpo.py",                    module: "12" },
  { file: "12-rlhf/evaluate_alignment.py",     label: "evaluate_alignment.py",     module: "12" },
  { file: "12-rlhf/ppo_scratch.py",            label: "ppo_scratch.py",            module: "12" },
  { file: "12-rlhf/reward_model.py",           label: "reward_model.py",           module: "12" },
  { file: "13-multimodal/captioning.py",       label: "captioning.py",             module: "13" },
  { file: "13-multimodal/clip_scratch.py",     label: "clip_scratch.py",           module: "13" },
  { file: "13-multimodal/vit_patch.py",        label: "vit_patch.py",              module: "13" },
  { file: "13-multimodal/zero_shot.py",        label: "zero_shot.py",              module: "13" },
];

const LANGUAGE_META = [
  { file: "lang-c.md",      label: "C",          tag: "C",   badge: "lang-c",   desc: "Systems · Memory · Pointers",    icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M17 11.5a4.5 4.5 0 1 1 0 1"/><path d="M6 12a6 6 0 1 0 12 0 6 6 0 0 0-12 0z"/></svg>` },
  { file: "lang-cpp.md",    label: "C++",         tag: "C++", badge: "lang-cpp", desc: "OOP · RAII · Templates · STL",   icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M17 11.5a4.5 4.5 0 1 1 0 1"/><path d="M6 12a6 6 0 1 0 12 0 6 6 0 0 0-12 0z"/><line x1="19" y1="9" x2="19" y2="15"/><line x1="22" y1="12" x2="16" y2="12"/></svg>` },
  { file: "lang-python.md", label: "Python",      tag: "PY",  badge: "lang-py",  desc: "Internals · Async · Decorators", icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2C8 2 6 4 6 7v2h6v1H5c-2 0-3 1.5-3 4s1 4 3 4h1v-2.5c0-2 1-3 3-3h6c2 0 3-1 3-3V7c0-3-2-5-6-5z"/><path d="M12 22c4 0 6-2 6-5v-2h-6v-1h7c2 0 3-1.5 3-4s-1-4-3-4h-1v2.5c0 2-1 3-3 3H9c-2 0-3 1-3 3v3c0 3 2 5 6 5z"/><circle cx="9" cy="7" r="1" fill="currentColor" stroke="none"/><circle cx="15" cy="17" r="1" fill="currentColor" stroke="none"/></svg>` },
  { file: "lang-js.md",     label: "JavaScript",  tag: "JS",  badge: "lang-js",  desc: "V8 · Event Loop · Promises",     icon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2"/><path d="M14 17c0 2-3 2-3 0v-5M8 12v5c0 2 4 2 4 0"/></svg>` },
];

const PROJECT_META = [
  // Module 01 — Math for ML
  { file: "m01a.md", label: "PCA Image Compressor",        module: "01", difficulty: "Beginner",     type: "guided"   },
  { file: "m01b.md", label: "Gradient Descent Animator",   module: "01", difficulty: "Beginner",     type: "thinking" },
  { file: "m01c.md", label: "Bayesian Spam Classifier",    module: "01", difficulty: "Intermediate", type: "thinking" },
  // Module 02 — ML Basics
  { file: "m02a.md", label: "Titanic ML Pipeline",         module: "02", difficulty: "Beginner",     type: "guided"   },
  { file: "m02b.md", label: "Credit Risk Predictor",       module: "02", difficulty: "Intermediate", type: "thinking" },
  { file: "m02c.md", label: "Customer Churn XGBoost",      module: "02", difficulty: "Intermediate", type: "thinking" },
  // Module 03 — Databases & Vector DBs
  { file: "m03a.md", label: "Semantic Code Search",        module: "03", difficulty: "Intermediate", type: "guided"   },
  { file: "m03b.md", label: "Hybrid BM25+Dense Search",    module: "03", difficulty: "Intermediate", type: "thinking" },
  { file: "m03c.md", label: "Recipe Recommender",          module: "03", difficulty: "Intermediate", type: "thinking" },
  // Module 04 — Backend with Flask
  { file: "m04a.md", label: "Production ML API",           module: "04", difficulty: "Intermediate", type: "guided"   },
  { file: "m04b.md", label: "Rate-Limited API Gateway",    module: "04", difficulty: "Intermediate", type: "thinking" },
  { file: "m04c.md", label: "Real-Time WebSocket Server",  module: "04", difficulty: "Advanced",     type: "thinking" },
  // Module 05 — Deep Learning & MLOps
  { file: "m05a.md", label: "NN Training Dashboard",       module: "05", difficulty: "Intermediate", type: "guided"   },
  { file: "m05b.md", label: "ONNX Export Pipeline",        module: "05", difficulty: "Advanced",     type: "thinking" },
  { file: "m05c.md", label: "Drift Detection System",      module: "05", difficulty: "Advanced",     type: "thinking" },
  // Module 06 — GenAI Core
  { file: "m06a.md", label: "Word Analogy Explorer",       module: "06", difficulty: "Intermediate", type: "guided"   },
  { file: "m06b.md", label: "Semantic Document Retrieval", module: "06", difficulty: "Intermediate", type: "thinking" },
  { file: "m06c.md", label: "Embedding Eval Benchmark",    module: "06", difficulty: "Advanced",     type: "thinking" },
  // Module 07 — Transformers
  { file: "m07a.md", label: "Shakespeare GPT",             module: "07", difficulty: "Advanced",     type: "guided"   },
  { file: "m07b.md", label: "Custom BPE Tokenizer",        module: "07", difficulty: "Advanced",     type: "thinking" },
  { file: "m07c.md", label: "Mini Translation Model",      module: "07", difficulty: "Expert",       type: "thinking" },
  // Module 08 — RAG
  { file: "m08a.md", label: "Personal Document Q&A",       module: "08", difficulty: "Advanced",     type: "guided"   },
  { file: "m08b.md", label: "Multi-Source Research Bot",   module: "08", difficulty: "Advanced",     type: "thinking" },
  { file: "m08c.md", label: "RAGAS-Driven RAG Optimizer",  module: "08", difficulty: "Expert",       type: "thinking" },
  // Module 09 — Fine-Tuning
  { file: "m09a.md", label: "Domain LoRA Fine-Tuner",      module: "09", difficulty: "Expert",       type: "guided"   },
  { file: "m09b.md", label: "Instruction Dataset Builder", module: "09", difficulty: "Advanced",     type: "thinking" },
  { file: "m09c.md", label: "PEFT Method Comparison",      module: "09", difficulty: "Expert",       type: "thinking" },
  // Module 10 — Agents
  { file: "m10a.md", label: "Multi-Agent Crypto Analyst",  module: "10", difficulty: "Expert",       type: "guided"   },
  { file: "m10b.md", label: "Research Assistant Agent",    module: "10", difficulty: "Advanced",     type: "thinking" },
  { file: "m10c.md", label: "Self-Debugging Code Agent",   module: "10", difficulty: "Expert",       type: "thinking" },
  // Module 11 — Deployment
  { file: "m11a.md", label: "A/B Testing Deploy API",      module: "11", difficulty: "Advanced",     type: "guided"   },
  { file: "m11b.md", label: "Canary Deployment Monitor",   module: "11", difficulty: "Advanced",     type: "thinking" },
  { file: "m11c.md", label: "Multi-Model Serving Gateway", module: "11", difficulty: "Expert",       type: "thinking" },
  // Module 12 — RLHF & Alignment
  { file: "m12a.md", label: "Mini-RLHF Preference Tuner", module: "12", difficulty: "Expert",       type: "guided"   },
  { file: "m12b.md", label: "Reward Model Evaluator",      module: "12", difficulty: "Expert",       type: "thinking" },
  { file: "m12c.md", label: "DPO Dataset Curator",         module: "12", difficulty: "Expert",       type: "thinking" },
  // Module 13 — Multimodal
  { file: "m13a.md", label: "Image-Text Hybrid Search",    module: "13", difficulty: "Expert",       type: "guided"   },
  { file: "m13b.md", label: "Zero-Shot Image Classifier",  module: "13", difficulty: "Advanced",     type: "thinking" },
  { file: "m13c.md", label: "Visual Q&A System",           module: "13", difficulty: "Expert",       type: "thinking" },
  // Module 14 — Frontend
  { file: "m14a.md", label: "Full-Stack AI Blog",          module: "14", difficulty: "Advanced",     type: "guided"   },
  { file: "m14b.md", label: "Real-Time Voice Assistant",   module: "14", difficulty: "Expert",       type: "thinking" },
  { file: "m14c.md", label: "Interactive ML Dashboard",    module: "14", difficulty: "Advanced",     type: "thinking" },
];

// ── State ───────────────────────────────────────────────────────
let state = {
  currentFile:      null,
  currentType:      null,
  progress:         loadProgress(),
  projectsProgress: loadProjectsProgress(),
  focusMode:        false,
  floatingMode:     false,
  lightMode:        localStorage.getItem("aiml_theme") ? localStorage.getItem("aiml_theme") === "light" : true,
  notes:            localStorage.getItem("aiml_notes") || "",
  activeTabIdx:     0,
};

// ── Persistence ─────────────────────────────────────────────────
function loadProgress() {
  try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; }
  catch { return {}; }
}

function loadProjectsProgress() {
  try { return JSON.parse(localStorage.getItem(LS_KEY_PROJECTS)) || {}; }
  catch { return {}; }
}

function saveProgress()         { localStorage.setItem(LS_KEY,          JSON.stringify(state.progress));         }
function saveProjectsProgress() { localStorage.setItem(LS_KEY_PROJECTS, JSON.stringify(state.projectsProgress)); }

// ── Loading screen ───────────────────────────────────────────────
function hideLoadingScreen() {
  const elapsed = Date.now() - LOAD_START;
  const delay   = Math.max(0, MIN_LOAD_MS - elapsed);
  setTimeout(() => {
    const screen = document.getElementById("loading-screen");
    if (screen) screen.classList.add("fade-out");
  }, delay);
}

// ── Greeting ─────────────────────────────────────────────────────
function getGreetingPhrase() {
  const h = new Date().getHours();
  if (h < 5)  return "Working late, Harshit";
  if (h < 12) return "Good morning, Harshit";
  if (h < 17) return "Good afternoon, Harshit";
  if (h < 21) return "Good evening, Harshit";
  return "Burning midnight oil, Harshit";
}

function updateGreeting() {
  const done  = MODULE_META.filter(m => state.progress[m.file] === "done").length;
  const pdone = PROJECT_META.filter(p => state.projectsProgress[p.file] === "done").length;
  const pct   = Math.round((done / MODULE_META.length) * 100);

  const timeEl  = document.getElementById("greeting-time");
  const statsEl = document.getElementById("greeting-stats");

  if (timeEl)  timeEl.textContent  = getGreetingPhrase();
  if (statsEl) statsEl.textContent = `${done}/${MODULE_META.length} modules · ${pdone}/${PROJECT_META.length} projects · ${pct}% complete`;

  updateRing(pct);
}

// ── Progress Ring ─────────────────────────────────────────────────
function updateRing(pct) {
  const circumference = 2 * Math.PI * 50; // 314.16
  const offset = circumference * (1 - pct / 100);
  const ringFill = document.getElementById("ring-fill");
  const ringPct  = document.getElementById("ring-pct");
  if (ringFill) ringFill.style.strokeDashoffset = offset;
  if (ringPct)  ringPct.textContent = pct + "%";
}

// ── marked.js config ────────────────────────────────────────────
marked.setOptions({
  highlight: (code, lang) => {
    if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang }).value;
    return hljs.highlightAuto(code).value;
  },
  breaks: false,
  gfm: true,
});

// ── DOM helper ───────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// ── Theme Toggle ─────────────────────────────────────────────────
function applyTheme(light) {
  document.body.classList.toggle("light-mode", light);

  const hljsLink = document.getElementById("hljs-theme");
  if (hljsLink) {
    hljsLink.href = light
      ? "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css"
      : "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/base16/onedark.min.css";
  }

  const iconWrap = document.getElementById("dock-theme-icon");
  if (iconWrap) {
    iconWrap.innerHTML = light
      // sun icon
      ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
           <circle cx="12" cy="12" r="5"/>
           <line x1="12" y1="1" x2="12" y2="3"/>
           <line x1="12" y1="21" x2="12" y2="23"/>
           <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
           <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
           <line x1="1" y1="12" x2="3" y2="12"/>
           <line x1="21" y1="12" x2="23" y2="12"/>
           <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
           <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
         </svg>`
      // moon icon
      : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
           <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/>
         </svg>`;
  }

  const dockTheme = $("dock-theme");
  if (dockTheme) {
    dockTheme.classList.toggle("active-dock", light);
    const lbl = dockTheme.querySelector(".dock-label");
    if (lbl) lbl.textContent = light ? "Dark Mode" : "Light Mode";
  }
}

function toggleTheme() {
  state.lightMode = !state.lightMode;
  localStorage.setItem("aiml_theme", state.lightMode ? "light" : "dark");
  applyTheme(state.lightMode);
}


// ── Floating Sidebar Mode ───────────────────────────────────────
function toggleFloatingMode() {
  state.floatingMode = !state.floatingMode;
  document.body.classList.toggle("floating-mode", state.floatingMode);
}

document.addEventListener("DOMContentLoaded", () => {
  const pinBtn = document.getElementById("btn-pin-sidebar");
  if (pinBtn) {
    pinBtn.addEventListener("click", toggleFloatingMode);
  }
});

// ── Focus Mode ───────────────────────────────────────────────────
function toggleFocusMode() {
  state.focusMode = !state.focusMode;
  document.body.classList.toggle("focus-mode", state.focusMode);

  const dockFocus = $("dock-focus");
  const topFocus  = $("btn-focus");

  if (dockFocus) {
    dockFocus.classList.toggle("active-dock", state.focusMode);
    const lbl = dockFocus.querySelector(".dock-label");
    if (lbl) lbl.textContent = state.focusMode ? "Pin Sidebar" : "Focus Mode";
  }
  if (topFocus) topFocus.classList.toggle("active", state.focusMode);

  if (state.focusMode) {
    if (document.documentElement.requestFullscreen) {
      document.documentElement.requestFullscreen().catch(() => {});
    }
  } else {
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    }
    $("sidebar").classList.remove("focus-revealed");
  }
}

// Sync if user presses Escape to exit fullscreen
document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement && state.focusMode) {
    state.focusMode = false;
    document.body.classList.remove("focus-mode");
    $("sidebar").classList.remove("focus-revealed");
    const dockFocus = $("dock-focus");
    if (dockFocus) {
      dockFocus.classList.remove("active-dock");
      const lbl = dockFocus.querySelector(".dock-label");
      if (lbl) lbl.textContent = "Focus";
    }
    const topFocus = $("btn-focus");
    if (topFocus) topFocus.classList.remove("active");
  }
});

// ── Sidebar hover in focus mode ───────────────────────────────────
(function setupSidebarHover() {
  const hoverZone = $("sidebar-hover-zone");
  const sidebar   = $("sidebar");
  if (!hoverZone || !sidebar) return;

  hoverZone.addEventListener("mouseenter", () => {
    if (state.focusMode || state.floatingMode) sidebar.classList.add("focus-revealed");
  });

  sidebar.addEventListener("mouseleave", () => {
    if (state.focusMode || state.floatingMode) sidebar.classList.remove("focus-revealed");
  });
})();

// ── Mobile sidebar ───────────────────────────────────────────────
function openSidebar() {
  $("sidebar").classList.add("mobile-open");
  $("sidebar-overlay").classList.add("visible");
  const toggle = $("sidebar-toggle");
  toggle.classList.add("open");
  toggle.setAttribute("aria-expanded", "true");
}

function closeSidebar() {
  $("sidebar").classList.remove("mobile-open");
  $("sidebar-overlay").classList.remove("visible");
  const toggle = $("sidebar-toggle");
  toggle.classList.remove("open");
  toggle.setAttribute("aria-expanded", "false");
}

// ── Content fade helper ──────────────────────────────────────────
function fadeContent(fn) {
  const ca = $("content-area");
  ca.style.opacity = "0";
  requestAnimationFrame(() => {
    setTimeout(() => {
      fn();
      ca.style.opacity = "1";
    }, 160);
  });
}

// ── Build module sidebar nav ─────────────────────────────────────
function buildNav() {
  const ul = $("module-list");
  ul.innerHTML = "";

  MODULE_META.forEach((mod, idx) => {
    const completed = state.progress[mod.file] === "done";
    const li        = document.createElement("li");
    li.dataset.file = mod.file;
    li.dataset.idx  = idx;
    li.className    = completed ? "completed" : "";
    li.setAttribute("role", "listitem");
    li.setAttribute("tabindex", "0");
    li.setAttribute("aria-label", `Module ${mod.tag}: ${mod.label}${completed ? " (completed)" : ""}`);

    li.innerHTML = `
      <span class="module-num" aria-hidden="true">${mod.tag}</span>
      <span class="module-name">${mod.label}</span>
      <span class="status-dot" aria-hidden="true"></span>
    `;

    li.addEventListener("click", () => { openModule(mod.file, mod.label); closeSidebar(); });
    li.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openModule(mod.file, mod.label); closeSidebar(); }
    });
    ul.appendChild(li);
  });

  updateProgressBar();
}

// ── Build project sidebar nav ────────────────────────────────────
function buildProjectNav() {
  const ul = $("project-list");
  ul.innerHTML = "";

  PROJECT_META.forEach((proj, idx) => {
    const completed = state.projectsProgress[proj.file] === "done";
    const li        = document.createElement("li");
    li.dataset.file = proj.file;
    li.dataset.idx  = idx;
    li.className    = "project-item" + (completed ? " completed" : "");
    li.setAttribute("role", "listitem");
    li.setAttribute("tabindex", "0");
    li.setAttribute("aria-label", `Project: ${proj.label}${completed ? " (completed)" : ""}`);

    li.innerHTML = `
      <span class="module-num" aria-hidden="true">${proj.type === "guided" ? "▶" : "◈"} ${proj.module}</span>
      <span class="module-name">${proj.label}</span>
      <span class="status-dot" aria-hidden="true"></span>
    `;

    li.addEventListener("click", () => { openProject(proj.file, proj.label); closeSidebar(); });
    li.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openProject(proj.file, proj.label); closeSidebar(); }
    });
    ul.appendChild(li);
  });
}

// ── Build code sidebar nav ────────────────────────────────────────
function buildCodeNav() {
  const ul = $("code-list");
  ul.innerHTML = "";

  let currentModule = "";

  CODE_META.forEach((code, idx) => {
    if (code.module !== currentModule) {
      currentModule = code.module;
      const modMeta = MODULE_META.find(m => m.tag === currentModule);
      const groupHeader = document.createElement("div");
      groupHeader.className = "module-group-header";
      groupHeader.textContent = `${currentModule} · ${modMeta ? modMeta.label : ""}`;
      ul.appendChild(groupHeader);
    }

    const li = document.createElement("li");
    li.dataset.file = code.file;
    li.dataset.idx  = idx;
    li.className    = "code-item";
    li.setAttribute("role", "listitem");
    li.setAttribute("tabindex", "0");
    li.setAttribute("aria-label", `Code file: ${code.label}`);

    li.innerHTML = `
      <span class="module-num" aria-hidden="true" style="opacity:0.4;">.py</span>
      <span class="module-name">${code.label}</span>
    `;

    li.addEventListener("click", () => { openCode(code.file, code.label); closeSidebar(); });
    li.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openCode(code.file, code.label); closeSidebar(); }
    });
    ul.appendChild(li);
  });
}

// ── Build language sidebar nav ────────────────────────────────────
function buildLanguageNav() {
  const ul = $("language-list");
  ul.innerHTML = "";

  LANGUAGE_META.forEach((lang, idx) => {
    const li = document.createElement("li");
    li.dataset.file = lang.file;
    li.dataset.idx  = idx;
    li.className    = "project-item";
    li.setAttribute("role", "listitem");
    li.setAttribute("tabindex", "0");
    li.setAttribute("aria-label", `Language: ${lang.label}`);

    li.innerHTML = `
      <span class="module-num lang-nav-icon" aria-hidden="true">${lang.icon}</span>
      <span class="module-name">${lang.label}</span>
    `;

    li.addEventListener("click", () => { openLanguage(lang.file, lang.label); closeSidebar(); });
    li.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openLanguage(lang.file, lang.label); closeSidebar(); }
    });
    ul.appendChild(li);
  });
}

// ── Build languages welcome grid ─────────────────────────────────
function buildLanguagesGrid() {
  const grid = $("languages-grid");
  if (!grid) return;
  grid.innerHTML = "";

  LANGUAGE_META.forEach((lang, idx) => {
    const card = document.createElement("div");
    card.className = `welcome-module-card language-card wlc-${idx}`;
    card.style.animationDelay = `${idx * 35}ms`;
    card.setAttribute("role", "listitem");
    card.setAttribute("tabindex", "0");
    card.setAttribute("aria-label", `Open language: ${lang.label}`);

    card.innerHTML = `
      <div class="wmc-icon" aria-hidden="true">${lang.icon}</div>
      <span class="wmc-num" aria-hidden="true">LANGUAGE</span>
      <div class="wmc-title">${lang.label}</div>
      <div class="wmc-lang-desc">${lang.desc}</div>
      <span class="lang-badge ${lang.badge}" aria-label="${lang.label}">${lang.tag}</span>
      <div class="wmc-status" aria-hidden="true"></div>
    `;

    card.addEventListener("click", () => openLanguage(lang.file, lang.label));
    card.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openLanguage(lang.file, lang.label); }
    });
    grid.appendChild(card);
  });
}

// ── Build module welcome grid ────────────────────────────────────
function buildWelcomeGrid() {
  const grid = $("welcome-grid");
  if (!grid) return;
  grid.innerHTML = "";

  MODULE_META.forEach((mod, idx) => {
    const completed = state.progress[mod.file] === "done";
    const card = document.createElement("div");
    card.className = `welcome-module-card wmc-${idx}`;
    card.style.animationDelay = `${idx * 30}ms`;
    card.setAttribute("role", "listitem");
    card.setAttribute("tabindex", "0");
    card.setAttribute("aria-label", `Open module ${mod.tag}: ${mod.label}`);

    card.innerHTML = `
      <div class="wmc-icon" aria-hidden="true">${mod.icon}</div>
      <span class="wmc-num" aria-hidden="true">MODULE ${mod.tag}</span>
      <div class="wmc-title">${mod.label}</div>
      <div class="wmc-status ${completed ? "done" : ""}" aria-hidden="true"></div>
    `;

    card.addEventListener("click", () => openModule(mod.file, mod.label));
    card.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openModule(mod.file, mod.label); }
    });
    grid.appendChild(card);
  });
}

// ── Build projects welcome grid ──────────────────────────────────
function buildProjectsGrid() {
  const grid = $("projects-grid");
  if (!grid) return;
  grid.innerHTML = "";

  PROJECT_META.forEach((proj, idx) => {
    const completed = state.projectsProgress[proj.file] === "done";
    const card = document.createElement("div");
    card.className = `welcome-module-card project-card wpc-${idx}`;
    card.style.animationDelay = `${idx * 28}ms`;
    card.setAttribute("role", "listitem");
    card.setAttribute("tabindex", "0");
    card.setAttribute("aria-label", `Open project: ${proj.label} (${proj.difficulty})`);

    const projIcon = proj.type === "guided"
      ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`;
    card.innerHTML = `
      <div class="wmc-icon" aria-hidden="true">${projIcon}</div>
      <span class="wmc-num" aria-hidden="true">MOD ${proj.module} · <span class="proj-type-tag type-${proj.type}">${proj.type === "guided" ? "GUIDED" : "THINKING"}</span></span>
      <div class="wmc-title">${proj.label}</div>
      <span class="difficulty-badge diff-${proj.difficulty.toLowerCase()}" aria-label="Difficulty: ${proj.difficulty}">${proj.difficulty}</span>
      <div class="wmc-status ${completed ? "done" : ""}" aria-hidden="true"></div>
    `;

    card.addEventListener("click", () => openProject(proj.file, proj.label));
    card.addEventListener("keydown", e => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openProject(proj.file, proj.label); }
    });
    grid.appendChild(card);
  });
}

// ── Build code welcome — grouped by module ───────────────────────
function buildCodeGrid() {
  const container = $("code-by-module");
  if (!container) return;
  container.innerHTML = "";

  // Group files by module tag
  const groups = {};
  CODE_META.forEach(code => {
    if (!groups[code.module]) groups[code.module] = [];
    groups[code.module].push(code);
  });

  let sectionIdx = 0;
  Object.keys(groups).sort().forEach(moduleTag => {
    const modMeta = MODULE_META.find(m => m.tag === moduleTag);
    const section = document.createElement("div");
    section.className = "code-module-section";

    const header = document.createElement("div");
    header.className = "code-module-header";
    header.innerHTML = `
      <span class="code-mod-tag">Module ${moduleTag}</span>
      <span class="code-mod-name">${modMeta ? modMeta.label : ""}</span>
    `;
    section.appendChild(header);

    const grid = document.createElement("div");
    grid.className = "code-module-grid";

    groups[moduleTag].forEach((code, fileIdx) => {
      const ext  = code.file.split(".").pop();
      const name = code.file.split("/").pop();

      const card = document.createElement("div");
      card.className = "code-file-card";
      card.style.animationDelay = `${(sectionIdx * 4 + fileIdx) * 25}ms`;
      card.setAttribute("role", "listitem");
      card.setAttribute("tabindex", "0");
      card.setAttribute("aria-label", `Open code file: ${name}`);

      card.innerHTML = `
        <span class="code-file-ext">${ext}</span>
        <div class="code-file-name">${name}</div>
      `;

      card.addEventListener("click", () => openCode(code.file, code.label));
      card.addEventListener("keydown", e => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openCode(code.file, code.label); }
      });
      grid.appendChild(card);
    });

    section.appendChild(grid);
    container.appendChild(section);
    sectionIdx++;
  });
}

// ── Progress bar ────────────────────────────────────────────────
function updateProgressBar() {
  const total = MODULE_META.length;
  const done  = MODULE_META.filter(m => state.progress[m.file] === "done").length;
  const pct   = Math.round((done / total) * 100);

  $("progress-count").textContent    = done;
  $("module-total").textContent      = total;
  $("progress-bar-fill").style.width = `${(done / total) * 100}%`;

  const container = $("progress-container");
  if (container) container.setAttribute("aria-valuenow", done);

  updateRing(pct);
  updateGreeting();
}

// ── Open module ──────────────────────────────────────────────────
async function openModule(file, label) {
  state.currentFile = file;
  state.currentType = "module";

  document.querySelectorAll("#module-list li").forEach(li =>
    li.classList.toggle("active", li.dataset.file === file));
  document.querySelectorAll("#project-list li").forEach(li =>
    li.classList.remove("active"));
  document.querySelectorAll("#code-list li").forEach(li =>
    li.classList.remove("active"));

  $("breadcrumb").textContent = label;

  const isDone = state.progress[file] === "done";
  const btn    = $("btn-complete");
  btn.classList.remove("hidden");
  btn.textContent = isDone ? "✓ COMPLETED" : "MARK COMPLETE";
  btn.className   = "complete-btn" + (isDone ? " done" : "");

  fadeContent(async () => {
    $("welcome-screen").classList.add("hidden");
    $("doc-view").classList.remove("hidden");
    $("doc-rendered").innerHTML = `<p style="font-family:monospace;color:#444;font-size:13px;">Loading ${file}…</p>`;

    try {
      const res = await fetch(`docs/${file}`);
      if (!res.ok) {
        if (res.status === 404) { renderNotReady(file, label); return; }
        throw new Error(`HTTP ${res.status}`);
      }
      renderMarkdown(await res.text());
    } catch (err) {
      $("doc-rendered").innerHTML = `<div class="error-box"><strong>Error loading ${file}</strong><br>${err.message}</div>`;
    }
  });
}

// ── Open project ─────────────────────────────────────────────────
async function openProject(file, label) {
  state.currentFile = file;
  state.currentType = "project";

  document.querySelectorAll("#module-list li").forEach(li =>
    li.classList.remove("active"));
  document.querySelectorAll("#project-list li").forEach(li =>
    li.classList.toggle("active", li.dataset.file === file));
  document.querySelectorAll("#code-list li").forEach(li =>
    li.classList.remove("active"));

  $("breadcrumb").textContent = "PROJECT — " + label;

  const isDone = state.projectsProgress[file] === "done";
  const btn    = $("btn-complete");
  btn.classList.remove("hidden");
  btn.textContent = isDone ? "✓ COMPLETED" : "MARK COMPLETE";
  btn.className   = "complete-btn" + (isDone ? " done" : "");

  fadeContent(async () => {
    $("welcome-screen").classList.add("hidden");
    $("doc-view").classList.remove("hidden");
    $("doc-rendered").innerHTML = `<p style="font-family:monospace;color:#444;font-size:13px;">Loading ${file}…</p>`;

    try {
      const res = await fetch(`docs/projects/${file}`);
      if (!res.ok) {
        if (res.status === 404) { renderNotReady(file, label); return; }
        throw new Error(`HTTP ${res.status}`);
      }
      renderMarkdown(await res.text());
    } catch (err) {
      $("doc-rendered").innerHTML = `<div class="error-box"><strong>Error loading ${file}</strong><br>${err.message}</div>`;
    }
  });
}

// ── Open code file ───────────────────────────────────────────────
async function openCode(file, label) {
  state.currentFile = file;
  state.currentType = "code";

  document.querySelectorAll("#module-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#project-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#code-list li").forEach(li =>
    li.classList.toggle("active", li.dataset.file === file));

  $("breadcrumb").textContent = "CODE — " + label;
  $("btn-complete").classList.add("hidden");

  fadeContent(async () => {
    $("welcome-screen").classList.add("hidden");
    $("doc-view").classList.remove("hidden");
    $("doc-rendered").innerHTML = `<p style="font-family:monospace;color:#444;font-size:13px;">Loading ${file}…</p>`;

    try {
      const res = await fetch(`src/${file}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      let ext = file.split(".").pop();
      if (ext === "py")             ext = "python";
      else if (ext === "cpp" || ext === "h") ext = "cpp";
      else if (ext === "sql")       ext = "sql";

      renderMarkdown("```" + ext + "\n" + text + "\n```");
    } catch (err) {
      $("doc-rendered").innerHTML = `<div class="error-box"><strong>Error loading ${file}</strong><br>${err.message}</div>`;
    }
  });
}

// ── Open language file ───────────────────────────────────────────
async function openLanguage(file, label) {
  state.currentFile = file;
  state.currentType = "language";

  document.querySelectorAll("#module-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#project-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#code-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#language-list li").forEach(li =>
    li.classList.toggle("active", li.dataset.file === file));

  $("breadcrumb").textContent = "LANGUAGE — " + label;
  $("btn-complete").classList.add("hidden");

  fadeContent(async () => {
    $("welcome-screen").classList.add("hidden");
    $("doc-view").classList.remove("hidden");
    $("doc-rendered").innerHTML = `<p style="font-family:monospace;color:#444;font-size:13px;">Loading ${file}…</p>`;

    try {
      const res = await fetch(`docs/${file}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      renderMarkdown(await res.text());
    } catch (err) {
      $("doc-rendered").innerHTML = `<div class="error-box"><strong>Error loading ${file}</strong><br>${err.message}</div>`;
    }
  });
}

// ── Render markdown ──────────────────────────────────────────────
function renderMarkdown(md) {
  $("doc-rendered").innerHTML = marked.parse(md);

  $("doc-rendered").querySelectorAll("pre code").forEach(block => hljs.highlightElement(block));

  $("doc-rendered").querySelectorAll("a[href]").forEach(a => {
    const basename = a.getAttribute("href").split("/").pop();
    const mod  = MODULE_META.find(m => m.file === basename);
    const proj = PROJECT_META.find(p => p.file === basename);
    if (mod)       a.addEventListener("click", e => { e.preventDefault(); openModule(mod.file,   mod.label);   });
    else if (proj) a.addEventListener("click", e => { e.preventDefault(); openProject(proj.file, proj.label); });
  });

  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetPromise([$("doc-rendered")]).catch(console.error);
  }

  $("content-area").scrollTop = 0;
}

// ── Not-ready placeholder ────────────────────────────────────────
function renderNotReady(file, label) {
  const moduleNum = file.split("-")[0];
  $("doc-rendered").innerHTML = `
    <div style="border:1px solid rgba(255,255,255,0.1);padding:40px;background:#0d0d0d;max-width:600px;border-radius:2px;">
      <div style="display:inline-block;background:rgba(230,255,0,0.08);color:#E6FF00;font-family:monospace;
                  font-size:10px;letter-spacing:2px;padding:4px 12px;margin-bottom:20px;border:1px solid rgba(230,255,0,0.2);">
        NOT YET GENERATED
      </div>
      <h2 style="font-size:26px;font-weight:900;margin-bottom:12px;font-family:'Space Grotesk',sans-serif;">${label}</h2>
      <p style="font-family:monospace;color:#555;margin-bottom:20px;font-size:13px;">This content has not been generated yet.</p>
      <div style="border:1px solid rgba(255,255,255,0.08);padding:16px;font-family:monospace;font-size:12px;background:rgba(255,255,255,0.02);color:#666;">
        File: docs/${file}<br>
        Code: src/${moduleNum}-*/
      </div>
    </div>
  `;
}

// ── Toggle complete ──────────────────────────────────────────────
function toggleComplete() {
  if (!state.currentFile) return;

  const isProject   = state.currentType === "project";
  const progressObj = isProject ? state.projectsProgress : state.progress;
  const isDone      = progressObj[state.currentFile] === "done";

  if (isDone) delete progressObj[state.currentFile];
  else        progressObj[state.currentFile] = "done";

  if (isProject) {
    saveProjectsProgress();
    buildProjectNav();
    buildProjectsGrid();
  } else {
    saveProgress();
    buildNav();
    buildWelcomeGrid();
    updateProgressBar();
  }

  const nowDone = progressObj[state.currentFile] === "done";
  const btn = $("btn-complete");
  btn.textContent = nowDone ? "✓ COMPLETED" : "MARK COMPLETE";
  btn.className   = "complete-btn" + (nowDone ? " done" : "");

  const listId = isProject ? "project-list" : "module-list";
  document.querySelectorAll(`#${listId} li`).forEach(li => {
    if (li.dataset.file === state.currentFile) li.classList.toggle("completed", nowDone);
  });

  updateGreeting();
}

// ── Reset all progress ───────────────────────────────────────────
function resetProgress() {
  if (!confirm("Reset all progress? This cannot be undone.")) return;
  state.progress         = {};
  state.projectsProgress = {};
  saveProgress();
  saveProjectsProgress();
  buildNav();
  buildProjectNav();
  buildLanguageNav();
  buildCodeNav();
  buildWelcomeGrid();
  buildProjectsGrid();
  buildLanguagesGrid();
  buildCodeGrid();
  updateProgressBar();
}

// ── Go Home (roadmap) ────────────────────────────────────────────
function goHome() {
  state.currentFile = null;
  state.currentType = null;
  document.querySelectorAll("#module-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#project-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#language-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#code-list li").forEach(li => li.classList.remove("active"));

  fadeContent(() => {
    $("breadcrumb").textContent = "Roadmap";
    $("doc-view").classList.add("hidden");
    $("welcome-screen").classList.remove("hidden");
    $("btn-complete").classList.add("hidden");
    buildWelcomeGrid();
    buildProjectsGrid();
    buildLanguagesGrid();
    buildCodeGrid();
    updateGreeting();
  });
}

// ── Sidebar roadmap btn ──────────────────────────────────────────
$("btn-roadmap").addEventListener("click", goHome);

// ── Mobile sidebar events ────────────────────────────────────────
$("sidebar-toggle").addEventListener("click", () => {
  const isOpen = $("sidebar").classList.contains("mobile-open");
  isOpen ? closeSidebar() : openSidebar();
});

$("sidebar-overlay").addEventListener("click", closeSidebar);

// ── Focus button (topbar) ────────────────────────────────────────
$("btn-focus").addEventListener("click", toggleFocusMode);

// ── Keyboard shortcuts ───────────────────────────────────────────
document.addEventListener("keydown", e => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    if (state.currentFile) toggleComplete();
    return;
  }
  if (e.key === "f" || e.key === "F") {
    // Only toggle focus if not typing in an input
    const tag = document.activeElement.tagName;
    if (tag !== "INPUT" && tag !== "TEXTAREA") toggleFocusMode();
    return;
  }
  if (e.key === "Escape") {
    if (state.focusMode) { toggleFocusMode(); return; }
    if ($("sidebar").classList.contains("mobile-open")) { closeSidebar(); return; }
    const searchModal = $("search-modal");
    if (searchModal && !searchModal.classList.contains("hidden")) return;
    goHome();
  }
});

// ── Sidebar Tabs — Dynamic Drum Selector ─────────────────────────
const ALL_NAV_LISTS = ["module-list", "project-list", "language-list", "code-list"];

function setActiveTab(activeIdx) {
  state.activeTabIdx = activeIdx;
  const tabs = Array.from(document.querySelectorAll(".sidebar-tab"));

  tabs.forEach((tab, i) => {
    const dist = Math.abs(i - activeIdx);
    tab.style.setProperty("--dist", dist);
    tab.classList.toggle("active", i === activeIdx);
    tab.setAttribute("aria-selected", i === activeIdx ? "true" : "false");
  });

  const target = tabs[activeIdx]?.getAttribute("data-target");
  ALL_NAV_LISTS.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("hidden", id !== target);
  });
}

document.querySelectorAll(".sidebar-tab").forEach((tab, i) => {
  tab.addEventListener("click", () => setActiveTab(i));

  tab.addEventListener("mouseenter", () => {
    if (i === state.activeTabIdx) return;
    const tabs = Array.from(document.querySelectorAll(".sidebar-tab"));
    tabs.forEach((t, j) => {
      const distFromHover = Math.abs(j - i);
      const distFromActive = Math.abs(j - state.activeTabIdx);
      const blendedDist = Math.min(distFromHover * 0.5, distFromActive);
      t.style.setProperty("--dist", blendedDist.toFixed(2));
    });
  });

  tab.addEventListener("mouseleave", () => {
    setActiveTab(state.activeTabIdx);
  });
});


// ── Welcome Screen Tabs ──────────────────────────────────────────
document.querySelectorAll(".welcome-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".welcome-tab").forEach(t => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");

    const panelId = tab.getAttribute("data-panel");
    document.querySelectorAll(".welcome-panel").forEach(panel => {
      panel.classList.toggle("hidden", panel.id !== panelId);
    });

    // Rebuild panels on switch to keep animations fresh
    if (panelId === "modules-panel")   buildWelcomeGrid();
    if (panelId === "projects-panel")  buildProjectsGrid();
    if (panelId === "languages-panel") buildLanguagesGrid();
    if (panelId === "code-panel")      buildCodeGrid();
  });
});

// ── Dock button handlers ─────────────────────────────────────────
$("dock-home").addEventListener("click",   goHome);
$("dock-focus").addEventListener("click",  toggleFocusMode);
$("dock-theme").addEventListener("click",  toggleTheme);
$("dock-reset").addEventListener("click",  resetProgress);

// ── Notes ────────────────────────────────────────────────────────
(function setupNotes() {
  const notesText = $("notes-content");
  if (notesText) {
    notesText.value = state.notes;
    notesText.addEventListener("input", e => {
      state.notes = e.target.value;
      localStorage.setItem("aiml_notes", state.notes);
    });
  }

  function toggleNotes() {
    const panel = $("notes-panel");
    if (panel) panel.classList.toggle("hidden");
  }

  $("dock-notes").addEventListener("click", toggleNotes);
  $("close-notes").addEventListener("click", () => $("notes-panel").classList.add("hidden"));
})();

// ── Search ───────────────────────────────────────────────────────
(function setupSearch() {
  const searchModal   = $("search-modal");
  const searchInput   = $("search-input");
  const searchResults = $("search-results");
  let selectedIndex   = -1;

  function openSearch() {
    searchModal.classList.remove("hidden");
    searchInput.value = "";
    searchResults.innerHTML = "";
    selectedIndex = -1;
    setTimeout(() => searchInput.focus(), 80);
  }

  function closeSearch() {
    searchModal.classList.add("hidden");
  }

  $("dock-search").addEventListener("click", openSearch);

  document.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "k") {
      e.preventDefault();
      openSearch();
    }
    if (e.key === "Escape" && !searchModal.classList.contains("hidden")) {
      e.stopPropagation();
      closeSearch();
    }
  });

  searchInput.addEventListener("input", e => {
    const q = e.target.value.toLowerCase().trim();
    searchResults.innerHTML = "";
    selectedIndex = -1;
    if (!q) return;

    const res = [];
    MODULE_META.forEach(m => {
      if (m.label.toLowerCase().includes(q) || m.file.toLowerCase().includes(q))
        res.push({ type: "Module", label: m.label, execute: () => openModule(m.file, m.label) });
    });
    PROJECT_META.forEach(p => {
      if (p.label.toLowerCase().includes(q) || p.file.toLowerCase().includes(q))
        res.push({ type: "Project", label: p.label, execute: () => openProject(p.file, p.label) });
    });
    LANGUAGE_META.forEach(l => {
      if (l.label.toLowerCase().includes(q) || l.file.toLowerCase().includes(q))
        res.push({ type: "Language", label: l.label, execute: () => openLanguage(l.file, l.label) });
    });
    CODE_META.forEach(c => {
      if (c.label.toLowerCase().includes(q) || c.file.toLowerCase().includes(q))
        res.push({ type: "Code", label: c.label, execute: () => openCode(c.file, c.label) });
    });

    res.slice(0, 12).forEach((item, idx) => {
      const li = document.createElement("li");
      li.innerHTML = `<span>${item.label}</span><span class="sr-type">${item.type}</span>`;
      li.addEventListener("click", () => { closeSearch(); item.execute(); });
      li.addEventListener("mouseover", () => {
        Array.from(searchResults.children).forEach(c => c.classList.remove("selected"));
        li.classList.add("selected");
        selectedIndex = idx;
      });
      searchResults.appendChild(li);
    });
  });

  searchInput.addEventListener("keydown", e => {
    const items = Array.from(searchResults.children);
    if (e.key === "ArrowDown") {
      e.preventDefault();
      selectedIndex = (selectedIndex + 1) % items.length;
      updateSel();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      selectedIndex = (selectedIndex - 1 + items.length) % items.length;
      updateSel();
    } else if (e.key === "Enter" && selectedIndex >= 0) {
      e.preventDefault();
      items[selectedIndex]?.click();
    }

    function updateSel() {
      items.forEach(c => c.classList.remove("selected"));
      if (items[selectedIndex]) items[selectedIndex].classList.add("selected");
    }
  });

  searchModal.addEventListener("click", e => {
    if (e.target === searchModal) closeSearch();
  });
})();

// ── Init ─────────────────────────────────────────────────────────
function init() {
  // Apply saved theme before any content renders
  applyTheme(state.lightMode);

  // Jump animation on very first load
  if (!localStorage.getItem("aiml_first_load_jump")) {
    const themeBtn = document.getElementById("dock-theme");
    if (themeBtn) {
      themeBtn.classList.add("dock-jump", "dock-reveal-label");
      setTimeout(() => {
        themeBtn.classList.remove("dock-jump", "dock-reveal-label");
        localStorage.setItem("aiml_first_load_jump", "1");
      }, 2500);
    }
  }

  buildNav();
  buildProjectNav();
  buildLanguageNav();
  buildCodeNav();
  buildWelcomeGrid();
  buildProjectsGrid();
  buildLanguagesGrid();
  buildCodeGrid();
  setActiveTab(0);
  updateProgressBar();
  updateGreeting();

  if (window.location.hash) {
    const hashFile  = window.location.hash.slice(1);
    const match     = MODULE_META.find(m => m.file === hashFile);
    const projMatch = PROJECT_META.find(p => p.file === hashFile);
    if (match)          openModule(match.file, match.label);
    else if (projMatch) openProject(projMatch.file, projMatch.label);
  }

  hideLoadingScreen();
}

init();
