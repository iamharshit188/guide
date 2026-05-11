"use strict";

// ── Constants ──────────────────────────────────────────────────
const LS_KEY          = "aiml_platform_progress";
const LS_KEY_PROJECTS = "aiml_platform_projects_progress";
const LOAD_START      = Date.now();
const MIN_LOAD_MS     = 1400;

const MODULE_META = [
  { file: "01-math.md",           label: "Math for ML",               tag: "01" },
  { file: "02-ml-basics.md",      label: "ML Basics → Advanced",      tag: "02" },
  { file: "03-databases.md",      label: "Databases & Vector DBs",    tag: "03" },
  { file: "04-backend.md",        label: "Backend with Flask",        tag: "04" },
  { file: "05-deep-learning.md",  label: "Deep Learning & MLOps",    tag: "05" },
  { file: "06-genai-core.md",     label: "GenAI Core",               tag: "06" },
  { file: "07-transformers.md",   label: "Transformers from Scratch", tag: "07" },
  { file: "08-rag.md",            label: "RAG Chatbot",              tag: "08" },
  { file: "09-finetuning.md",     label: "Fine-Tuning",              tag: "09" },
  { file: "10-agents.md",         label: "LLM Agents & Tool Use",    tag: "10" },
  { file: "11-deployment.md",     label: "Deployment & Production",  tag: "11" },
  { file: "12-rlhf.md",           label: "RLHF & Alignment",         tag: "12" },
  { file: "13-multimodal.md",     label: "Multimodal Models",        tag: "13" },
  { file: "14-frontend.md",       label: "Frontend (React+Tailwind)",tag: "14" },
];

const CODE_META = [
  {
    "file": "01-math/calculus_demo.py",
    "label": "calculus_demo.py",
    "module": "01"
  },
  {
    "file": "01-math/matrix_ops.py",
    "label": "matrix_ops.py",
    "module": "01"
  },
  {
    "file": "01-math/probability.py",
    "label": "probability.py",
    "module": "01"
  },
  {
    "file": "01-math/vectors.py",
    "label": "vectors.py",
    "module": "01"
  },
  {
    "file": "02-ml/clustering.py",
    "label": "clustering.py",
    "module": "02"
  },
  {
    "file": "02-ml/decision_tree.py",
    "label": "decision_tree.py",
    "module": "02"
  },
  {
    "file": "02-ml/evaluation.py",
    "label": "evaluation.py",
    "module": "02"
  },
  {
    "file": "02-ml/gradient_boosting.py",
    "label": "gradient_boosting.py",
    "module": "02"
  },
  {
    "file": "02-ml/linear_regression.py",
    "label": "linear_regression.py",
    "module": "02"
  },
  {
    "file": "02-ml/logistic_regression.py",
    "label": "logistic_regression.py",
    "module": "02"
  },
  {
    "file": "02-ml/pca.py",
    "label": "pca.py",
    "module": "02"
  },
  {
    "file": "02-ml/random_forest.py",
    "label": "random_forest.py",
    "module": "02"
  },
  {
    "file": "02-ml/svm.py",
    "label": "svm.py",
    "module": "02"
  },
  {
    "file": "03-databases/chroma_demo.py",
    "label": "chroma_demo.py",
    "module": "03"
  },
  {
    "file": "03-databases/faiss_demo.py",
    "label": "faiss_demo.py",
    "module": "03"
  },
  {
    "file": "03-databases/nosql_patterns.py",
    "label": "nosql_patterns.py",
    "module": "03"
  },
  {
    "file": "03-databases/pinecone_demo.py",
    "label": "pinecone_demo.py",
    "module": "03"
  },
  {
    "file": "03-databases/sql_basics.py",
    "label": "sql_basics.py",
    "module": "03"
  },
  {
    "file": "04-backend/app.py",
    "label": "app.py",
    "module": "04"
  },
  {
    "file": "04-backend/async_tasks.py",
    "label": "async_tasks.py",
    "module": "04"
  },
  {
    "file": "04-backend/middleware.py",
    "label": "middleware.py",
    "module": "04"
  },
  {
    "file": "04-backend/ml_serving.py",
    "label": "ml_serving.py",
    "module": "04"
  },
  {
    "file": "05-deep-learning/mlflow_demo.py",
    "label": "mlflow_demo.py",
    "module": "05"
  },
  {
    "file": "05-deep-learning/monitoring.py",
    "label": "monitoring.py",
    "module": "05"
  },
  {
    "file": "05-deep-learning/nn_numpy.py",
    "label": "nn_numpy.py",
    "module": "05"
  },
  {
    "file": "05-deep-learning/onnx_export.py",
    "label": "onnx_export.py",
    "module": "05"
  },
  {
    "file": "05-deep-learning/optimizers.py",
    "label": "optimizers.py",
    "module": "05"
  },
  {
    "file": "06-genai/attention.py",
    "label": "attention.py",
    "module": "06"
  },
  {
    "file": "06-genai/kv_cache.py",
    "label": "kv_cache.py",
    "module": "06"
  },
  {
    "file": "06-genai/multihead_attention.py",
    "label": "multihead_attention.py",
    "module": "06"
  },
  {
    "file": "06-genai/positional_encoding.py",
    "label": "positional_encoding.py",
    "module": "06"
  },
  {
    "file": "06-genai/word2vec.py",
    "label": "word2vec.py",
    "module": "06"
  },
  {
    "file": "07-transformer/model.cpp",
    "label": "model.cpp",
    "module": "07"
  },
  {
    "file": "07-transformer/model.py",
    "label": "model.py",
    "module": "07"
  },
  {
    "file": "07-transformer/model_numpy.py",
    "label": "model_numpy.py",
    "module": "07"
  },
  {
    "file": "07-transformer/tokenizer.py",
    "label": "tokenizer.py",
    "module": "07"
  },
  {
    "file": "07-transformer/train.py",
    "label": "train.py",
    "module": "07"
  },
  {
    "file": "08-rag/app.py",
    "label": "app.py",
    "module": "08"
  },
  {
    "file": "08-rag/embed_store.py",
    "label": "embed_store.py",
    "module": "08"
  },
  {
    "file": "08-rag/evaluate.py",
    "label": "evaluate.py",
    "module": "08"
  },
  {
    "file": "08-rag/generator.py",
    "label": "generator.py",
    "module": "08"
  },
  {
    "file": "08-rag/ingest.py",
    "label": "ingest.py",
    "module": "08"
  },
  {
    "file": "08-rag/retriever.py",
    "label": "retriever.py",
    "module": "08"
  },
  {
    "file": "09-finetuning/evaluate.py",
    "label": "evaluate.py",
    "module": "09"
  },
  {
    "file": "09-finetuning/lora_theory.py",
    "label": "lora_theory.py",
    "module": "09"
  },
  {
    "file": "09-finetuning/merge_push.py",
    "label": "merge_push.py",
    "module": "09"
  },
  {
    "file": "09-finetuning/prepare_dataset.py",
    "label": "prepare_dataset.py",
    "module": "09"
  },
  {
    "file": "09-finetuning/train_lora.py",
    "label": "train_lora.py",
    "module": "09"
  },
  {
    "file": "09-finetuning/train_qlora.py",
    "label": "train_qlora.py",
    "module": "09"
  },
  {
    "file": "10-agents/agent_eval.py",
    "label": "agent_eval.py",
    "module": "10"
  },
  {
    "file": "10-agents/agent_memory.py",
    "label": "agent_memory.py",
    "module": "10"
  },
  {
    "file": "10-agents/react_agent.py",
    "label": "react_agent.py",
    "module": "10"
  },
  {
    "file": "10-agents/tool_calling.py",
    "label": "tool_calling.py",
    "module": "10"
  },
  {
    "file": "11-deployment/ab_serving.py",
    "label": "ab_serving.py",
    "module": "11"
  },
  {
    "file": "11-deployment/health_check.py",
    "label": "health_check.py",
    "module": "11"
  },
  {
    "file": "11-deployment/onnx_export.py",
    "label": "onnx_export.py",
    "module": "11"
  },
  {
    "file": "11-deployment/quantize.py",
    "label": "quantize.py",
    "module": "11"
  },
  {
    "file": "12-rlhf/dpo.py",
    "label": "dpo.py",
    "module": "12"
  },
  {
    "file": "12-rlhf/evaluate_alignment.py",
    "label": "evaluate_alignment.py",
    "module": "12"
  },
  {
    "file": "12-rlhf/ppo_scratch.py",
    "label": "ppo_scratch.py",
    "module": "12"
  },
  {
    "file": "12-rlhf/reward_model.py",
    "label": "reward_model.py",
    "module": "12"
  },
  {
    "file": "13-multimodal/captioning.py",
    "label": "captioning.py",
    "module": "13"
  },
  {
    "file": "13-multimodal/clip_scratch.py",
    "label": "clip_scratch.py",
    "module": "13"
  },
  {
    "file": "13-multimodal/vit_patch.py",
    "label": "vit_patch.py",
    "module": "13"
  },
  {
    "file": "13-multimodal/zero_shot.py",
    "label": "zero_shot.py",
    "module": "13"
  }
];

const PROJECT_META = [
  { file: "p01-pca-compressor.md",     label: "PCA Image Compressor",        module: "01", difficulty: "Starter"      },
  { file: "p02-titanic-pipeline.md",   label: "Titanic Survival Pipeline",   module: "02", difficulty: "Starter"      },
  { file: "p03-semantic-search.md",    label: "Semantic Code Search Engine", module: "03", difficulty: "Intermediate" },
  { file: "p04-ml-api.md",             label: "Production ML Serving API",   module: "04", difficulty: "Intermediate" },
  { file: "p05-training-dashboard.md", label: "NN Training Dashboard",       module: "05", difficulty: "Intermediate" },
  { file: "p06-word-analogy.md",       label: "Word Analogy Explorer",       module: "06", difficulty: "Intermediate" },
  { file: "p07-gpt-shakespeare.md",    label: "Shakespeare GPT",             module: "07", difficulty: "Advanced"     },
  { file: "p08-document-qa.md",        label: "Personal Document Q&A",       module: "08", difficulty: "Advanced"     },
  { file: "p09-domain-tuner.md",       label: "Domain-Specific Tuner",       module: "09", difficulty: "Expert"       },
  { file: "p10-crypto-bot-agent.md",   label: "Multi-Agent Crypto Analyst",  module: "10", difficulty: "Expert"       },
  { file: "p11-ab-test-deploy.md",     label: "A/B Testing Deployment API",  module: "11", difficulty: "Advanced"     },
  { file: "p12-small-rlhf.md",         label: "Mini-RLHF Preference Tuner",  module: "12", difficulty: "Expert"       },
  { file: "p13-multimodal-search.md",  label: "Image-Text Hybrid Search",    module: "13", difficulty: "Expert"       },
  { file: "p14-fullstack-ai-blog.md",  label: "Full-Stack AI Blog",          module: "14", difficulty: "Advanced"     },
  { file: "p15-voice-assistant.md",    label: "Real-Time Voice Assistant",   module: "13", difficulty: "Expert"       },
  { file: "p16-mlops-feature-store.md",label: "MLOps Feature Store",         module: "11", difficulty: "Expert"       },
];

// ── State ───────────────────────────────────────────────────────
let state = {
  currentFile:      null,
  currentType:      null,
  progress:         loadProgress(),
  projectsProgress: loadProjectsProgress(),
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
  const done   = MODULE_META.filter(m => state.progress[m.file] === "done").length;
  const pdone  = PROJECT_META.filter(p => state.projectsProgress[p.file] === "done").length;
  const pct    = Math.round((done / MODULE_META.length) * 100);

  const timeEl  = document.getElementById("greeting-time");
  const statsEl = document.getElementById("greeting-stats");

  if (timeEl)  timeEl.textContent  = getGreetingPhrase();
  if (statsEl) statsEl.textContent = `${done}/${MODULE_META.length} modules · ${pdone}/${PROJECT_META.length} projects · ${pct}% complete`;
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
    }, 180);
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
    li.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openModule(mod.file, mod.label); closeSidebar(); } });
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
    li.setAttribute("aria-label", `Project ${proj.module}: ${proj.label}${completed ? " (completed)" : ""}`);

    li.innerHTML = `
      <span class="module-num" aria-hidden="true">P${proj.module}</span>
      <span class="module-name">${proj.label}</span>
      <span class="status-dot" aria-hidden="true"></span>
    `;

    li.addEventListener("click", () => { openProject(proj.file, proj.label); closeSidebar(); });
    li.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openProject(proj.file, proj.label); closeSidebar(); } });
    ul.appendChild(li);
  });
}


// ── Build code sidebar nav ────────────────────────────────────
function buildCodeNav() {
  const ul = $("code-list");
  ul.innerHTML = "";

  let currentModule = "";

  CODE_META.forEach((code, idx) => {
    if (code.module !== currentModule) {
      currentModule = code.module;
      const groupHeader = document.createElement("div");
      groupHeader.className = "module-group-header";
      groupHeader.textContent = `MODULE ${currentModule}`;
      ul.appendChild(groupHeader);
    }

    const li        = document.createElement("li");
    li.dataset.file = code.file;
    li.dataset.idx  = idx;
    li.className    = "code-item";
    li.setAttribute("role", "listitem");
    li.setAttribute("tabindex", "0");
    li.setAttribute("aria-label", `Code ${code.module}: ${code.label}`);

    li.innerHTML = `
      <span class="module-num" aria-hidden="true" style="opacity: 0.5;">C${code.module}</span>
      <span class="module-name">${code.label}</span>
      <span class="status-dot" aria-hidden="true"></span>
    `;

    li.addEventListener("click", () => { openCode(code.file, code.label); closeSidebar(); });
    li.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openCode(code.file, code.label); closeSidebar(); } });
    ul.appendChild(li);
  });
}

// ── Build module welcome grid ────────────────────────────────────
function buildWelcomeGrid() {
  const grid = $("welcome-grid");
  grid.innerHTML = "";

  MODULE_META.forEach((mod, idx) => {
    const completed = state.progress[mod.file] === "done";
    const card = document.createElement("div");
    card.className = `welcome-module-card wmc-${idx}`;
    card.style.animationDelay = `${idx * 35}ms`;
    card.setAttribute("role", "listitem");
    card.setAttribute("tabindex", "0");
    card.setAttribute("aria-label", `Open module ${mod.tag}: ${mod.label}`);

    card.innerHTML = `
      <div class="wmc-num" aria-hidden="true">MODULE ${mod.tag}</div>
      <div class="wmc-title">${mod.label}</div>
      <div class="wmc-status ${completed ? "done" : ""}" aria-hidden="true"></div>
    `;

    card.addEventListener("click", () => openModule(mod.file, mod.label));
    card.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openModule(mod.file, mod.label); } });
    grid.appendChild(card);
  });
}

// ── Build projects welcome grid ──────────────────────────────────
function buildProjectsGrid() {
  const grid = $("projects-grid");
  grid.innerHTML = "";

  PROJECT_META.forEach((proj, idx) => {
    const completed = state.projectsProgress[proj.file] === "done";
    const card = document.createElement("div");
    card.className = `welcome-module-card project-card wpc-${idx}`;
    card.style.animationDelay = `${idx * 35}ms`;
    card.setAttribute("role", "listitem");
    card.setAttribute("tabindex", "0");
    card.setAttribute("aria-label", `Open project: ${proj.label} (${proj.difficulty})`);

    card.innerHTML = `
      <div class="wmc-num" aria-hidden="true">PROJECT ${proj.module}</div>
      <div class="wmc-title">${proj.label}</div>
      <span class="difficulty-badge diff-${proj.difficulty.toLowerCase()}" aria-label="Difficulty: ${proj.difficulty}">${proj.difficulty}</span>
      <div class="wmc-status ${completed ? "done" : ""}" aria-hidden="true"></div>
    `;

    card.addEventListener("click", () => openProject(proj.file, proj.label));
    card.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openProject(proj.file, proj.label); } });
    grid.appendChild(card);
  });
}


// ── Build code welcome grid ──────────────────────────────────
function buildCodeGrid() {
  const grid = $("code-grid");
  if(!grid) return;
  grid.innerHTML = "";

  CODE_META.forEach((code, idx) => {
    const card = document.createElement("div");
    card.className = `welcome-module-card project-card wmc-${idx}`;
    card.style.animationDelay = `${(idx % 15) * 30}ms`;
    card.setAttribute("role", "listitem");
    card.setAttribute("tabindex", "0");
    card.setAttribute("aria-label", `Open code: ${code.label}`);

    card.innerHTML = `
      <div class="wmc-num" aria-hidden="true">C${code.module}</div>
      <div class="wmc-title">${code.label}</div>
    `;

    card.addEventListener("click", () => openCode(code.file, code.label));
    card.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openCode(code.file, code.label); } });
    grid.appendChild(card);
  });
}

// ── Progress bar ────────────────────────────────────────────────
function updateProgressBar() {
  const total = MODULE_META.length;
  const done  = MODULE_META.filter(m => state.progress[m.file] === "done").length;

  $("progress-count").textContent    = done;
  $("module-total").textContent      = total;
  $("progress-bar-fill").style.width = `${(done / total) * 100}%`;

  const container = $("progress-container");
  if (container) container.setAttribute("aria-valuenow", done);

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

  $("breadcrumb").textContent = label;

  const isDone = state.progress[file] === "done";
  const btn    = $("btn-complete");
  btn.classList.remove("hidden");
  btn.textContent = isDone ? "✓ COMPLETED" : "MARK COMPLETE";
  btn.className   = "complete-btn" + (isDone ? " done" : "");

  fadeContent(async () => {
    $("welcome-screen").classList.add("hidden");
    $("doc-view").classList.remove("hidden");
    $("doc-rendered").innerHTML = `<p style="font-family:monospace;color:#888;padding:8px 0;">Loading ${file}…</p>`;

    try {
      const res = await fetch(`docs/${file}`);
      if (!res.ok) { if (res.status === 404) { renderNotReady(file, label); return; } throw new Error(`HTTP ${res.status}`); }
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

  $("breadcrumb").textContent = "PROJECT — " + label;

  const isDone = state.projectsProgress[file] === "done";
  const btn    = $("btn-complete");
  btn.classList.remove("hidden");
  btn.textContent = isDone ? "✓ COMPLETED" : "MARK COMPLETE";
  btn.className   = "complete-btn" + (isDone ? " done" : "");

  fadeContent(async () => {
    $("welcome-screen").classList.add("hidden");
    $("doc-view").classList.remove("hidden");
    $("doc-rendered").innerHTML = `<p style="font-family:monospace;color:#888;padding:8px 0;">Loading ${file}…</p>`;

    try {
      const res = await fetch(`docs/projects/${file}`);
      if (!res.ok) { if (res.status === 404) { renderNotReady(file, label); return; } throw new Error(`HTTP ${res.status}`); }
      renderMarkdown(await res.text());
    } catch (err) {
      $("doc-rendered").innerHTML = `<div class="error-box"><strong>Error loading ${file}</strong><br>${err.message}</div>`;
    }
  });
}


// ── Open code ─────────────────────────────────────────────────
async function openCode(file, label) {
  state.currentFile = file;
  state.currentType = "code";

  document.querySelectorAll("#module-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#project-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#code-list li").forEach(li => li.classList.toggle("active", li.dataset.file === file));

  $("breadcrumb").textContent = "CODE — " + label;
  $("btn-complete").classList.add("hidden"); // No completion for code files

  fadeContent(async () => {
    $("welcome-screen").classList.add("hidden");
    $("doc-view").classList.remove("hidden");
    $("doc-rendered").innerHTML = `<p style="font-family:monospace;color:#888;padding:8px 0;">Loading ${file}…</p>`;

    try {
      const res = await fetch(`src/${file}`);
      if (!res.ok) { throw new Error(`HTTP ${res.status}`); }
      const text = await res.text();
      let ext = file.split('.').pop();
      if(ext === 'py') ext = 'python';
      else if(ext === 'cpp' || ext === 'h') ext = 'cpp';
      else if(ext === 'sql') ext = 'sql';
      
      const md = "```" + ext + "\n" + text + "\n```";
      renderMarkdown(md);
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
    <div style="border:3px solid #000;box-shadow:4px 4px 0 #000;padding:40px;background:#fff;max-width:600px;">
      <div style="display:inline-block;background:#000;color:#FFD600;font-family:monospace;
                  font-weight:900;font-size:11px;letter-spacing:2px;padding:4px 12px;margin-bottom:16px;">
        NOT YET GENERATED
      </div>
      <h2 style="font-size:28px;font-weight:900;margin-bottom:12px;">${label}</h2>
      <p style="font-family:monospace;color:#555;margin-bottom:20px;">This content has not been generated yet.</p>
      <div style="border:2px solid #000;padding:16px;font-family:monospace;font-size:13px;background:#fffde7;">
        <strong>File:</strong> docs/${file}<br>
        <strong>Code:</strong> src/${moduleNum}-*/
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
  buildCodeNav();
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
  buildCodeNav();
  buildWelcomeGrid();
  buildProjectsGrid();
  buildCodeGrid();
  updateProgressBar();
}

// ── Roadmap button ───────────────────────────────────────────────
$("btn-roadmap").addEventListener("click", () => {
  state.currentFile = null;
  state.currentType = null;
  document.querySelectorAll("#module-list li").forEach(li => li.classList.remove("active"));
  document.querySelectorAll("#project-list li").forEach(li => li.classList.remove("active"));

  fadeContent(() => {
    $("breadcrumb").textContent = "Roadmap";
    $("doc-view").classList.add("hidden");
    $("welcome-screen").classList.remove("hidden");
    $("btn-complete").classList.add("hidden");
    buildWelcomeGrid();
    buildProjectsGrid();
    if($('code-grid')) buildCodeGrid();
    updateGreeting();
  });
});

// ── Mobile sidebar events ────────────────────────────────────────
$("sidebar-toggle").addEventListener("click", () => {
  const isOpen = $("sidebar").classList.contains("mobile-open");
  isOpen ? closeSidebar() : openSidebar();
});

$("sidebar-overlay").addEventListener("click", closeSidebar);

// ── Keyboard shortcuts ───────────────────────────────────────────
document.addEventListener("keydown", e => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    if (state.currentFile) toggleComplete();
  }
  if (e.key === "Escape") {
    if ($("sidebar").classList.contains("mobile-open")) { closeSidebar(); return; }
    $("btn-roadmap").click();
  }
});

// ── Init ─────────────────────────────────────────────────────────
function init() {
  buildNav();
  buildProjectNav();
  buildCodeNav();
  buildWelcomeGrid();
  buildProjectsGrid();
  buildCodeGrid();
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

// ── Sidebar Tabs ────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const tabs = document.querySelectorAll(".sidebar-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      
      const target = tab.getAttribute("data-target");
      ["module-list", "project-list", "code-list"].forEach(id => {
        const el = document.getElementById(id);
        if(el) {
          if(id === target) el.classList.remove("hidden");
          else el.classList.add("hidden");
        }
      });
    });
  });
});

// Sidenote logic
state.notes = localStorage.getItem("aiml_notes") || "";

document.addEventListener("DOMContentLoaded", () => {
  const notesText = document.getElementById("notes-content");
  if(notesText) {
    notesText.value = state.notes;
    notesText.addEventListener("input", (e) => {
      state.notes = e.target.value;
      localStorage.setItem("aiml_notes", state.notes);
    });
  }

  document.getElementById("btn-notes")?.addEventListener("click", () => {
    document.getElementById("notes-panel")?.classList.toggle("hidden");
  });
  document.getElementById("close-notes")?.addEventListener("click", () => {
    document.getElementById("notes-panel")?.classList.add("hidden");
  });
});

// Search logic
document.addEventListener("DOMContentLoaded", () => {
  const searchModal = document.getElementById("search-modal");
  const searchInput = document.getElementById("search-input");
  const searchResults = document.getElementById("search-results");
  let selectedIndex = -1;

  function openSearch() {
    searchModal.classList.remove("hidden");
    searchInput.value = "";
    searchResults.innerHTML = "";
    selectedIndex = -1;
    setTimeout(() => searchInput.focus(), 100);
  }

  function closeSearch() {
    searchModal.classList.add("hidden");
  }

  document.getElementById("btn-search")?.addEventListener("click", openSearch);

  // Global Ctrl+K / Cmd+K
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "k") {
      e.preventDefault();
      openSearch();
    }
    if (e.key === "Escape" && !searchModal.classList.contains("hidden")) {
      closeSearch();
    }
  });

  // Fuzzy Search
  searchInput?.addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase().trim();
    searchResults.innerHTML = "";
    selectedIndex = -1;
    if(!q) return;

    let res = [];
    MODULE_META.forEach(m => {
      if(m.label.toLowerCase().includes(q) || m.file.toLowerCase().includes(q)) 
        res.push({ type: 'Module', label: m.label, file: m.file, execute: () => openModule(m.file, m.label) });
    });
    PROJECT_META.forEach(p => {
      if(p.label.toLowerCase().includes(q) || p.file.toLowerCase().includes(q))
        res.push({ type: 'Project', label: p.label, file: p.file, execute: () => openProject(p.file, p.label) });
    });
    CODE_META.forEach(c => {
      if(c.label.toLowerCase().includes(q) || c.file.toLowerCase().includes(q))
        res.push({ type: 'Code', label: c.label, file: c.file, execute: () => openCode(c.file, c.label) });
    });

    res.slice(0, 10).forEach((item, idx) => {
      const li = document.createElement("li");
      li.innerHTML = `<span>${item.label}</span><span class="sr-type">${item.type}</span>`;
      li.dataset.idx = idx;
      li.addEventListener("click", () => {
        closeSearch();
        item.execute();
      });
      li.addEventListener("mouseover", () => {
        Array.from(searchResults.children).forEach(c => c.classList.remove("selected"));
        li.classList.add("selected");
        selectedIndex = idx;
      });
      searchResults.appendChild(li);
    });
  });

  // Arrow keys navigation in search
  searchInput?.addEventListener("keydown", (e) => {
    const items = Array.from(searchResults.children);
    if(e.key === "ArrowDown") {
      e.preventDefault();
      selectedIndex = (selectedIndex + 1) % items.length;
      updateSelection();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      selectedIndex = (selectedIndex - 1 + items.length) % items.length;
      updateSelection();
    } else if (e.key === "Enter" && selectedIndex >= 0 && selectedIndex < items.length) {
      e.preventDefault();
      items[selectedIndex].click();
    }

    function updateSelection() {
      items.forEach(c => c.classList.remove("selected"));
      if(items[selectedIndex]) items[selectedIndex].classList.add("selected");
    }
  });
});
