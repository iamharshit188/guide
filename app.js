"use strict";

// ── Constants ──────────────────────────────────────────────────
const LS_KEY          = "aiml_platform_progress";
const LS_KEY_PROJECTS = "aiml_platform_projects_progress";

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
];

const PROJECT_META = [
  { file: "p01-pca-compressor.md",     label: "PCA Image Compressor",              module: "01", difficulty: "Beginner"     },
  { file: "p02-titanic-pipeline.md",   label: "Titanic Survival Predictor",        module: "02", difficulty: "Beginner"     },
  { file: "p03-semantic-search.md",    label: "Semantic Code Search Engine",       module: "03", difficulty: "Intermediate" },
  { file: "p04-ml-api.md",             label: "Production ML Serving API",         module: "04", difficulty: "Intermediate" },
  { file: "p05-training-dashboard.md", label: "NN Training Dashboard",             module: "05", difficulty: "Intermediate" },
  { file: "p06-word-analogy.md",       label: "Word Analogy Explorer",             module: "06", difficulty: "Intermediate" },
  { file: "p07-gpt-shakespeare.md",    label: "Shakespeare GPT",                   module: "07", difficulty: "Advanced"     },
  { file: "p08-document-qa.md",        label: "Personal Document Q&A",             module: "08", difficulty: "Advanced"     },
  { file: "p09-domain-tuner.md",       label: "Domain-Specific Tuner",             module: "09", difficulty: "Advanced"     },
];

// ── State ───────────────────────────────────────────────────────
let state = {
  currentFile:      null,
  currentType:      null,   // "module" | "project"
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

// ── marked.js config ────────────────────────────────────────────
marked.setOptions({
  highlight: (code, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
  breaks: false,
  gfm: true,
});

// ── DOM helpers ─────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

// ── Build module sidebar nav ─────────────────────────────────────
function buildNav() {
  const ul = $("module-list");
  ul.innerHTML = "";

  MODULE_META.forEach((mod, idx) => {
    const completed = state.progress[mod.file] === "done";
    const li = document.createElement("li");
    li.dataset.file = mod.file;
    li.dataset.idx  = idx;
    li.className    = completed ? "completed" : "";

    li.innerHTML = `
      <span class="module-num">${mod.tag}</span>
      <span class="module-name">${mod.label}</span>
      <span class="status-dot"></span>
    `;

    li.addEventListener("click", () => openModule(mod.file, mod.label));
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
    const li = document.createElement("li");
    li.dataset.file = proj.file;
    li.dataset.idx  = idx;
    li.className    = "project-item" + (completed ? " completed" : "");

    li.innerHTML = `
      <span class="module-num">P${proj.module}</span>
      <span class="module-name">${proj.label}</span>
      <span class="status-dot"></span>
    `;

    li.addEventListener("click", () => openProject(proj.file, proj.label));
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
    card.innerHTML = `
      <div class="wmc-num">MODULE ${mod.tag}</div>
      <div class="wmc-title">${mod.label}</div>
      <div class="wmc-status ${completed ? "done" : ""}"></div>
    `;
    card.addEventListener("click", () => openModule(mod.file, mod.label));
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
    card.innerHTML = `
      <div class="wmc-num">PROJECT ${proj.module}</div>
      <div class="wmc-title">${proj.label}</div>
      <span class="difficulty-badge diff-${proj.difficulty.toLowerCase()}">${proj.difficulty}</span>
      <div class="wmc-status ${completed ? "done" : ""}"></div>
    `;
    card.addEventListener("click", () => openProject(proj.file, proj.label));
    grid.appendChild(card);
  });
}

// ── Progress bar ────────────────────────────────────────────────
function updateProgressBar() {
  const total = MODULE_META.length;
  const done  = MODULE_META.filter((m) => state.progress[m.file] === "done").length;

  $("progress-count").textContent = done;
  $("module-total").textContent   = total;
  $("progress-bar-fill").style.width = `${(done / total) * 100}%`;
}

// ── Open module ──────────────────────────────────────────────────
async function openModule(file, label) {
  state.currentFile = file;
  state.currentType = "module";

  document.querySelectorAll("#module-list li").forEach((li) =>
    li.classList.toggle("active", li.dataset.file === file));
  document.querySelectorAll("#project-list li").forEach((li) =>
    li.classList.remove("active"));

  $("breadcrumb").textContent = label;
  $("welcome-screen").classList.add("hidden");
  $("doc-view").classList.remove("hidden");

  const btn    = $("btn-complete");
  const isDone = state.progress[file] === "done";
  btn.classList.remove("hidden");
  btn.textContent = isDone ? "✓ COMPLETED" : "MARK COMPLETE";
  btn.className   = "complete-btn" + (isDone ? " done" : "");

  $("doc-rendered").innerHTML = `<p style="font-family:monospace;color:#888;">Loading ${file}…</p>`;

  try {
    const res = await fetch(`docs/${file}`);
    if (!res.ok) {
      if (res.status === 404) { renderNotReady(file, label); return; }
      throw new Error(`HTTP ${res.status}`);
    }
    renderMarkdown(await res.text());
  } catch (err) {
    $("doc-rendered").innerHTML = `
      <div style="border:3px solid red;padding:20px;font-family:monospace;">
        <strong>Error loading ${file}</strong><br>${err.message}
      </div>`;
  }
}

// ── Open project ─────────────────────────────────────────────────
async function openProject(file, label) {
  state.currentFile = file;
  state.currentType = "project";

  document.querySelectorAll("#module-list li").forEach((li) =>
    li.classList.remove("active"));
  document.querySelectorAll("#project-list li").forEach((li) =>
    li.classList.toggle("active", li.dataset.file === file));

  $("breadcrumb").textContent = "PROJECT — " + label;
  $("welcome-screen").classList.add("hidden");
  $("doc-view").classList.remove("hidden");

  const btn    = $("btn-complete");
  const isDone = state.projectsProgress[file] === "done";
  btn.classList.remove("hidden");
  btn.textContent = isDone ? "✓ COMPLETED" : "MARK COMPLETE";
  btn.className   = "complete-btn" + (isDone ? " done" : "");

  $("doc-rendered").innerHTML = `<p style="font-family:monospace;color:#888;">Loading ${file}…</p>`;

  try {
    const res = await fetch(`docs/projects/${file}`);
    if (!res.ok) {
      if (res.status === 404) { renderNotReady(file, label); return; }
      throw new Error(`HTTP ${res.status}`);
    }
    renderMarkdown(await res.text());
  } catch (err) {
    $("doc-rendered").innerHTML = `
      <div style="border:3px solid red;padding:20px;font-family:monospace;">
        <strong>Error loading ${file}</strong><br>${err.message}
      </div>`;
  }
}

// ── Render markdown ──────────────────────────────────────────────
function renderMarkdown(md) {
  $("doc-rendered").innerHTML = marked.parse(md);

  $("doc-rendered").querySelectorAll("pre code").forEach((block) => {
    hljs.highlightElement(block);
  });

  // Intercept .md links — route modules and projects in-app
  $("doc-rendered").querySelectorAll("a[href]").forEach((a) => {
    const basename = a.getAttribute("href").split("/").pop();
    const mod  = MODULE_META.find((m) => m.file === basename);
    const proj = PROJECT_META.find((p) => p.file === basename);
    if (mod) {
      a.addEventListener("click", (e) => { e.preventDefault(); openModule(mod.file, mod.label); });
    } else if (proj) {
      a.addEventListener("click", (e) => { e.preventDefault(); openProject(proj.file, proj.label); });
    }
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
      <p style="font-family:monospace;color:#555;margin-bottom:20px;">
        This content has not been generated yet.
      </p>
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

  if (isDone) { delete progressObj[state.currentFile]; }
  else        { progressObj[state.currentFile] = "done"; }

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
  document.querySelectorAll(`#${listId} li`).forEach((li) => {
    if (li.dataset.file === state.currentFile) {
      li.classList.toggle("completed", nowDone);
    }
  });
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
  buildWelcomeGrid();
  buildProjectsGrid();
  updateProgressBar();
}

// ── Roadmap button ───────────────────────────────────────────────
$("btn-roadmap").addEventListener("click", () => {
  state.currentFile = null;
  state.currentType = null;
  document.querySelectorAll("#module-list li").forEach((li) => li.classList.remove("active"));
  document.querySelectorAll("#project-list li").forEach((li) => li.classList.remove("active"));
  $("breadcrumb").textContent = "Roadmap";
  $("welcome-screen").classList.remove("hidden");
  $("doc-view").classList.add("hidden");
  $("btn-complete").classList.add("hidden");
  buildWelcomeGrid();
  buildProjectsGrid();
});

// ── Keyboard shortcuts ───────────────────────────────────────────
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    if (state.currentFile) toggleComplete();
  }
  if (e.key === "Escape") { $("btn-roadmap").click(); }
});

// ── Init ─────────────────────────────────────────────────────────
function init() {
  buildNav();
  buildProjectNav();
  buildWelcomeGrid();
  buildProjectsGrid();
  updateProgressBar();

  if (window.location.hash) {
    const hashFile = window.location.hash.slice(1);
    const match    = MODULE_META.find((m) => m.file === hashFile);
    const projMatch = PROJECT_META.find((p) => p.file === hashFile);
    if (match)     openModule(match.file, match.label);
    else if (projMatch) openProject(projMatch.file, projMatch.label);
  }
}

init();
