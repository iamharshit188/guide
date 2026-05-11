"use strict";

// ── Constants ──────────────────────────────────────────────────
const LS_KEY = "aiml_platform_progress";

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
];

// ── State ───────────────────────────────────────────────────────
let state = {
  currentFile: null,
  progress: loadProgress(),
};

// ── Persistence ─────────────────────────────────────────────────
function loadProgress() {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY)) || {};
  } catch {
    return {};
  }
}

function saveProgress() {
  localStorage.setItem(LS_KEY, JSON.stringify(state.progress));
}

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

// ── Build sidebar nav ────────────────────────────────────────────
function buildNav() {
  const ul = $("module-list");
  ul.innerHTML = "";

  MODULE_META.forEach((mod, idx) => {
    const completed = state.progress[mod.file] === "done";
    const li = document.createElement("li");
    li.dataset.file = mod.file;
    li.dataset.idx = idx;
    li.className = completed ? "completed" : "";

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

// ── Build welcome grid ───────────────────────────────────────────
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

// ── Progress bar ────────────────────────────────────────────────
function updateProgressBar() {
  const total = MODULE_META.length;
  const done = MODULE_META.filter((m) => state.progress[m.file] === "done").length;

  $("progress-count").textContent = done;
  $("module-total").textContent = total;
  $("progress-bar-fill").style.width = `${(done / total) * 100}%`;
}

// ── Open module ──────────────────────────────────────────────────
async function openModule(file, label) {
  state.currentFile = file;

  // Update nav active state
  document.querySelectorAll("#module-list li").forEach((li) => {
    li.classList.toggle("active", li.dataset.file === file);
  });

  $("breadcrumb").textContent = label;
  $("welcome-screen").classList.add("hidden");
  $("doc-view").classList.remove("hidden");

  // Complete button state
  const btn = $("btn-complete");
  btn.classList.remove("hidden");
  const isDone = state.progress[file] === "done";
  btn.textContent = isDone ? "✓ COMPLETED" : "MARK COMPLETE";
  btn.className = "complete-btn" + (isDone ? " done" : "");

  // Fetch and render markdown
  $("doc-rendered").innerHTML = `<p style="font-family:monospace;color:#888;">Loading ${file}…</p>`;

  try {
    const res = await fetch(`docs/${file}`);
    if (!res.ok) {
      if (res.status === 404) {
        renderNotReady(file, label);
        return;
      }
      throw new Error(`HTTP ${res.status}`);
    }
    const md = await res.text();
    renderMarkdown(md);
  } catch (err) {
    $("doc-rendered").innerHTML = `
      <div style="border:3px solid red;padding:20px;font-family:monospace;">
        <strong>Error loading ${file}</strong><br>${err.message}
      </div>`;
  }
}

// ── Render markdown ──────────────────────────────────────────────
function renderMarkdown(md) {
  const html = marked.parse(md);
  $("doc-rendered").innerHTML = html;

  // Re-run syntax highlighting on injected code blocks
  $("doc-rendered").querySelectorAll("pre code").forEach((block) => {
    hljs.highlightElement(block);
  });

  // Intercept .md links so they open inside the app instead of navigating away
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

  // Re-render MathJax
  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetPromise([$("doc-rendered")]).catch(console.error);
  }

  // Scroll to top
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
        This module has not been generated yet. Complete the current module
        and request the next one to continue.
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
  const isDone = state.progress[state.currentFile] === "done";

  if (isDone) {
    delete state.progress[state.currentFile];
  } else {
    state.progress[state.currentFile] = "done";
  }

  saveProgress();
  buildNav();
  buildWelcomeGrid();

  const btn = $("btn-complete");
  const nowDone = state.progress[state.currentFile] === "done";
  btn.textContent = nowDone ? "✓ COMPLETED" : "MARK COMPLETE";
  btn.className = "complete-btn" + (nowDone ? " done" : "");

  // Update sidebar item class
  document.querySelectorAll("#module-list li").forEach((li) => {
    if (li.dataset.file === state.currentFile) {
      li.classList.toggle("completed", nowDone);
    }
  });
}

// ── Reset all progress ───────────────────────────────────────────
function resetProgress() {
  if (!confirm("Reset all progress? This cannot be undone.")) return;
  state.progress = {};
  saveProgress();
  buildNav();
  buildWelcomeGrid();
  updateProgressBar();
}

// ── Roadmap button ───────────────────────────────────────────────
$("btn-roadmap").addEventListener("click", () => {
  state.currentFile = null;
  document.querySelectorAll("#module-list li").forEach((li) => li.classList.remove("active"));
  $("breadcrumb").textContent = "Roadmap";
  $("welcome-screen").classList.remove("hidden");
  $("doc-view").classList.add("hidden");
  $("btn-complete").classList.add("hidden");
  buildWelcomeGrid();
});

// ── Keyboard shortcuts ───────────────────────────────────────────
document.addEventListener("keydown", (e) => {
  // Cmd/Ctrl + Enter to mark complete
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    if (state.currentFile) toggleComplete();
  }

  // Escape → roadmap
  if (e.key === "Escape") {
    $("btn-roadmap").click();
  }
});

// ── Init ─────────────────────────────────────────────────────────
function init() {
  buildNav();
  buildWelcomeGrid();
  updateProgressBar();

  // Auto-open first incomplete module on load
  const firstIncomplete = MODULE_META.find((m) => state.progress[m.file] !== "done");
  if (firstIncomplete && window.location.hash) {
    const hashFile = window.location.hash.slice(1);
    const match = MODULE_META.find((m) => m.file === hashFile);
    if (match) openModule(match.file, match.label);
  }
}

init();
