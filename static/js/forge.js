// static/js/forge.js — NeuralForge: Train, QLoRA, RL Loop, Ingest, Personalizer, Self-Train
// Each feature lives in its own modal, accessible from the main nav rail.

import * as Modals from './modalManager.js';

const API = '';

// ── Shared training helpers ───────────────────────────────────────────────────

function _esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
}

function _trainFormHtml(type) {
  let h = '<div class="settings-col" style="gap:8px;margin-top:8px;">';
  h += `<div class="settings-row"><label class="settings-label" style="min-width:110px">Base Model</label><input type="text" class="memory-search-input" id="fg-${type}-model-id" placeholder="meta-llama/Llama-3.2-1B" style="flex:1"></div>`;
  h += `<div class="settings-row"><label class="settings-label" style="min-width:110px">Dataset</label><input type="text" class="memory-search-input" id="fg-${type}-dataset" placeholder="HF dataset id or local JSONL path" style="flex:1"></div>`;
  h += `<div class="settings-row"><label class="settings-label" style="min-width:110px">Output Name</label><input type="text" class="memory-search-input" id="fg-${type}-output-name" placeholder="${type}-adapter" style="flex:1"></div>`;
  h += _gpuRow(type);
  h += '<details style="margin-top:2px"><summary style="cursor:pointer;font-size:11px;opacity:0.6;list-style:none;display:flex;align-items:center;gap:4px"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="6 9 12 15 18 9"/></svg>Hyperparameters</summary>';
  h += '<div class="settings-col" style="gap:6px;margin-top:8px;">';
  h += `<div class="settings-row"><label class="settings-label" style="min-width:110px">LoRA Rank</label><input type="number" class="memory-search-input" id="fg-${type}-lora-rank" value="16" min="4" max="128" style="width:70px"></div>`;
  h += `<div class="settings-row"><label class="settings-label" style="min-width:110px">Learning Rate</label><input type="text" class="memory-search-input" id="fg-${type}-lr" value="${type==='sft'?'2e-4':'1e-4'}" style="width:80px"></div>`;
  h += `<div class="settings-row"><label class="settings-label" style="min-width:110px">Epochs</label><input type="number" class="memory-search-input" id="fg-${type}-epochs" value="${type==='cpt'?1:3}" min="1" max="20" style="width:70px"></div>`;
  h += `<div class="settings-row"><label class="settings-label" style="min-width:110px">Batch Size</label><input type="number" class="memory-search-input" id="fg-${type}-batch-size" value="2" min="1" max="16" style="width:70px"></div>`;
  h += `<div class="settings-row"><label class="settings-label" style="min-width:110px">Seed</label><input type="number" class="memory-search-input" id="fg-${type}-seed" value="42" style="width:80px"></div>`;
  h += '</div></details>';
  h += '</div>';
  h += `<button class="cookbook-btn" id="fg-${type}-start-btn" style="margin-top:10px;width:100%">Start ${type.toUpperCase()}</button>`;
  return h;
}

function _statusHtml(prefix) {
  return `<div id="${prefix}-job-status" style="margin-top:10px;display:none">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
      <span id="${prefix}-job-badge" style="font-size:11px;padding:2px 7px;border-radius:99px;background:var(--panel);border:1px solid var(--border);">idle</span>
      <button id="${prefix}-job-stop" class="cookbook-btn" style="padding:3px 8px;font-size:11px;opacity:0.7" disabled>Stop</button>
    </div>
    <div id="${prefix}-job-log" style="font-family:monospace;font-size:10px;max-height:200px;overflow-y:auto;background:var(--panel);border:1px solid var(--border);border-radius:4px;padding:6px 8px;white-space:pre-wrap;word-break:break-word;"></div>
  </div>`;
}

function _field(id) {
  const el = document.getElementById(id);
  return el ? el.value.trim() : '';
}

// ── GPU selector ──────────────────────────────────────────────────────────────
// On a heterogeneous box you want to pin each job to one card (per-GPU
// independent jobs) rather than spanning all of them. The row is populated from
// /api/training/gpus; it degrades to an Auto-only select if nvidia-smi is absent.
function _gpuRow(idBase) {
  return `<div class="settings-row"><label class="settings-label" style="min-width:110px">GPU</label>`
    + `<select class="memory-search-input fg-gpu-select" id="fg-${idBase}-gpu" style="flex:1">`
    + `<option value="">Auto (all GPUs)</option></select></div>`;
}

// Returns the chosen GPU index as a number, or null for "Auto".
function _gpuVal(idBase) {
  const el = document.getElementById(`fg-${idBase}-gpu`);
  if (!el || el.value === '') return null;
  const n = parseInt(el.value, 10);
  return Number.isNaN(n) ? null : n;
}

let _gpuCache = null;
async function _populateGpuSelects(root) {
  try {
    if (!_gpuCache) {
      const r = await fetch(`${API}/api/training/gpus`, { credentials: 'same-origin' });
      _gpuCache = r.ok ? ((await r.json()).gpus || []) : [];
    }
    (root || document).querySelectorAll('select.fg-gpu-select').forEach(sel => {
      if (sel.dataset.filled === '1') return;
      for (const g of _gpuCache) {
        const gb = Math.round((g.memory_mb || 0) / 1024);
        const warn = g.bf16 ? '' : ' · fp16-only';
        const opt = document.createElement('option');
        opt.value = String(g.index);
        opt.textContent = `GPU ${g.index}: ${g.name} (${gb}GB${warn})`;
        sel.appendChild(opt);
      }
      sel.dataset.filled = '1';
    });
  } catch (_) { /* leave Auto-only */ }
}

let _pollers = {};

async function _startJob(payload) {
  const res = await fetch(`${API}/api/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function _stopJob(jobId) {
  await fetch(`${API}/api/training/stop/${jobId}`, { method: 'POST', credentials: 'same-origin' });
}

function _streamLogs(jobId, logEl, badgeEl, stopBtn) {
  if (_pollers[jobId]) return;
  let offset = 0;
  _pollers[jobId] = setInterval(async () => {
    try {
      const res = await fetch(`${API}/api/training/logs/${jobId}?offset=${offset}`, { credentials: 'same-origin' });
      if (!res.ok) return;
      const lines = (await res.text()).split('\n');
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const d = JSON.parse(line.slice(6));
          if (d.line !== undefined) {
            if (logEl) { logEl.textContent += d.line + '\n'; logEl.scrollTop = logEl.scrollHeight; }
            offset += new TextEncoder().encode(d.line + '\n').length;
            const m = d.line.match(/Level (\d): (\w+) complete/);
            if (m) {
              document.querySelector(`.rl-level-pip[data-level="${m[2].toLowerCase()}"]`)?.classList.add('rl-level-done');
            }
          }
          if (d.done) {
            clearInterval(_pollers[jobId]);
            delete _pollers[jobId];
            if (badgeEl) { badgeEl.textContent = 'done'; badgeEl.style.color = 'var(--green,#50fa7b)'; }
            if (stopBtn) stopBtn.disabled = true;
          }
        } catch (_) {}
      }
    } catch (_) {}
  }, 800);
}

function _wireTrainBtn(type, body) {
  const btn = body.querySelector(`#fg-${type}-start-btn`);
  if (!btn) return;
  btn.addEventListener('click', async () => {
    const modelId = _field(`fg-${type}-model-id`);
    if (!modelId) { alert('Please enter a base model ID.'); return; }
    const statusDiv = body.querySelector(`#fg-train-job-status`);
    const logEl     = body.querySelector(`#fg-train-job-log`);
    const badgeEl   = body.querySelector(`#fg-train-job-badge`);
    const stopBtn   = body.querySelector(`#fg-train-job-stop`);
    if (statusDiv) statusDiv.style.display = '';
    if (logEl) logEl.textContent = '';
    if (badgeEl) { badgeEl.textContent = 'starting…'; badgeEl.style.color = 'var(--fg)'; }
    btn.disabled = true;
    try {
      const result = await _startJob({
        job_type: type,
        model_id: modelId,
        dataset_path: _field(`fg-${type}-dataset`) || null,
        seed: parseInt(_field(`fg-${type}-seed`) || '42'),
        lora_rank: parseInt(_field(`fg-${type}-lora-rank`) || '16'),
        lr: parseFloat(_field(`fg-${type}-lr`) || '2e-4'),
        epochs: parseInt(_field(`fg-${type}-epochs`) || '3'),
        batch_size: parseInt(_field(`fg-${type}-batch-size`) || '2'),
        output_name: _field(`fg-${type}-output-name`) || null,
        gpu: _gpuVal(type),
      });
      if (badgeEl) { badgeEl.textContent = 'running'; badgeEl.style.color = 'var(--green,#50fa7b)'; }
      if (stopBtn) { stopBtn.disabled = false; stopBtn.onclick = () => _stopJob(result.job_id); }
      _streamLogs(result.job_id, logEl, badgeEl, stopBtn);
    } catch (e) {
      if (badgeEl) { badgeEl.textContent = 'error'; badgeEl.style.color = 'var(--red)'; }
      if (logEl) logEl.textContent = String(e);
      btn.disabled = false;
    }
  });
}

// ── Generic modal builder ─────────────────────────────────────────────────────

function _makeModal(id, title, iconSvg, width='min(700px,92vw)') {
  const el = document.createElement('div');
  el.id = id;
  el.className = 'modal hidden';
  el.innerHTML = `
    <div class="modal-content" role="dialog" aria-label="${title}" style="width:${width};height:92vh;max-height:92vh;background:var(--bg);display:flex;flex-direction:column;">
      <div class="modal-header">
        <h4 style="margin:0;margin-right:auto">${iconSvg}${title}</h4>
        <button class="close-btn" id="close-${id}" aria-label="Close ${title}">✖</button>
      </div>
      <div class="modal-body" id="${id}-body" style="flex:1;overflow-y:auto;padding:16px;"></div>
    </div>`;
  document.body.appendChild(el);
  document.getElementById(`close-${id}`).addEventListener('click', () => _closeModal(id));
  return el;
}

function _openModal(id, railBtnId, buildFn) {
  const modal = document.getElementById(id);
  if (!modal) return;
  if (Modals.isMinimized(id)) { Modals.restore(id); return; }
  if (!modal.classList.contains('hidden')) return;
  modal.classList.remove('hidden');
  Modals.register(id, {
    railBtnId,
    closeFn: () => _closeModal(id),
    restoreFn: () => {},
  });
  const body = document.getElementById(`${id}-body`);
  if (body && body.dataset.built !== '1') {
    buildFn(body);
    body.dataset.built = '1';
  }
}

function _closeModal(id) {
  const modal = document.getElementById(id);
  if (!modal) return;
  modal.classList.add('hidden');
  Modals.unregister(id);
}

// ── TRAIN MODAL ───────────────────────────────────────────────────────────────

function _buildTrain(body) {
  let html = '<div class="admin-card">';
  html += '<h2><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-1px;margin-right:5px;opacity:0.7"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>Fine-Tuning</h2>';
  html += '<p class="memory-desc" style="margin-bottom:10px;">Train a model with SFT, DPO, or CPT using QLoRA. Requires TRL, PEFT, and bitsandbytes.<br><code style="font-size:10px">pip install trl peft transformers datasets bitsandbytes</code></p>';
  html += '<div class="cookbook-subtabs" id="fg-train-subtabs">';
  html += '<button class="cookbook-subtab active" data-fg-train-tab="sft">SFT</button>';
  html += '<button class="cookbook-subtab" data-fg-train-tab="dpo">DPO</button>';
  html += '<button class="cookbook-subtab" data-fg-train-tab="cpt">CPT</button>';
  html += '</div>';
  html += '<div class="fg-train-subpanel" data-fg-train-panel="sft"><p class="memory-desc" style="margin:6px 0 8px">Supervised Fine-Tuning: teach instruction following from a labeled dataset.</p>' + _trainFormHtml('sft') + '</div>';
  html += '<div class="fg-train-subpanel" data-fg-train-panel="dpo" style="display:none"><p class="memory-desc" style="margin:6px 0 8px">Direct Preference Optimization: align a model to prefer good over bad responses.</p>' + _trainFormHtml('dpo') + '</div>';
  html += '<div class="fg-train-subpanel" data-fg-train-panel="cpt" style="display:none"><p class="memory-desc" style="margin:6px 0 8px">Continued Pre-Training: inject raw domain text before instruction tuning.</p>' + _trainFormHtml('cpt') + '</div>';
  html += _statusHtml('fg-train');
  html += '</div>';
  body.innerHTML = html;
  _populateGpuSelects(body);

  body.querySelectorAll('.cookbook-subtab[data-fg-train-tab]').forEach(tab => {
    tab.addEventListener('click', () => {
      body.querySelectorAll('.cookbook-subtab[data-fg-train-tab]').forEach(t => t.classList.toggle('active', t === tab));
      body.querySelectorAll('.fg-train-subpanel[data-fg-train-panel]').forEach(p => {
        p.style.display = p.dataset.fgTrainPanel === tab.dataset.fgTrainTab ? '' : 'none';
      });
    });
  });
  _wireTrainBtn('sft', body);
  _wireTrainBtn('dpo', body);
  _wireTrainBtn('cpt', body);
}

export function openTrain() {
  _openModal('forge-train-modal', 'rail-forge-train', _buildTrain);
}

// ── QLORA MODAL ───────────────────────────────────────────────────────────────

function _buildQLoRA(body) {
  let html = '<div class="admin-card">';
  html += '<h2><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-1px;margin-right:5px;opacity:0.7"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3"/></svg>QLoRA Merge</h2>';
  html += '<p class="memory-desc" style="margin-bottom:10px;">Merge a trained LoRA adapter into the base model to produce a single deployable model.</p>';
  html += '<div class="settings-col" style="gap:8px;">';
  html += '<div class="settings-row"><label class="settings-label" style="min-width:110px">Base Model</label><input type="text" class="memory-search-input" id="fg-qlora-base-model" placeholder="meta-llama/Llama-3.2-1B" style="flex:1"></div>';
  html += '<div class="settings-row"><label class="settings-label" style="min-width:110px">Adapter Path</label><input type="text" class="memory-search-input" id="fg-qlora-adapter-path" placeholder="data/lora_adapters/sft-xxxxx" style="flex:1"></div>';
  html += '<div class="settings-row"><label class="settings-label" style="min-width:110px">Output Name</label><input type="text" class="memory-search-input" id="fg-qlora-output-name" placeholder="my-merged-model" style="flex:1"></div>';
  html += _gpuRow('qlora');
  html += '</div>';
  html += '<button class="cookbook-btn" id="fg-qlora-start-btn" style="margin-top:12px;width:100%">Merge Adapter</button>';
  html += _statusHtml('fg-qlora');
  html += '</div>';
  body.innerHTML = html;
  _populateGpuSelects(body);

  const btn = body.querySelector('#fg-qlora-start-btn');
  const logEl   = body.querySelector('#fg-qlora-job-log');
  const badgeEl = body.querySelector('#fg-qlora-job-badge');
  const stopBtn = body.querySelector('#fg-qlora-job-stop');
  const statusDiv = body.querySelector('#fg-qlora-job-status');

  btn.addEventListener('click', async () => {
    const modelId     = _field('fg-qlora-base-model');
    const adapterPath = _field('fg-qlora-adapter-path');
    if (!modelId || !adapterPath) { alert('Please enter a base model and adapter path.'); return; }
    statusDiv.style.display = '';
    if (logEl) logEl.textContent = '';
    if (badgeEl) { badgeEl.textContent = 'starting…'; badgeEl.style.color = 'var(--fg)'; }
    btn.disabled = true;
    try {
      const result = await _startJob({
        job_type: 'qlora',
        model_id: modelId,
        adapter_path: adapterPath,
        output_name: _field('fg-qlora-output-name') || null,
        gpu: _gpuVal('qlora'),
      });
      if (badgeEl) { badgeEl.textContent = 'running'; badgeEl.style.color = 'var(--green,#50fa7b)'; }
      if (stopBtn) { stopBtn.disabled = false; stopBtn.onclick = () => _stopJob(result.job_id); }
      _streamLogs(result.job_id, logEl, badgeEl, stopBtn);
    } catch (e) {
      if (badgeEl) { badgeEl.textContent = 'error'; badgeEl.style.color = 'var(--red)'; }
      if (logEl) logEl.textContent = String(e);
      btn.disabled = false;
    }
  });
}

export function openQLoRA() {
  _openModal('forge-qlora-modal', 'rail-forge-qlora', _buildQLoRA);
}

// ── RL LOOP MODAL ─────────────────────────────────────────────────────────────

function _buildRLLoop(body) {
  let html = '<div class="admin-card">';
  html += '<h2><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-1px;margin-right:5px;opacity:0.7"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>RL Loop <span style="font-size:10px;opacity:0.55;font-weight:normal;margin-left:4px;">Student → PhD</span></h2>';
  html += '<p class="memory-desc" style="margin-bottom:10px;">Trains a model to become a knowledge expert using GRPO reinforcement learning. Progresses through 5 curriculum levels: Student → Intermediate → Advanced → Expert → PhD.</p>';
  html += '<div class="settings-col" style="gap:8px;">';
  html += '<div class="settings-row"><label class="settings-label" style="min-width:110px">Base Model</label><input type="text" class="memory-search-input" id="fg-rl-model-id" placeholder="meta-llama/Llama-3.2-1B" style="flex:1"><button class="cookbook-btn" id="fg-rl-model-pick-btn" style="padding:4px 8px;margin-left:4px;flex-shrink:0" title="Browse downloaded models">Browse</button></div>';
  html += '<div class="settings-row"><label class="settings-label" style="min-width:110px">Book</label><select class="memory-search-input" id="fg-rl-book-select" style="flex:1"><option value="">— select or upload —</option></select><button class="cookbook-btn" id="fg-rl-book-upload-btn" style="padding:4px 8px;margin-left:4px;flex-shrink:0">Upload</button><input type="file" id="fg-rl-book-file-input" accept=".pdf,.epub,.txt,.md" style="display:none"></div>';
  html += '<div class="settings-row"><label class="settings-label" style="min-width:110px">Seed</label><input type="number" class="memory-search-input" id="fg-rl-seed" value="42" min="0" max="999999" style="width:90px"><span style="opacity:0.5;font-size:11px;margin-left:8px">Reproducibility seed</span></div>';
  html += _gpuRow('rl');
  html += '<details style="margin-top:4px"><summary style="cursor:pointer;font-size:11px;opacity:0.6;list-style:none;display:flex;align-items:center;gap:4px"><svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="6 9 12 15 18 9"/></svg>Hyperparameters</summary>';
  html += '<div class="settings-col" style="gap:6px;margin-top:8px;">';
  html += '<div class="settings-row"><label class="settings-label" style="min-width:110px">LoRA Rank</label><input type="number" class="memory-search-input" id="fg-rl-lora-rank" value="16" min="4" max="128" style="width:70px"></div>';
  html += '<div class="settings-row"><label class="settings-label" style="min-width:110px">Learning Rate</label><input type="text" class="memory-search-input" id="fg-rl-lr" value="2e-5" style="width:80px"></div>';
  html += '<div class="settings-row"><label class="settings-label" style="min-width:110px">Epochs/Level</label><input type="number" class="memory-search-input" id="fg-rl-epochs" value="1" min="1" max="10" style="width:70px"></div>';
  html += '<div class="settings-row"><label class="settings-label" style="min-width:110px">Output Name</label><input type="text" class="memory-search-input" id="fg-rl-output-name" placeholder="expert-on-book" style="flex:1"></div>';
  html += '</div></details>';
  html += '</div>';
  html += '<div style="display:flex;gap:8px;margin-top:12px;">';
  html += '<button class="cookbook-btn" id="fg-rl-start-btn" style="flex:1">Start RL Loop</button>';
  html += '<button class="cookbook-btn" id="fg-rl-stop-btn" style="padding:6px 12px;opacity:0.7" disabled>Stop</button>';
  html += '</div>';
  html += '<div id="fg-rl-level-progress" style="margin-top:10px;display:none">';
  html += '<div style="font-size:11px;opacity:0.7;margin-bottom:4px">Curriculum Progress</div>';
  html += '<div class="rl-level-bar">';
  for (const lvl of ['Student','Intermediate','Advanced','Expert','PhD']) {
    html += `<div class="rl-level-pip" data-level="${lvl.toLowerCase()}" title="${lvl}">${lvl[0]}</div>`;
  }
  html += '</div></div>';
  html += _statusHtml('fg-rl');
  html += '</div>';
  body.innerHTML = html;
  _populateGpuSelects(body);

  const rlBookSelect = body.querySelector('#fg-rl-book-select');
  const rlUploadBtn  = body.querySelector('#fg-rl-book-upload-btn');
  const rlFileInput  = body.querySelector('#fg-rl-book-file-input');
  const rlStartBtn   = body.querySelector('#fg-rl-start-btn');
  const rlStopBtn    = body.querySelector('#fg-rl-stop-btn');
  const logEl        = body.querySelector('#fg-rl-job-log');
  const badgeEl      = body.querySelector('#fg-rl-job-badge');
  const statusDiv    = body.querySelector('#fg-rl-job-status');
  const levelBar     = body.querySelector('#fg-rl-level-progress');

  fetch(`${API}/api/training/books`, { credentials: 'same-origin' })
    .then(r => r.json()).then(data => {
      (data.books || []).forEach(b => {
        const opt = document.createElement('option');
        opt.value = b.filename;
        opt.textContent = `${b.filename} (${b.size_mb} MB)`;
        rlBookSelect.appendChild(opt);
      });
    }).catch(() => {});

  rlUploadBtn.addEventListener('click', () => rlFileInput.click());
  rlFileInput.addEventListener('change', async () => {
    const file = rlFileInput.files[0];
    if (!file) return;
    const fd = new FormData(); fd.append('file', file);
    rlUploadBtn.textContent = 'Uploading…'; rlUploadBtn.disabled = true;
    try {
      const res = await fetch(`${API}/api/training/upload-book`, { method: 'POST', credentials: 'same-origin', body: fd });
      const data = await res.json();
      if (data.ok && rlBookSelect) {
        const opt = document.createElement('option');
        opt.value = data.filename; opt.textContent = `${data.filename} (${data.size_mb} MB)`;
        rlBookSelect.appendChild(opt); rlBookSelect.value = data.filename;
      }
    } catch (e) { alert('Upload failed: ' + e); }
    finally { rlUploadBtn.textContent = 'Upload'; rlUploadBtn.disabled = false; rlFileInput.value = ''; }
  });

  rlStartBtn.addEventListener('click', async () => {
    const modelId = _field('fg-rl-model-id');
    const book = rlBookSelect ? rlBookSelect.value : '';
    if (!modelId) { alert('Please enter a base model ID.'); return; }
    if (!book) { alert('Please select or upload a book.'); return; }
    statusDiv.style.display = ''; levelBar.style.display = '';
    if (logEl) logEl.textContent = '';
    if (badgeEl) { badgeEl.textContent = 'starting…'; badgeEl.style.color = 'var(--fg)'; }
    rlStartBtn.disabled = true; rlStopBtn.disabled = false;
    body.querySelectorAll('.rl-level-pip').forEach(p => p.classList.remove('rl-level-done','rl-level-active'));
    try {
      const result = await _startJob({
        job_type: 'rl_loop',
        model_id: modelId,
        book_filename: book,
        seed: parseInt(_field('fg-rl-seed') || '42'),
        lora_rank: parseInt(_field('fg-rl-lora-rank') || '16'),
        lr: parseFloat(_field('fg-rl-lr') || '2e-5'),
        epochs: parseInt(_field('fg-rl-epochs') || '1'),
        batch_size: 1, grad_accum: 8,
        output_name: _field('fg-rl-output-name') || null,
        gpu: _gpuVal('rl'),
      });
      if (badgeEl) { badgeEl.textContent = 'running'; badgeEl.style.color = 'var(--green,#50fa7b)'; }
      _streamLogs(result.job_id, logEl, badgeEl, rlStopBtn);
    } catch (e) {
      if (badgeEl) { badgeEl.textContent = 'error'; badgeEl.style.color = 'var(--red)'; }
      if (logEl) logEl.textContent = String(e);
      rlStartBtn.disabled = false; rlStopBtn.disabled = true;
    }
  });

  rlStopBtn.addEventListener('click', async () => {
    const state = await fetch(`${API}/api/training/state`, { credentials: 'same-origin' }).then(r=>r.json()).catch(()=>({jobs:[]}));
    const running = (state.jobs||[]).find(j=>j.status==='running');
    if (running) await _stopJob(running.id);
    rlStartBtn.disabled = false; rlStopBtn.disabled = true;
  });
}

export function openRLLoop() {
  _openModal('forge-rlloop-modal', 'rail-forge-rlloop', _buildRLLoop);
}

// ── INGEST MODAL ──────────────────────────────────────────────────────────────

function _updateIngestUI(body, s) {
  const overallEl = body.querySelector('#fg-ingest-overall');
  const labelEl   = body.querySelector('#fg-ingest-overall-label-text');
  const pctEl     = body.querySelector('#fg-ingest-overall-pct');
  const barFill   = body.querySelector('#fg-ingest-overall-bar');
  const statFiles = body.querySelector('#fg-ingest-stat-files');
  const statChunks= body.querySelector('#fg-ingest-stat-chunks');
  const fileList  = body.querySelector('#fg-ingest-file-list');

  if (!overallEl) return;
  // Show the Stop button only while a job is actively running.
  const stopBtn = body.querySelector('#fg-ingest-stop-btn');
  if (stopBtn) {
    stopBtn.style.display = s.running ? '' : 'none';
    if (s.running) { stopBtn.disabled = false; stopBtn.textContent = 'Stop'; }
  }
  if (!s.running && !(s.files || []).length) { overallEl.style.display = 'none'; return; }
  overallEl.style.display = '';

  const total = s.total_files || 0;
  const done  = s.done_files  || 0;
  const pct   = total > 0 ? Math.round((done / total) * 100) : (s.running ? 10 : 100);
  if (labelEl)    labelEl.textContent  = s.running ? 'Processing…' : 'Done';
  if (pctEl)      pctEl.textContent    = pct + '%';
  if (barFill)    barFill.style.width  = pct + '%';
  if (statFiles)  statFiles.textContent = `${done}/${total}`;
  if (statChunks) statChunks.textContent = s.total_chunks || 0;

  if (fileList) {
    fileList.innerHTML = (s.files || []).map(f => {
      const icon = f.status === 'done' ? '✓' : f.status === 'error' ? '✗' : '…';
      const color = f.status === 'done' ? 'var(--green,#50fa7b)' : f.status === 'error' ? 'var(--red)' : 'var(--fg)';
      return `<div style="display:flex;align-items:center;gap:6px;padding:4px 0;font-size:12px;border-bottom:1px solid color-mix(in srgb,var(--border) 40%,transparent)">
        <span style="color:${color};flex-shrink:0">${icon}</span>
        <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${_esc(f.name||f.filename||'')}</span>
        <span style="opacity:0.45;font-size:10px;flex-shrink:0">${f.chunks||''} chunks</span>
      </div>`;
    }).join('');
  }
}

let _ingestPollTimer = null;

function _startIngestPoll(body) {
  if (_ingestPollTimer) return;
  _ingestPollTimer = setInterval(async () => {
    const s = await fetch(`${API}/api/ingest/progress`, { credentials: 'same-origin' }).then(r=>r.json()).catch(()=>null);
    if (!s) return;
    _updateIngestUI(body, s);
    if (!s.running) { clearInterval(_ingestPollTimer); _ingestPollTimer = null; }
  }, 1500);
}

async function _submitIngestFiles(body, fileList) {
  if (!fileList.length) return;
  const overallEl = body.querySelector('#fg-ingest-overall');
  if (overallEl) overallEl.style.display = '';
  const labelEl = body.querySelector('#fg-ingest-overall-label-text');
  if (labelEl) labelEl.textContent = `Uploading ${fileList.length} file(s)…`;

  // The backend /upload endpoint takes the whole batch in one multipart request
  // under the field name `files` and kicks off the ingest job itself — there is
  // no separate /start step. Posting files one at a time (or under `file`)
  // tripped a 422 and a 409 "already running", so nothing got ingested.
  const fd = new FormData();
  for (const file of fileList) fd.append('files', file);
  let res = null;
  try {
    res = await fetch(`${API}/api/ingest/upload`, { method: 'POST', credentials: 'same-origin', body: fd });
  } catch (_) { res = null; }
  if (res && res.ok) {
    _startIngestPoll(body);
  } else {
    let msg = 'Ingest failed';
    if (res) { try { msg = (await res.json()).detail || msg; } catch (_) {} }
    if (labelEl) labelEl.textContent = msg;
  }
}

function _buildIngest(body) {
  let html = '<div class="admin-card">';
  html += '<h2><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-1px;margin-right:5px;opacity:0.7"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>RAG Ingest</h2>';
  html += '<p class="memory-desc" style="margin-bottom:10px;">Upload files or folders to index into the RAG knowledge base. Supports PDF, EPUB, TXT, MD, DOCX.</p>';
  html += '<div class="ingest-drop-zone" id="fg-ingest-drop-zone">';
  html += '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="display:block;margin:0 auto 6px"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>';
  html += '<div class="ingest-drop-label">Drop files here, or use the buttons below</div>';
  html += '<div class="ingest-drop-hint">PDF, EPUB, TXT, MD, DOCX — up to 200 MB each</div>';
  html += '</div>';
  html += '<div class="ingest-actions">';
  html += '<button class="cookbook-btn" id="fg-ingest-pick-files" style="flex:1"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" style="vertical-align:-1px;margin-right:4px"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>Pick Files</button>';
  html += '<button class="cookbook-btn" id="fg-ingest-pick-folder" style="flex:1"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" style="vertical-align:-1px;margin-right:4px"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>Pick Folder</button>';
  html += '<button class="cookbook-btn" id="fg-ingest-stop-btn" style="display:none" title="Finish the current file, then stop and defer the rest">Stop</button>';
  html += '<button class="cookbook-btn" id="fg-ingest-clear-btn" style="opacity:0.6" title="Clear results">Clear</button>';
  html += '</div>';
  html += '<input type="file" id="fg-ingest-file-input" multiple accept=".pdf,.epub,.txt,.md,.docx,.rst,.csv" style="display:none">';
  html += '<input type="file" id="fg-ingest-folder-input" multiple webkitdirectory style="display:none">';
  html += '<div class="ingest-overall" id="fg-ingest-overall" style="display:none">';
  html += '<div class="ingest-overall-label"><span id="fg-ingest-overall-label-text">Processing…</span><span id="fg-ingest-overall-pct">0%</span></div>';
  html += '<div class="ingest-bar-track"><div class="ingest-bar-fill" id="fg-ingest-overall-bar"></div></div>';
  html += '<div class="ingest-stats-row"><div class="ingest-stat">Files: <b id="fg-ingest-stat-files">0/0</b></div><div class="ingest-stat">Chunks: <b id="fg-ingest-stat-chunks">0</b></div></div>';
  html += '</div>';
  html += '<div class="ingest-file-list" id="fg-ingest-file-list"></div>';
  html += '</div>';
  body.innerHTML = html;

  const dropZone   = body.querySelector('#fg-ingest-drop-zone');
  const fileInput  = body.querySelector('#fg-ingest-file-input');
  const folderInput= body.querySelector('#fg-ingest-folder-input');
  const pickFiles  = body.querySelector('#fg-ingest-pick-files');
  const pickFolder = body.querySelector('#fg-ingest-pick-folder');
  const clearBtn   = body.querySelector('#fg-ingest-clear-btn');
  const stopBtn    = body.querySelector('#fg-ingest-stop-btn');

  stopBtn.addEventListener('click', async () => {
    stopBtn.disabled = true;
    stopBtn.textContent = 'Stopping…';
    try {
      const r = await fetch(`${API}/api/ingest/stop`, { method: 'POST', credentials: 'same-origin' });
      const d = await r.json().catch(() => ({}));
      if (d.deferred_count) {
        alert(`Stopping after the current file. ${d.deferred_count} file(s) deferred — re-select the same folder later to resume (already-ingested books are skipped).`);
      }
    } catch (_) {}
  });

  pickFiles.addEventListener('click', () => fileInput.click());
  pickFolder.addEventListener('click', () => folderInput.click());
  fileInput.addEventListener('change', () => _submitIngestFiles(body, [...fileInput.files]));
  folderInput.addEventListener('change', () => _submitIngestFiles(body, [...folderInput.files]));
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    const files = [...(e.dataTransfer?.files || [])];
    if (files.length) _submitIngestFiles(body, files);
  });
  clearBtn.addEventListener('click', async () => {
    try {
      const r = await fetch(`${API}/api/ingest/clear-state`, { method: 'POST', credentials: 'same-origin' });
      if (r.ok) _updateIngestUI(body, { files: [], running: false, done_files: 0, total_files: 0, total_chunks: 0 });
      else alert('Cannot clear while job is running');
    } catch (_) {}
  });

  fetch(`${API}/api/ingest/progress`, { credentials: 'same-origin' }).then(r=>r.json()).then(s => {
    _updateIngestUI(body, s);
    if (s.running) _startIngestPoll(body);
  }).catch(() => {});
}

export function openIngest() {
  _openModal('forge-ingest-modal', 'rail-forge-ingest', _buildIngest);
}

// ── PERSONALIZER MODAL ────────────────────────────────────────────────────────

const INTERVIEW_QUESTIONS = [
  "What's your name, and what do you do professionally?",
  "What are the main topics or domains you work in or care deeply about?",
  "What are your biggest ongoing projects right now?",
  "How do you prefer to receive information — detailed explanations, bullet points, or quick summaries?",
  "Are there any recurring tasks or challenges I should know about to help you better?",
  "What are your current goals — personal, professional, or both?",
  "Any strong preferences or things I should always avoid saying or doing?",
  "What time zone are you in, and roughly when are you most active?",
  "Do you have a preferred programming language, framework, or tech stack?",
  "Is there anything else fundamental about who you are that I should always remember?",
];

let _personalizerState = null;

function _buildPersonalizer(body) {
  body.innerHTML = `
    <div style="display:flex;flex-direction:column;height:100%;gap:0;">
      <div class="admin-card" style="flex-shrink:0;margin-bottom:12px;">
        <h2 style="margin-bottom:6px;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-1px;margin-right:5px;opacity:0.7"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>Personalizer</h2>
        <p class="memory-desc">I'll learn about you through a short interview, then mine your past chats for additional facts. Each answer is saved to your memory.</p>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;">
          <button class="cookbook-btn" id="fg-pers-start-interview" style="flex:1">Start Interview</button>
          <button class="cookbook-btn" id="fg-pers-mine-chats" style="flex:1;opacity:0.8">Mine Past Chats</button>
        </div>
      </div>
      <div id="fg-pers-chat" style="flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:10px;padding:4px 0;min-height:0;"></div>
      <div id="fg-pers-input-row" style="display:none;flex-shrink:0;margin-top:10px;display:flex;gap:8px;">
        <input type="text" id="fg-pers-answer" class="memory-search-input" placeholder="Type your answer…" style="flex:1">
        <button class="cookbook-btn" id="fg-pers-send">Send</button>
      </div>
    </div>`;

  _personalizerState = { qIdx: 0, active: false, memories: [] };

  const chatEl    = body.querySelector('#fg-pers-chat');
  const inputRow  = body.querySelector('#fg-pers-input-row');
  const answerEl  = body.querySelector('#fg-pers-answer');
  const sendBtn   = body.querySelector('#fg-pers-send');
  const startBtn  = body.querySelector('#fg-pers-start-interview');
  const mineBtn   = body.querySelector('#fg-pers-mine-chats');

  function _addMsg(role, text) {
    const div = document.createElement('div');
    div.style.cssText = `max-width:85%;padding:10px 13px;border-radius:12px;font-size:13px;line-height:1.5;word-break:break-word;`;
    if (role === 'assistant') {
      div.style.cssText += 'background:color-mix(in srgb,var(--fg) 6%,transparent);border:1px solid color-mix(in srgb,var(--border) 60%,transparent);align-self:flex-start;';
    } else {
      div.style.cssText += 'background:color-mix(in srgb,var(--red) 12%,transparent);border:1px solid color-mix(in srgb,var(--red) 30%,transparent);align-self:flex-end;color:var(--fg);';
    }
    div.textContent = text;
    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  async function _saveMemory(content, category, source) {
    try {
      // The /api/memory/add route requires a `text` field (MemoryAddRequest.text);
      // sending `content` 422s and the interview would silently save nothing.
      await fetch(`${API}/api/memory/add`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ text: content, category: category || 'identity', source: source || 'personalizer' }),
      });
    } catch (_) {}
  }

  function _finishInterview() {
    _addMsg('assistant', "That's everything I need for now — I've saved it all to your memory. You can keep chatting normally; I'll remember all of this.");
    if (inputRow) inputRow.style.display = 'none';
    if (_personalizerState) _personalizerState.active = false;
  }

  // Show a transient "…" bubble while the model composes its next question.
  function _showThinking() {
    const div = document.createElement('div');
    div.className = 'fg-pers-thinking';
    div.style.cssText = 'max-width:85%;padding:10px 13px;border-radius:12px;font-size:13px;align-self:flex-start;opacity:0.6;background:color-mix(in srgb,var(--fg) 6%,transparent);';
    div.textContent = '…';
    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
    return div;
  }

  // Model-driven: send the transcript so far, get the next adaptive question.
  // Falls back to the fixed question list if the endpoint is unavailable so the
  // interview still works with no model configured.
  async function _askNext() {
    const state = _personalizerState;
    if (!state || !state.active) return;

    if (state.useFallback) {
      if (state.qIdx >= INTERVIEW_QUESTIONS.length) { _finishInterview(); return; }
      const q = INTERVIEW_QUESTIONS[state.qIdx++];
      state.lastQuestion = q;
      _addMsg('assistant', q);
      return;
    }

    const thinking = _showThinking();
    let data = null;
    try {
      const res = await fetch(`${API}/api/memory/interview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ transcript: state.transcript }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      data = await res.json();
    } catch (_) {
      // First failure → fall back to the static script for the rest of the run.
      thinking.remove();
      state.useFallback = true;
      return _askNext();
    }
    thinking.remove();

    if (data?.done || !data?.question) { _finishInterview(); return; }
    const q = data.question;
    state.lastQuestion = q;
    state.transcript.push({ role: 'assistant', content: q });
    _addMsg('assistant', q);
  }

  async function _handleAnswer() {
    const answer = answerEl.value.trim();
    if (!answer || !_personalizerState?.active) return;
    const state = _personalizerState;
    answerEl.value = '';
    _addMsg('user', answer);
    state.transcript.push({ role: 'user', content: answer });
    const qText = state.lastQuestion || '';
    await _saveMemory(qText ? `${qText} — ${answer}` : answer, 'identity');
    await _askNext();
  }

  sendBtn.addEventListener('click', _handleAnswer);
  answerEl.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); _handleAnswer(); } });

  startBtn.addEventListener('click', () => {
    chatEl.innerHTML = '';
    _personalizerState = { qIdx: 0, active: true, memories: [], transcript: [], lastQuestion: '', useFallback: false };
    inputRow.style.display = 'flex';
    _addMsg('assistant', "Hi! I'm going to ask you a few questions to learn about you. Each answer is saved as a memory so I can personalise every conversation.\n\nLet's start:");
    setTimeout(() => _askNext(), 400);
  });

  mineBtn.addEventListener('click', async () => {
    mineBtn.disabled = true;
    mineBtn.textContent = 'Mining…';
    _addMsg('assistant', 'Mining your recent chats for facts about you…');
    try {
      const sessRes = await fetch(`${API}/api/sessions`, { credentials: 'same-origin' }).then(r=>r.json());
      const sessions = Array.isArray(sessRes) ? sessRes : (sessRes.sessions || []);
      const sids = sessions.slice(0, 10).map(s => s.id).filter(Boolean);
      // /api/memory/extract works per-session: it takes a `session` form field,
      // returns {suggestions: [...]}, and does NOT persist. So extract per
      // session, then save each suggestion via _saveMemory. (The old code posted
      // free text as JSON and read .memories — it 422'd and saved nothing.)
      const found = [];
      for (const sid of sids) {
        try {
          const fd = new FormData(); fd.append('session', sid);
          const res = await fetch(`${API}/api/memory/extract`, { method: 'POST', credentials: 'same-origin', body: fd });
          if (!res.ok) continue;
          const data = await res.json();
          for (const s of (data.suggestions || [])) {
            if (s && !found.includes(s)) { found.push(s); await _saveMemory(s, 'fact', 'personalizer-mining'); }
          }
        } catch (_) {}
      }
      if (found.length) {
        _addMsg('assistant', `Found and saved ${found.length} fact(s) from your recent chats:\n\n${found.slice(0,10).map(f=>`• ${f}`).join('\n')}`);
      } else {
        _addMsg('assistant', 'No new facts found in your recent chats. Try the interview instead!');
      }
    } catch (e) {
      _addMsg('assistant', `Mining failed: ${e}`);
    }
    mineBtn.disabled = false;
    mineBtn.textContent = 'Mine Past Chats';
  });

  fetch(`${API}/api/memory?category=identity&limit=5`, { credentials: 'same-origin' })
    .then(r => r.json()).then(data => {
      const mems = data.memories || data || [];
      if (Array.isArray(mems) && mems.length) {
        _addMsg('assistant', `Welcome back! I already have ${mems.length} thing(s) in memory about you. Click "Start Interview" to add more, or "Mine Past Chats" to extract more from your history.`);
      } else {
        _addMsg('assistant', 'Welcome! Click "Start Interview" to teach me about yourself, or "Mine Past Chats" to let me discover facts from your chat history automatically.');
      }
    }).catch(() => {
      _addMsg('assistant', 'Welcome! Click "Start Interview" to teach me about yourself.');
    });
}

export function openPersonalizer() {
  _openModal('forge-personalizer-modal', 'rail-forge-personalizer', _buildPersonalizer);
}

// ── SELF-TRAIN MODAL ──────────────────────────────────────────────────────────

async function _fetchSelfTrainStats() {
  const [sessRes, memRes, adaptRes] = await Promise.allSettled([
    fetch(`${API}/api/sessions`, { credentials: 'same-origin' }).then(r=>r.json()),
    fetch(`${API}/api/memory?limit=1`, { credentials: 'same-origin' }).then(r=>r.json()),
    fetch(`${API}/api/training/adapters`, { credentials: 'same-origin' }).then(r=>r.json()),
  ]);

  const sessions  = sessRes.status === 'fulfilled' ? (Array.isArray(sessRes.value) ? sessRes.value : (sessRes.value?.sessions || [])) : [];
  const memData   = memRes.status === 'fulfilled' ? memRes.value : {};
  const adaptData = adaptRes.status === 'fulfilled' ? adaptRes.value : {};

  const lastTrainKey = 'forge-last-train-ts';
  let lastTrain = 0;
  try { lastTrain = parseInt(localStorage.getItem(lastTrainKey) || '0'); } catch (_) {}

  const newSessions = sessions.filter(s => {
    const ts = new Date(s.last_message_at || s.updated_at || 0).getTime();
    return ts > lastTrain && (s.message_count || 0) > 2;
  }).length;

  const totalMem = memData.total || (Array.isArray(memData) ? memData.length : 0);
  const adapters = adaptData.adapters || [];
  const ready = newSessions >= 5;

  return { newSessions, totalMem, adapters, ready, lastTrain };
}

function _buildSelfTrain(body) {
  body.innerHTML = `
    <div style="display:flex;flex-direction:column;gap:12px;">
      <div class="admin-card" id="fg-st-stats-card">
        <h2><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-1px;margin-right:5px;opacity:0.7"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>Training Readiness</h2>
        <div id="fg-st-loading" style="font-size:12px;opacity:0.5;padding:8px 0;">Loading stats…</div>
        <div id="fg-st-rows" style="display:none;"></div>
      </div>
      <div class="admin-card" id="fg-st-action-card" style="display:none;">
        <h2 style="color:var(--green,#50fa7b)"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-1px;margin-right:5px;"><polyline points="20 6 9 17 4 12"/></svg>Ready to Train</h2>
        <p class="memory-desc" style="margin-bottom:10px;">You've accumulated enough new conversations. Start a training run to improve the model on your recent chats.</p>
        <div style="display:flex;gap:8px;">
          <button class="cookbook-btn" id="fg-st-train-btn" style="flex:1;background:color-mix(in srgb,var(--green,#50fa7b) 15%,transparent);border-color:color-mix(in srgb,var(--green,#50fa7b) 35%,transparent);color:var(--green,#50fa7b);">Start Training</button>
          <button class="cookbook-btn" id="fg-st-dismiss-btn" style="opacity:0.6;padding:6px 12px;">Later</button>
        </div>
      </div>
      <div class="admin-card">
        <h2><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-1px;margin-right:5px;opacity:0.7"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>Trained Adapters</h2>
        <div id="fg-st-adapters" style="font-size:12px;opacity:0.5;padding:4px 0;">—</div>
      </div>
      <div style="font-size:11px;opacity:0.4;text-align:right;">Auto-refresh every 30 s</div>
    </div>`;

  const loadingEl  = body.querySelector('#fg-st-loading');
  const rowsEl     = body.querySelector('#fg-st-rows');
  const actionCard = body.querySelector('#fg-st-action-card');
  const adaptersEl = body.querySelector('#fg-st-adapters');
  const trainBtn   = body.querySelector('#fg-st-train-btn');
  const dismissBtn = body.querySelector('#fg-st-dismiss-btn');

  async function _refresh() {
    try {
      const s = await _fetchSelfTrainStats();
      if (loadingEl) loadingEl.style.display = 'none';
      if (rowsEl) {
        rowsEl.style.display = '';
        const lastStr = s.lastTrain ? new Date(s.lastTrain).toLocaleDateString() : 'Never';
        rowsEl.innerHTML = `
          <div style="display:flex;justify-content:space-between;padding:5px 0;font-size:13px;border-bottom:1px solid color-mix(in srgb,var(--border) 50%,transparent)"><span style="opacity:0.65">New chats since last train</span><b>${s.newSessions}</b></div>
          <div style="display:flex;justify-content:space-between;padding:5px 0;font-size:13px;border-bottom:1px solid color-mix(in srgb,var(--border) 50%,transparent)"><span style="opacity:0.65">Total memories</span><b>${s.totalMem}</b></div>
          <div style="display:flex;justify-content:space-between;padding:5px 0;font-size:13px"><span style="opacity:0.65">Last training run</span><b>${lastStr}</b></div>`;
      }
      if (actionCard) actionCard.style.display = s.ready ? '' : 'none';
      if (adaptersEl) {
        adaptersEl.textContent = s.adapters.length
          ? s.adapters.map(a => typeof a === 'string' ? a : (a.name || JSON.stringify(a))).join(', ')
          : 'No adapters trained yet.';
      }
    } catch (e) {
      if (loadingEl) loadingEl.textContent = `Error: ${e}`;
    }
  }

  _refresh();
  const _iv = setInterval(_refresh, 30000);

  trainBtn?.addEventListener('click', () => {
    clearInterval(_iv);
    _closeModal('forge-selftrain-modal');
    openTrain();
  });

  dismissBtn?.addEventListener('click', () => {
    try { localStorage.setItem('forge-last-train-ts', String(Date.now())); } catch (_) {}
    actionCard.style.display = 'none';
  });

  body._cleanupInterval = _iv;
}

function _cleanupSelfTrain() {
  const body = document.getElementById('forge-selftrain-modal-body');
  if (body?._cleanupInterval) { clearInterval(body._cleanupInterval); body._cleanupInterval = null; }
}

export function openSelfTrain() {
  const id = 'forge-selftrain-modal';
  const modal = document.getElementById(id);
  if (!modal) return;
  if (Modals.isMinimized(id)) { Modals.restore(id); return; }
  if (!modal.classList.contains('hidden')) return;
  modal.classList.remove('hidden');
  Modals.register(id, {
    railBtnId: 'rail-forge-selftrain',
    closeFn: () => { _cleanupSelfTrain(); _closeModal(id); },
    restoreFn: () => {},
  });
  const body = document.getElementById(`${id}-body`);
  if (body && body.dataset.built !== '1') {
    _buildSelfTrain(body);
    body.dataset.built = '1';
  }
}

// ── DOM INIT — create modals and wire nav buttons ─────────────────────────────

function _svgIcon(d, extra='') {
  return `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:6px;${extra}">${d}</svg>`;
}

const FORGE_MODALS = [
  {
    id: 'forge-train-modal',
    railId: 'rail-forge-train',
    title: 'Train',
    icon: _svgIcon('<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>'),
    open: () => openTrain(),
  },
  {
    id: 'forge-qlora-modal',
    railId: 'rail-forge-qlora',
    title: 'QLoRA',
    icon: _svgIcon('<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="9"/><line x1="3" y1="12" x2="9" y2="12"/><line x1="15" y1="12" x2="21" y2="12"/>'),
    open: () => openQLoRA(),
  },
  {
    id: 'forge-rlloop-modal',
    railId: 'rail-forge-rlloop',
    title: 'RL Loop',
    icon: _svgIcon('<polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>'),
    open: () => openRLLoop(),
  },
  {
    id: 'forge-ingest-modal',
    railId: 'rail-forge-ingest',
    title: 'Ingest',
    icon: _svgIcon('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>'),
    open: () => openIngest(),
  },
  {
    id: 'forge-personalizer-modal',
    railId: 'rail-forge-personalizer',
    title: 'Personalizer',
    icon: _svgIcon('<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>'),
    open: () => openPersonalizer(),
  },
  {
    id: 'forge-selftrain-modal',
    railId: 'rail-forge-selftrain',
    title: 'Self-Train',
    icon: _svgIcon('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'),
    open: () => openSelfTrain(),
  },
];

function _init() {
  for (const m of FORGE_MODALS) {
    if (!document.getElementById(m.id)) {
      _makeModal(m.id, m.title, m.icon);
    }
    const toggle = () => {
      if (Modals.isMinimized(m.id)) { Modals.restore(m.id); return; }
      const modal = document.getElementById(m.id);
      if (modal && !modal.classList.contains('hidden')) { Modals.minimize(m.id); return; }
      m.open();
    };
    // Wire both entry points: the icon rail (only visible with the sidebar
    // collapsed) and the always-visible "Forge" section in the sidebar.
    const sbId = m.railId.replace(/^rail-forge-/, 'forge-sb-');
    for (const btnId of [m.railId, sbId]) {
      const btn = document.getElementById(btnId);
      if (btn) btn.addEventListener('click', toggle);
    }
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _init);
} else {
  _init();
}
