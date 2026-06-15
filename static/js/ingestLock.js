// ingestLock.js — pause chat while a heavy ingest / KG backfill runs.
//
// While the library is being embedded or the knowledge graph backfilled, the
// GPU runs flat-out and chat would crawl (and fight for VRAM). So we surface a
// banner over the composer with live progress + an "Interrupt ASAP" button, and
// disable the input. The server enforces the same lock (423), so this is the UX
// layer; the banner just makes the state visible and gives the user the stop.
//
// Polls /api/ingest/gate: slow when idle, faster while a job is active.

(() => {
  'use strict';

  const IDLE_MS = 6000;
  const ACTIVE_MS = 2000;

  let timer = null;
  let active = false;          // is a job currently running?
  let savedPlaceholder = null; // composer placeholder to restore
  let stopping = false;        // interrupt clicked, awaiting wind-down

  function el(id) { return document.getElementById(id); }

  function banner() {
    let b = el('ingest-lock-banner');
    if (b) return b;
    const bar = document.querySelector('.chat-input-bar');
    if (!bar || !bar.parentNode) return null;
    b = document.createElement('div');
    b.id = 'ingest-lock-banner';
    b.style.cssText = [
      'display:none', 'margin:0 auto 8px', 'max-width:min(48rem,100%)',
      'padding:10px 14px', 'border-radius:12px',
      'background:rgba(120,90,255,0.12)', 'border:1px solid rgba(120,90,255,0.35)',
      'font-size:13px', 'line-height:1.4', 'color:var(--text,#ddd)',
      'display:flex', 'align-items:center', 'gap:12px', 'flex-wrap:wrap',
    ].join(';');
    b.innerHTML = `
      <div style="flex:1;min-width:160px">
        <div style="font-weight:600" id="ingest-lock-title">Ingesting…</div>
        <div style="opacity:0.8;font-size:12px;margin-top:2px" id="ingest-lock-sub">Chat is paused while the GPU works.</div>
        <div style="height:4px;border-radius:3px;background:rgba(255,255,255,0.12);margin-top:8px;overflow:hidden">
          <div id="ingest-lock-fill" style="height:100%;width:0%;background:linear-gradient(90deg,#7a5aff,#a98bff);transition:width .4s ease"></div>
        </div>
      </div>
      <button type="button" id="ingest-lock-stop"
        style="white-space:nowrap;padding:7px 12px;border-radius:9px;border:1px solid rgba(255,255,255,0.25);background:rgba(255,80,80,0.18);color:var(--text,#fff);font-size:12px;font-weight:600;cursor:pointer">
        Interrupt ASAP
      </button>`;
    bar.parentNode.insertBefore(b, bar);
    el('ingest-lock-stop').addEventListener('click', onInterrupt);
    return b;
  }

  async function onInterrupt() {
    const btn = el('ingest-lock-stop');
    if (btn) { btn.disabled = true; btn.textContent = 'Stopping…'; }
    stopping = true;
    try {
      await fetch('/api/ingest/stop', { method: 'POST', credentials: 'same-origin' });
    } catch {}
    // Poll quickly so the banner clears as soon as the worker winds down.
    schedule(ACTIVE_MS);
  }

  function lockComposer(lock) {
    const ta = el('message');
    if (!ta) return;
    if (lock && !active) {
      savedPlaceholder = ta.placeholder;
      ta.placeholder = 'Chat paused — ingesting your library…';
    } else if (!lock && savedPlaceholder !== null) {
      ta.placeholder = savedPlaceholder;
      savedPlaceholder = null;
    }
    ta.disabled = lock;
    ta.classList.toggle('ingest-locked', lock);
  }

  function render(snap) {
    const b = banner();
    if (!b) return;
    const title = el('ingest-lock-title');
    const sub = el('ingest-lock-sub');
    const fill = el('ingest-lock-fill');
    if (title) title.textContent = `${snap.label || 'Ingesting'} — ${snap.pct || 0}%`;
    if (fill) fill.style.width = `${snap.pct || 0}%`;
    if (sub) {
      const cur = snap.current ? ` · ${snap.current}` : '';
      sub.textContent = stopping
        ? `Finishing the current item, then stopping…${cur}`
        : `Chat is paused while the GPU works${cur}.`;
    }
    b.style.display = 'flex';
  }

  function onActive(snap) {
    render(snap);
    lockComposer(true);
    active = true;
  }

  function onIdle() {
    const b = el('ingest-lock-banner');
    if (b) b.style.display = 'none';
    lockComposer(false);
    active = false;
    stopping = false;
    const btn = el('ingest-lock-stop');
    if (btn) { btn.disabled = false; btn.textContent = 'Interrupt ASAP'; }
  }

  async function poll() {
    let snap = null;
    try {
      const r = await fetch('/api/ingest/gate', { credentials: 'same-origin' });
      if (r.ok) snap = await r.json();
    } catch {}
    if (snap && snap.active) onActive(snap);
    else onIdle();
    schedule(snap && snap.active ? ACTIVE_MS : IDLE_MS);
  }

  function schedule(ms) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(poll, ms);
  }

  // Kick off after the DOM/composer exist.
  function start() { poll(); }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
