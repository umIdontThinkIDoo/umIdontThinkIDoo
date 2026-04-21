import { audio } from './audio.js';
import { loadState, get, set, saveState, timeSinceSave } from './state.js';
import { renderTuner } from './tuner.js';
import { renderDrills } from './drills.js';
import { renderRecord } from './record.js';
import { renderLibrary } from './library.js';
import { renderKaraoke } from './karaoke.js';

loadState();

const VIEWS = [
  { id: 'tuner',   title: 'Tuner',    make: renderTuner },
  { id: 'drills',  title: 'Drills',   make: renderDrills },
  { id: 'record',  title: 'Record',   make: renderRecord },
  { id: 'karaoke', title: 'Karaoke',  make: renderKaraoke },
  { id: 'library', title: 'Library',  make: renderLibrary }
];

const viewHost = document.getElementById('view');
const titleEl = document.getElementById('view-title');
const tabs = document.querySelectorAll('.tab');
const micStatus = document.getElementById('mic-status');

// Mount each view ONCE into its own panel — they stay alive across tab switches,
// so form inputs, scroll positions, running state all persist.
const instances = {};
for (const v of VIEWS) {
  const panel = document.createElement('div');
  panel.className = 'view-panel';
  panel.dataset.view = v.id;
  panel.style.display = 'none';
  viewHost.appendChild(panel);
  try {
    const api = v.make(panel) || {};
    instances[v.id] = { panel, api, def: v };
  } catch (e) {
    console.error('Failed to mount view', v.id, e);
    instances[v.id] = { panel, api: {}, def: v };
  }
}

let activeTab = null;
function go(name) {
  if (!instances[name]) name = 'tuner';
  if (activeTab === name) return;
  // Deactivate current
  if (activeTab && instances[activeTab].api.onDeactivate) {
    try { instances[activeTab].api.onDeactivate(); } catch (_) {}
  }
  // Swap visibility
  Object.values(instances).forEach((i) => { i.panel.style.display = 'none'; });
  instances[name].panel.style.display = 'block';
  titleEl.textContent = instances[name].def.title;
  tabs.forEach((t) => t.classList.toggle('active', t.dataset.view === name));
  // Activate new
  if (instances[name].api.onActivate) {
    try { instances[name].api.onActivate(); } catch (_) {}
  }
  activeTab = name;
  set('activeTab', name);
  if (location.hash !== '#' + name) location.hash = name;
}

// Expose globally for in-view navigation
window.__rangeGo = go;

tabs.forEach((t) => t.addEventListener('click', () => go(t.dataset.view)));
window.addEventListener('hashchange', () => go((location.hash || '').replace('#', '')));

audio.onStatusChange = (on) => {
  micStatus.textContent = on ? 'mic: on' : 'mic: off';
  micStatus.classList.toggle('on', on);
};

// Toast helper
const toastEl = document.getElementById('toast');
let toastTimer = null;
export function toast(msg, ms = 1800) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), ms);
}

// Restore active tab from saved state or hash
const initialTab = (location.hash || '').replace('#', '') || get('activeTab') || 'tuner';
go(initialTab);

// Notify user if we restored a session
const savedGap = timeSinceSave();
if (savedGap != null && savedGap < 24 * 60 * 60 * 1000) {
  const mins = Math.max(1, Math.round(savedGap / 60000));
  setTimeout(() => toast(`Session restored — last saved ${mins} min ago`), 700);
}

// Autosave indicator
const saveIndicator = document.getElementById('save-indicator');
function tickSaveIndicator() {
  const ms = timeSinceSave();
  if (saveIndicator) {
    saveIndicator.textContent = ms == null ? 'not saved' : `saved ${formatAgo(ms)}`;
  }
}
function formatAgo(ms) {
  const s = Math.round(ms / 1000);
  if (s < 60) return 'just now';
  const m = Math.round(s / 60);
  if (m < 60) return m + 'm ago';
  const h = Math.round(m / 60);
  return h + 'h ago';
}
setInterval(tickSaveIndicator, 15000);
tickSaveIndicator();
// Force first save so the indicator is meaningful
saveState();
tickSaveIndicator();

// Service worker
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  });
}
