import { audio } from './audio.js';
import { renderTuner } from './tuner.js';
import { renderDrills } from './drills.js';
import { renderRecord } from './record.js';
import { renderLibrary } from './library.js';
import { renderKaraoke } from './karaoke.js';

const VIEWS = {
  tuner:   { title: 'Tuner',   render: renderTuner },
  drills:  { title: 'Drills',  render: renderDrills },
  record:  { title: 'Record',  render: renderRecord },
  karaoke: { title: 'Karaoke', render: renderKaraoke },
  library: { title: 'Library', render: renderLibrary }
};

const viewEl = document.getElementById('view');
const titleEl = document.getElementById('view-title');
const tabs = document.querySelectorAll('.tab');
const micStatus = document.getElementById('mic-status');

let cleanup = null;

function go(name) {
  if (!VIEWS[name]) name = 'tuner';
  if (cleanup) { try { cleanup(); } catch (_) {} cleanup = null; }
  viewEl.innerHTML = '';
  titleEl.textContent = VIEWS[name].title;
  tabs.forEach((t) => t.classList.toggle('active', t.dataset.view === name));
  location.hash = name;
  cleanup = VIEWS[name].render(viewEl);
}

tabs.forEach((t) => {
  t.addEventListener('click', () => go(t.dataset.view));
});

window.addEventListener('hashchange', () => go((location.hash || '').replace('#', '')));

audio.onStatusChange = (on) => {
  micStatus.textContent = on ? 'mic: on' : 'mic: off';
  micStatus.classList.toggle('on', on);
};

// Initial view
go((location.hash || '').replace('#', '') || 'tuner');

// Toast helper (used across views)
const toastEl = document.getElementById('toast');
let toastTimer = null;
export function toast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove('show'), 1800);
}

// Service worker registration
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch(() => {});
  });
}
