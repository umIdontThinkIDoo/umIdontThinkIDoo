/**
 * Central UI / session state with localStorage persistence + 5-minute autosave.
 * Big recordings live in IndexedDB (storage.js); this is just for UI state that
 * should survive page reloads and tab switches.
 */

const KEY = 'range:state:v2';
const AUTOSAVE_MS = 5 * 60 * 1000;

const defaults = () => ({
  activeTab: 'tuner',
  tuner: {},
  drills: { drillId: 'major-scale', tonic: 'C4' },
  record: {
    pitchTrack: [],   // last saved/in-progress pitch track
    durationSec: 0,
    isRecording: false
  },
  karaoke: {
    audioName: null,
    melody: null,
    melodyName: null,
    detections: [],
    lastScore: null
  },
  pipeline: {
    lastFileName: null,
    lastDurationSec: 0,
    lastNoteCount: 0,
    lastRunAt: 0
  },
  lastSavedAt: 0,
  lastRestoredAt: 0
});

let state = defaults();
const listeners = new Set();
let saveTimer = null;

export function loadState() {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      state = { ...defaults(), ...parsed };
      state.lastRestoredAt = Date.now();
    }
  } catch (_) {}
  // Start autosave interval
  setInterval(saveState, AUTOSAVE_MS);
  window.addEventListener('beforeunload', saveState);
  document.addEventListener('visibilitychange', () => { if (document.hidden) saveState(); });
  return state;
}

export function getState() { return state; }

export function get(key) { return state[key]; }

export function update(key, patch) {
  state[key] = { ...(state[key] || {}), ...patch };
  scheduleSave();
  emit(key);
}

export function set(key, value) {
  state[key] = value;
  scheduleSave();
  emit(key);
}

export function saveState() {
  try {
    state.lastSavedAt = Date.now();
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch (_) {}
}

function scheduleSave() {
  if (saveTimer) return;
  saveTimer = setTimeout(() => { saveTimer = null; saveState(); }, 1500);
}

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function emit(key) {
  listeners.forEach((fn) => { try { fn(key, state); } catch (_) {} });
}

export function timeSinceSave() {
  if (!state.lastSavedAt) return null;
  return Date.now() - state.lastSavedAt;
}
