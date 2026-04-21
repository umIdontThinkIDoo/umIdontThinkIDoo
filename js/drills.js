import { audio } from './audio.js';
import { midiToFreq, midiToNoteName, noteNameToMidi } from './notes.js';
import { addSession } from './storage.js';
import { toast } from './app.js';
import { get, update } from './state.js';

/**
 * Drill definitions — list of MIDI offsets from a tonic.
 * Tonic is chosen per-user (default C4) so singers pick what's comfortable.
 */
const DRILLS = [
  { id: 'major-scale', name: 'Major scale (up & down)', offsets: [0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0] },
  { id: 'minor-scale', name: 'Natural minor scale', offsets: [0, 2, 3, 5, 7, 8, 10, 12, 10, 8, 7, 5, 3, 2, 0] },
  { id: 'arpeggio', name: 'Major arpeggio (1-3-5-8)', offsets: [0, 4, 7, 12, 7, 4, 0] },
  { id: 'fifths', name: 'Perfect fifths', offsets: [0, 7, 0, 7, 0] },
  { id: 'octave', name: 'Octave jumps', offsets: [0, 12, 0, 12, 0] },
  { id: 'siren-up', name: 'Siren — up a fifth', offsets: [0, 2, 4, 5, 7] },
  { id: 'rnb-run', name: 'R&B run (1-3-5-6-5-3-1)', offsets: [0, 4, 7, 9, 7, 4, 0] }
];

const DEFAULT_TONIC = 'C4';
const NOTE_SEC = 1.4; // time per note (play + sing)
const SCORE_WINDOW_SEC = 0.8; // time after tone to evaluate pitch

export function renderDrills(root) {
  const saved = get('drills') || {};
  let state = {
    drill: DRILLS.find((d) => d.id === saved.drillId) || DRILLS[0],
    tonic: saved.tonic || DEFAULT_TONIC,
    running: false,
    idx: 0,
    scores: [],
    recordedDetections: []
  };
  let unsubscribe = null;
  let drillTimeout = null;
  let evalTimeout = null;

  const redraw = () => {
    const noteMidi = noteNameToMidi(state.tonic);
    const current = state.running ? state.drill.offsets[state.idx] : null;
    const currentNote = current != null ? midiToNoteName(noteMidi + current) : '—';
    const progress = state.running ? ((state.idx) / state.drill.offsets.length) * 100 : 0;
    const avg = state.scores.length ? Math.round(state.scores.reduce((a, b) => a + b, 0) / state.scores.length) : null;

    root.innerHTML = `
      <div class="card">
        <h2>Drill</h2>
        <label class="hint">Pick a drill</label>
        <select id="drill-select">
          ${DRILLS.map((d) => `<option value="${d.id}" ${d.id === state.drill.id ? 'selected' : ''}>${d.name}</option>`).join('')}
        </select>
        <div style="height:8px"></div>
        <label class="hint">Starting note (tonic)</label>
        <select id="tonic-select">
          ${tonicOptions().map((n) => `<option value="${n}" ${n === state.tonic ? 'selected' : ''}>${n}</option>`).join('')}
        </select>
      </div>

      <div class="card">
        <div class="drill-target" id="drill-target">${currentNote}</div>
        <div class="drill-progress"><div style="width:${progress}%"></div></div>
        <div class="row between">
          <div class="hint">${state.running ? `Note ${state.idx + 1} / ${state.drill.offsets.length}` : 'Press Start to begin'}</div>
          <div class="hint" id="live-feedback">—</div>
        </div>
      </div>

      <div class="card">
        <div class="row" style="gap:8px;">
          <button class="btn primary full" id="start-drill">${state.running ? 'Stop' : 'Start drill'}</button>
        </div>
        <div class="hint" style="margin-top:8px;">The app plays the target note, then listens for ~0.8s while you match it. Use headphones to avoid the mic picking up the tone.</div>
      </div>

      ${state.scores.length ? `
        <div class="card">
          <h2>This run</h2>
          <div class="kv"><span>Average accuracy</span><span>${avg}%</span></div>
          <div class="kv"><span>Notes scored</span><span>${state.scores.length} / ${state.drill.offsets.length}</span></div>
        </div>` : ''}
    `;

    root.querySelector('#drill-select').addEventListener('change', (e) => {
      state.drill = DRILLS.find((d) => d.id === e.target.value);
      update('drills', { drillId: state.drill.id });
    });
    root.querySelector('#tonic-select').addEventListener('change', (e) => {
      state.tonic = e.target.value;
      update('drills', { tonic: state.tonic });
    });
    root.querySelector('#start-drill').addEventListener('click', () => {
      state.running ? stopDrill() : startDrill();
    });
  };

  async function startDrill() {
    if (!audio.isRunning) {
      try { await audio.start(); }
      catch (e) { alert(e.message || e); return; }
    }
    state.running = true;
    state.idx = 0;
    state.scores = [];
    state.recordedDetections = [];
    unsubscribe = audio.subscribe(onFrame);
    redraw();
    nextNote();
  }

  function stopDrill() {
    state.running = false;
    if (drillTimeout) clearTimeout(drillTimeout);
    if (evalTimeout) clearTimeout(evalTimeout);
    if (unsubscribe) { unsubscribe(); unsubscribe = null; }
    if (state.scores.length) {
      const avg = state.scores.reduce((a, b) => a + b, 0) / state.scores.length;
      addSession({
        kind: 'drill',
        drillId: state.drill.id,
        tonic: state.tonic,
        avgAccuracy: avg,
        notes: state.recordedDetections
      }).catch(() => {});
      toast(`Drill saved — ${Math.round(avg)}% accuracy`);
    }
    redraw();
  }

  let currentTargetFreq = null;
  let currentDetections = [];
  let scoring = false;
  function nextNote() {
    if (!state.running) return;
    if (state.idx >= state.drill.offsets.length) { stopDrill(); return; }

    const tonicMidi = noteNameToMidi(state.tonic);
    const targetMidi = tonicMidi + state.drill.offsets[state.idx];
    currentTargetFreq = midiToFreq(targetMidi);
    currentDetections = [];
    scoring = false;
    // Play the target tone
    const toneMs = 600;
    audio.playTone(currentTargetFreq, toneMs);
    redraw();
    // Only begin collecting detections AFTER the tone finishes to avoid the mic biasing toward its own output
    setTimeout(() => { scoring = state.running; }, toneMs + 100);
    // After tone finishes, open scoring window
    evalTimeout = setTimeout(() => {
      scoring = false;
      // Score: compute median cents error from detections
      if (currentDetections.length) {
        const centsErrors = currentDetections.map((f) => 1200 * Math.log2(f / currentTargetFreq));
        centsErrors.sort((a, b) => a - b);
        const medianCents = Math.abs(centsErrors[Math.floor(centsErrors.length / 2)]);
        // score: 100 at 0 cents, 0 at 50+ cents
        const score = Math.max(0, Math.min(100, Math.round(100 - (medianCents * 2))));
        state.scores.push(score);
        state.recordedDetections.push({ targetMidi, medianCents, score });
      } else {
        state.scores.push(0);
        state.recordedDetections.push({ targetMidi, medianCents: null, score: 0 });
      }
      state.idx++;
      drillTimeout = setTimeout(nextNote, 200);
    }, (toneMs + 100 + SCORE_WINDOW_SEC * 1000));
  }

  function onFrame(frame) {
    if (!state.running || currentTargetFreq == null) return;
    const el = root.querySelector('#live-feedback');
    if (!el) return;
    if (frame.freq > 0) {
      const cents = 1200 * Math.log2(frame.freq / currentTargetFreq);
      const absC = Math.abs(cents);
      el.textContent = `${cents > 0 ? '+' : ''}${cents.toFixed(0)} cents`;
      el.style.color = absC < 15 ? 'var(--ok)' : absC < 40 ? 'var(--warn)' : 'var(--bad)';
      if (scoring) currentDetections.push(frame.freq);
    } else {
      el.textContent = '—';
      el.style.color = '';
    }
  }

  redraw();

  return () => {
    stopDrill();
  };
}

function tonicOptions() {
  const out = [];
  for (let oct = 2; oct <= 5; oct++) {
    for (const n of ['C', 'D', 'E', 'F', 'G', 'A', 'B']) {
      out.push(`${n}${oct}`);
    }
  }
  return out;
}
