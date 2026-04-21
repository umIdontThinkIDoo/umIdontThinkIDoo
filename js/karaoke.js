import { audio } from './audio.js';
import { addRecording } from './storage.js';
import { midiToNoteName, noteNameToMidi, midiToFreq } from './notes.js';
import { toast } from './app.js';

/**
 * Simple karaoke mode:
 *   - User uploads a backing track (MP3/M4A)
 *   - Optionally provides a reference melody as JSON: [{t: seconds, note: "G4", dur: 0.5}, ...]
 *   - Or uses the built-in demo melody
 *   - Plays backing track + mic listening, scores pitch accuracy against reference melody over time
 */

const DEMO_MELODY = [
  { t: 0.0, note: 'C4', dur: 0.5 },
  { t: 0.5, note: 'E4', dur: 0.5 },
  { t: 1.0, note: 'G4', dur: 0.5 },
  { t: 1.5, note: 'C5', dur: 1.0 },
  { t: 2.5, note: 'G4', dur: 0.5 },
  { t: 3.0, note: 'E4', dur: 0.5 },
  { t: 3.5, note: 'C4', dur: 1.0 }
];

export function renderKaraoke(root) {
  let state = {
    audioUrl: null,
    audioName: null,
    melody: DEMO_MELODY,
    melodyName: 'Demo (C major arpeggio)',
    playing: false,
    detections: [], // {t, freq, midi}
    startTime: 0,
    score: null
  };
  let audioEl = null;
  let unsubscribe = null;

  const draw = () => {
    const canvas = root.querySelector('#karaoke-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = '#15151d';
    ctx.fillRect(0, 0, w, h);
    const midiMin = 45, midiMax = 84;
    const totalT = Math.max(
      state.melody.length ? state.melody[state.melody.length - 1].t + state.melody[state.melody.length - 1].dur : 5,
      state.detections.length ? state.detections[state.detections.length - 1].t : 0,
      5
    );

    // Gridlines
    ctx.strokeStyle = '#2a2a38';
    ctx.lineWidth = 1;
    for (let m = midiMin; m <= midiMax; m += 12) {
      const y = h - ((m - midiMin) / (midiMax - midiMin)) * h;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Reference melody (as bars)
    for (const n of state.melody) {
      const midi = noteNameToMidi(n.note);
      if (!isFinite(midi)) continue;
      const x = (n.t / totalT) * w;
      const wx = (n.dur / totalT) * w;
      const y = h - ((midi - midiMin) / (midiMax - midiMin)) * h;
      ctx.fillStyle = 'rgba(124, 92, 255, 0.45)';
      ctx.fillRect(x, y - 6, wx, 12);
    }

    // Detected pitches
    ctx.strokeStyle = '#29d9c5';
    ctx.lineWidth = 2;
    ctx.beginPath();
    let moved = false;
    for (const d of state.detections) {
      if (!d.midi) { moved = false; continue; }
      const x = (d.t / totalT) * w;
      const y = h - ((d.midi - midiMin) / (midiMax - midiMin)) * h;
      if (!moved) { ctx.moveTo(x, y); moved = true; } else { ctx.lineTo(x, y); }
    }
    ctx.stroke();

    // Playhead
    if (audioEl && state.playing) {
      const t = audioEl.currentTime;
      const x = (t / totalT) * w;
      ctx.strokeStyle = '#ff5470';
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
  };

  const redraw = () => {
    root.innerHTML = `
      <div class="card">
        <h2>Karaoke test</h2>
        <p>Upload a backing track, then sing along. We compare your pitch to a reference melody and score accuracy.</p>
        <div class="col">
          <label class="btn ghost full">
            <input type="file" id="audio-file" accept="audio/*" hidden>
            <span>${state.audioName || 'Choose backing track…'}</span>
          </label>
          <label class="btn ghost full">
            <input type="file" id="melody-file" accept="application/json,.json" hidden>
            <span>${state.melodyName || 'Melody JSON (optional)'}</span>
          </label>
        </div>
      </div>

      <div class="card">
        <canvas id="karaoke-canvas" class="pitch-track" width="900" height="180"></canvas>
        <audio id="karaoke-audio" preload="auto" controls style="width:100%; margin-top:8px;"></audio>
      </div>

      <div class="card">
        <div class="row" style="gap:8px;">
          <button class="btn primary full" id="k-start">${state.playing ? '■ Stop' : '▶ Start & record'}</button>
        </div>
        ${state.score != null ? `
          <div style="margin-top:12px;">
            <div class="score-ring" style="--val:${state.score}"><div>${state.score}%</div></div>
            <p class="hint" style="text-align:center;">${grade(state.score)}</p>
          </div>` : ''}
      </div>

      <div class="card">
        <h2>Custom melody format</h2>
        <p class="hint">JSON array of notes: <code>[{"t": 0, "note": "G4", "dur": 0.5}]</code>. Times/durations in seconds relative to the track start. Get a reference melody from sheet music or by transcribing the vocal line.</p>
      </div>
    `;
    audioEl = root.querySelector('#karaoke-audio');
    if (state.audioUrl) audioEl.src = state.audioUrl;

    root.querySelector('#audio-file').addEventListener('change', (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
      state.audioUrl = URL.createObjectURL(f);
      state.audioName = f.name;
      state.audioBlob = f;
      redraw();
    });

    root.querySelector('#melody-file').addEventListener('change', async (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      try {
        const text = await f.text();
        const parsed = JSON.parse(text);
        if (!Array.isArray(parsed)) throw new Error('JSON must be an array of notes');
        for (const n of parsed) {
          if (typeof n.t !== 'number' || typeof n.note !== 'string' || typeof n.dur !== 'number') {
            throw new Error('Each note needs t (seconds), note (e.g. G4), dur (seconds).');
          }
        }
        state.melody = parsed;
        state.melodyName = f.name;
        redraw();
      } catch (err) { alert('Invalid melody JSON: ' + err.message); }
    });

    root.querySelector('#k-start').addEventListener('click', () => {
      state.playing ? stop() : start();
    });

    // Animation frame for playhead
    const loop = () => {
      if (!state.playing) return;
      draw();
      requestAnimationFrame(loop);
    };
    if (state.playing) loop();
    draw();
  };

  async function start() {
    if (!state.audioUrl) { alert('Choose a backing track first.'); return; }
    try {
      if (!audio.isRunning) await audio.start();
    } catch (e) { alert(e.message || e); return; }

    state.detections = [];
    state.score = null;
    state.playing = true;
    state.startTime = performance.now();

    // Start mic recording
    audio.startRecording();
    unsubscribe = audio.subscribe((frame) => {
      const t = audioEl ? audioEl.currentTime : (performance.now() - state.startTime) / 1000;
      state.detections.push({
        t,
        freq: frame.freq > 0 ? frame.freq : 0,
        midi: frame.note ? frame.note.midi : null
      });
      draw();
    });

    audioEl.currentTime = 0;
    try { await audioEl.play(); } catch (e) { toast('Tap Start again (iOS autoplay)'); }
    audioEl.onended = () => stop();
    redraw();
  }

  async function stop() {
    if (!state.playing) return;
    state.playing = false;
    if (audioEl) audioEl.pause();
    if (unsubscribe) { unsubscribe(); unsubscribe = null; }

    const melodyAtTime = (t) => {
      for (const n of state.melody) if (t >= n.t && t < n.t + n.dur) return noteNameToMidi(n.note);
      return null;
    };

    let scored = 0, sum = 0;
    for (const d of state.detections) {
      if (d.midi == null) continue;
      const target = melodyAtTime(d.t);
      if (target == null) continue;
      const refFreq = midiToFreq(target);
      const detFreq = d.freq;
      const cents = Math.abs(1200 * Math.log2(detFreq / refFreq));
      // Octave-insensitive grading: take cents modulo 1200 closest to 0
      const adj = Math.min(cents % 1200, 1200 - (cents % 1200));
      const noteScore = Math.max(0, 100 - (adj / 2));
      sum += noteScore;
      scored++;
    }
    state.score = scored ? Math.round(sum / scored) : 0;

    try {
      const rec = await audio.stopRecording();
      if (rec && rec.blob) {
        await addRecording({
          kind: 'karaoke',
          blob: rec.blob,
          mimeType: rec.mimeType,
          durationSec: rec.durationSec,
          score: state.score,
          songName: state.audioName,
          pitchTrack: state.detections
        });
        toast(`Karaoke scored ${state.score}% — saved`);
      }
    } catch (e) { console.error(e); }

    redraw();
  }

  function grade(s) {
    if (s >= 90) return 'Studio-ready. Great pitch control.';
    if (s >= 75) return 'Solid — you\'re landing most notes.';
    if (s >= 60) return 'Getting there — focus on sustained notes.';
    if (s >= 40) return 'Keep working the drills.';
    return 'Check your key — try transposing.';
  }

  redraw();

  return () => {
    if (state.playing) stop();
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  };
}
