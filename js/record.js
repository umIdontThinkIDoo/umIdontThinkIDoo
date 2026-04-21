import { audio } from './audio.js';
import { addRecording } from './storage.js';
import { toast } from './app.js';

export function renderRecord(root) {
  root.innerHTML = `
    <div class="card">
      <h2>Isolated voice recording</h2>
      <p>Mic is captured raw (no echo cancel, no noise suppression, no auto-gain) for accurate pitch tracking.</p>
      <canvas class="pitch-track" id="pitch-track" width="800" height="160"></canvas>
      <div class="row between" style="margin-top:8px;">
        <div class="hint" id="rec-time">0.0s</div>
        <div class="hint" id="rec-pitch">—</div>
      </div>
    </div>

    <div class="card">
      <div class="row" style="gap:8px;">
        <button class="btn primary full" id="rec-toggle">● Start recording</button>
      </div>
      <div class="hint" style="margin-top:8px;">Recordings save to this device only (IndexedDB). iOS Safari saves as .mp4; other browsers as .webm.</div>
    </div>
  `;

  const canvas = root.querySelector('#pitch-track');
  const ctx2d = canvas.getContext('2d');
  const timeEl = root.querySelector('#rec-time');
  const pitchEl = root.querySelector('#rec-pitch');
  const btn = root.querySelector('#rec-toggle');

  let pitchPoints = []; // {t, freq, note}
  let startTime = 0;
  let recording = false;
  let unsubscribe = null;

  const draw = () => {
    const w = canvas.width, h = canvas.height;
    ctx2d.fillStyle = '#15151d';
    ctx2d.fillRect(0, 0, w, h);

    // Horizontal gridlines every octave (C2=36 → C6=84)
    ctx2d.strokeStyle = '#2a2a38';
    ctx2d.lineWidth = 1;
    const midiMin = 36, midiMax = 84;
    for (let m = midiMin; m <= midiMax; m += 12) {
      const y = h - ((m - midiMin) / (midiMax - midiMin)) * h;
      ctx2d.beginPath();
      ctx2d.moveTo(0, y); ctx2d.lineTo(w, y);
      ctx2d.stroke();
    }

    if (!pitchPoints.length) return;
    const totalT = Math.max(3, pitchPoints[pitchPoints.length - 1].t);
    ctx2d.strokeStyle = '#29d9c5';
    ctx2d.lineWidth = 2;
    ctx2d.beginPath();
    let moved = false;
    for (const p of pitchPoints) {
      if (!p.midi) { moved = false; continue; }
      const x = (p.t / totalT) * w;
      const y = h - ((p.midi - midiMin) / (midiMax - midiMin)) * h;
      if (!moved) { ctx2d.moveTo(x, y); moved = true; } else { ctx2d.lineTo(x, y); }
    }
    ctx2d.stroke();
  };
  draw();

  const updateBtn = () => {
    btn.textContent = recording ? '■ Stop & save' : '● Start recording';
    btn.classList.toggle('danger', recording);
    btn.classList.toggle('primary', !recording);
  };
  updateBtn();

  btn.addEventListener('click', async () => {
    if (!recording) {
      try {
        if (!audio.isRunning) await audio.start();
        audio.startRecording();
        pitchPoints = [];
        startTime = performance.now();
        recording = true;
        unsubscribe = audio.subscribe((frame) => {
          const t = (performance.now() - startTime) / 1000;
          pitchPoints.push({
            t,
            freq: frame.freq > 0 ? frame.freq : 0,
            midi: frame.note ? frame.note.midi : null
          });
          timeEl.textContent = `${t.toFixed(1)}s`;
          pitchEl.textContent = frame.note ? `${frame.note.name}  ${frame.note.cents > 0 ? '+' : ''}${frame.note.cents}¢` : '—';
          draw();
        });
        updateBtn();
      } catch (e) { alert(e.message || e); }
    } else {
      if (unsubscribe) { unsubscribe(); unsubscribe = null; }
      recording = false;
      updateBtn();
      try {
        const result = await audio.stopRecording();
        if (result && result.blob) {
          const durationSec = pitchPoints.length ? pitchPoints[pitchPoints.length - 1].t : result.durationSec;
          const validPitches = pitchPoints.filter((p) => p.midi != null);
          const lowMidi = validPitches.length ? Math.min(...validPitches.map((p) => p.midi)) : null;
          const highMidi = validPitches.length ? Math.max(...validPitches.map((p) => p.midi)) : null;
          await addRecording({
            kind: 'freeform',
            blob: result.blob,
            mimeType: result.mimeType,
            durationSec,
            pitchTrack: pitchPoints,
            lowMidi, highMidi
          });
          toast('Saved to Library');
        }
      } catch (e) { alert(e.message || e); }
    }
  });

  return () => {
    if (unsubscribe) unsubscribe();
    if (audio.isRecording) audio.stopRecording().catch(() => {});
  };
}
