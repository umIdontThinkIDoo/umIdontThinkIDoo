import { audio } from './audio.js';
import { addRecording } from './storage.js';
import { midiToNoteName, noteNameToMidi, midiToFreq, transposeForRange } from './notes.js';
import { toast } from './app.js';
import { get, update } from './state.js';
import { settings } from './storage.js';
import { analyzeFile, exportMelodyJson } from './analyzer.js';

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
  const saved = get('karaoke') || {};
  const state = {
    audioUrl: null,
    audioBlob: null,
    audioName: saved.audioName || null,
    melody: saved.melody || DEMO_MELODY,
    melodyName: saved.melodyName || 'Demo (C major arpeggio)',
    playing: false,
    detections: [],
    startTime: 0,
    score: saved.lastScore != null ? saved.lastScore : null,
    analyzing: false,
    analyzeProgress: 0
  };
  let audioEl = null;
  let unsubscribe = null;

  root.innerHTML = `
    <div class="card">
      <h2>Load a song</h2>
      <p>Upload a backing track (MP3/M4A/WAV). You can also run the track through the <strong>Analyzer</strong> below to auto-generate a melody JSON for scoring.</p>
      <div class="col">
        <label class="btn ghost full">
          <input type="file" id="audio-file" accept="audio/*" hidden>
          <span id="audio-file-label">${state.audioName ? state.audioName : 'Choose backing track…'}</span>
        </label>
        <label class="btn ghost full">
          <input type="file" id="melody-file" accept="application/json,.json" hidden>
          <span id="melody-file-label">${state.melodyName ? state.melodyName : 'Melody JSON (optional)'}</span>
        </label>
      </div>
      <div id="spotify-link-card" class="hint" style="margin-top:10px;">
        Want to study a song on Spotify? Open it, note the vocal line, then upload an instrumental or the full track here — the analyzer will extract the melody contour.
      </div>
    </div>

    <div class="card" id="pipeline-card">
      <h2>Melody pipeline <span class="badge" id="pipeline-status">idle</span></h2>
      <p>Runs YIN pitch detection over the uploaded audio and segments the result into notes. For stereo tracks the pipeline can first isolate the center channel (typical placement for lead vocals) to reject stereo-spread instruments.</p>
      <label class="row" style="gap:8px; margin-bottom:8px;">
        <input type="checkbox" id="isolate-toggle" checked>
        <span>Isolate vocals before pitch detection (stereo only)</span>
      </label>
      <button class="btn primary full" id="run-analyzer" disabled>Analyze uploaded track → melody JSON</button>
      <div class="drill-progress" style="margin-top:10px;"><div id="analyze-bar" style="width:0%"></div></div>
      <div class="row between" style="margin-top:8px;">
        <span class="hint" id="analyze-detail">Upload a track above first.</span>
        <button class="btn ghost" id="download-melody" disabled>Download JSON</button>
      </div>
    </div>

    <div class="card">
      <canvas id="karaoke-canvas" class="pitch-track" width="900" height="180"></canvas>
      <audio id="karaoke-audio" preload="auto" controls style="width:100%; margin-top:8px;"></audio>
    </div>

    <div class="card">
      <div class="grid-2">
        <button class="btn primary" id="k-start">▶ Start &amp; record</button>
        <button class="btn" id="k-transpose">↕ Transpose to my range</button>
      </div>
      <div class="hint" id="transpose-hint" style="margin-top:8px;">Shift the melody so it sits in your detected vocal range. Sing low/high extremes in the Tuner to set your range first.</div>
      <div id="score-area" style="margin-top:12px; ${state.score == null ? 'display:none;' : ''}">
        <div class="score-ring" style="--val:${state.score || 0}"><div id="score-val">${state.score || 0}%</div></div>
        <p class="hint" id="score-label" style="text-align:center;">${state.score != null ? grade(state.score) : ''}</p>
      </div>
    </div>

    <div class="card">
      <h2>Custom melody format</h2>
      <p class="hint">JSON array of notes: <code>[{"t": 0, "note": "G4", "dur": 0.5}]</code>. Times/durations in seconds relative to the track start. The analyzer above produces this format automatically.</p>
    </div>
  `;

  const els = {
    audioFile: root.querySelector('#audio-file'),
    audioFileLabel: root.querySelector('#audio-file-label'),
    melodyFile: root.querySelector('#melody-file'),
    melodyFileLabel: root.querySelector('#melody-file-label'),
    runAnalyzer: root.querySelector('#run-analyzer'),
    downloadMelody: root.querySelector('#download-melody'),
    pipelineStatus: root.querySelector('#pipeline-status'),
    analyzeBar: root.querySelector('#analyze-bar'),
    analyzeDetail: root.querySelector('#analyze-detail'),
    isolateToggle: root.querySelector('#isolate-toggle'),
    canvas: root.querySelector('#karaoke-canvas'),
    audio: root.querySelector('#karaoke-audio'),
    kStart: root.querySelector('#k-start'),
    kTranspose: root.querySelector('#k-transpose'),
    transposeHint: root.querySelector('#transpose-hint'),
    scoreArea: root.querySelector('#score-area'),
    scoreVal: root.querySelector('#score-val'),
    scoreLabel: root.querySelector('#score-label')
  };
  audioEl = els.audio;

  const draw = () => {
    const canvas = els.canvas;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = '#15151d';
    ctx.fillRect(0, 0, w, h);
    const midiMin = 45, midiMax = 84;
    const melodyEnd = state.melody.length
      ? state.melody[state.melody.length - 1].t + state.melody[state.melody.length - 1].dur
      : 5;
    const detectionEnd = state.detections.length ? state.detections[state.detections.length - 1].t : 0;
    const totalT = Math.max(melodyEnd, detectionEnd, audioEl.duration || 5, 5);

    ctx.strokeStyle = '#2a2a38';
    ctx.lineWidth = 1;
    for (let m = midiMin; m <= midiMax; m += 12) {
      const y = h - ((m - midiMin) / (midiMax - midiMin)) * h;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    for (const n of state.melody) {
      const midi = noteNameToMidi(n.note);
      if (!isFinite(midi)) continue;
      const x = (n.t / totalT) * w;
      const wx = (n.dur / totalT) * w;
      const y = h - ((midi - midiMin) / (midiMax - midiMin)) * h;
      ctx.fillStyle = 'rgba(124, 92, 255, 0.45)';
      ctx.fillRect(x, y - 6, wx, 12);
    }

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

    if (audioEl && state.playing) {
      const t = audioEl.currentTime;
      const x = (t / totalT) * w;
      ctx.strokeStyle = '#ff5470';
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
  };
  draw();

  const setPipelineStatus = (label, tone) => {
    els.pipelineStatus.textContent = label;
    els.pipelineStatus.className = 'badge' + (tone ? ' ' + tone : '');
  };

  const updateRunBtn = () => {
    els.runAnalyzer.disabled = !state.audioBlob || state.analyzing;
    els.runAnalyzer.textContent = state.analyzing
      ? `Analyzing… ${Math.round(state.analyzeProgress * 100)}%`
      : 'Analyze uploaded track → melody JSON';
  };

  els.audioFile.addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
    state.audioUrl = URL.createObjectURL(f);
    state.audioName = f.name;
    state.audioBlob = f;
    audioEl.src = state.audioUrl;
    els.audioFileLabel.textContent = f.name;
    els.analyzeDetail.textContent = `Ready: ${f.name} (${formatBytes(f.size)})`;
    updateRunBtn();
    update('karaoke', { audioName: f.name });
  });

  els.melodyFile.addEventListener('change', async (e) => {
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
      els.melodyFileLabel.textContent = f.name;
      update('karaoke', { melody: parsed, melodyName: f.name });
      draw();
      toast(`Loaded ${parsed.length} notes`);
    } catch (err) { alert('Invalid melody JSON: ' + err.message); }
  });

  els.runAnalyzer.addEventListener('click', async () => {
    if (!state.audioBlob || state.analyzing) return;
    state.analyzing = true;
    state.analyzeProgress = 0;
    setPipelineStatus('decoding…', 'warn');
    els.analyzeBar.style.width = '0%';
    els.analyzeDetail.textContent = 'Decoding audio…';
    updateRunBtn();
    const started = performance.now();
    try {
      const result = await analyzeFile(state.audioBlob, {
        isolateVocals: els.isolateToggle.checked,
        onProgress: (p, stage) => {
          state.analyzeProgress = p;
          els.analyzeBar.style.width = Math.round(p * 100) + '%';
          const label = stage === 'separate' ? 'Isolating vocals' : 'Detecting pitch';
          els.analyzeDetail.textContent = `${label}… ${Math.round(p * 100)}%`;
          setPipelineStatus(`${stage === 'separate' ? 'separating' : 'analyzing'} ${Math.round(p * 100)}%`, 'warn');
          updateRunBtn();
        }
      });
      const elapsed = ((performance.now() - started) / 1000).toFixed(1);
      state.melody = result.melody;
      state.melodyName = (state.audioName || 'track') + ' — auto melody';
      els.melodyFileLabel.textContent = state.melodyName;
      const isoNote = result.isolatedVocals ? ' (center-channel isolated)' : '';
      els.analyzeDetail.textContent = `${result.melody.length} notes extracted in ${elapsed}s${isoNote}`;
      setPipelineStatus('done', 'ok');
      els.analyzeBar.style.width = '100%';
      els.downloadMelody.disabled = false;
      els.downloadMelody.dataset.json = exportMelodyJson(result.melody);
      update('karaoke', { melody: result.melody, melodyName: state.melodyName });
      update('pipeline', {
        lastFileName: state.audioName,
        lastDurationSec: result.duration,
        lastNoteCount: result.melody.length,
        lastRunAt: Date.now()
      });
      draw();
      toast(`Melody generated (${result.melody.length} notes)`);
    } catch (err) {
      setPipelineStatus('failed', 'bad');
      els.analyzeDetail.textContent = 'Error: ' + (err.message || err);
      toast('Analyzer failed');
    } finally {
      state.analyzing = false;
      state.analyzeProgress = 0;
      updateRunBtn();
    }
  });

  els.downloadMelody.addEventListener('click', () => {
    const json = els.downloadMelody.dataset.json;
    if (!json) return;
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = (state.audioName ? state.audioName.replace(/\.[^.]+$/, '') : 'melody') + '.melody.json';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  els.kStart.addEventListener('click', () => { state.playing ? stop() : start(); });

  els.kTranspose.addEventListener('click', () => {
    const low = settings.get('range.low', null);
    const high = settings.get('range.high', null);
    if (!low || !high) {
      toast('Set your range first: open Tuner, sing a low note then a high note');
      return;
    }
    if (!state.melody || !state.melody.length) {
      toast('No melody loaded');
      return;
    }
    const { melody: shifted, shift } = transposeForRange(state.melody, low.midi, high.midi);
    if (shift === 0) {
      toast('Already in your range');
      return;
    }
    state.melody = shifted;
    const baseName = (state.melodyName || 'melody').replace(/\s*\([-+]?\d+\s*semitones?\)\s*$/, '');
    state.melodyName = `${baseName} (${shift > 0 ? '+' : ''}${shift} semitones)`;
    els.melodyFileLabel.textContent = state.melodyName;
    update('karaoke', { melody: shifted, melodyName: state.melodyName });
    // Refresh downloadable JSON too
    els.downloadMelody.dataset.json = exportMelodyJson(shifted);
    els.downloadMelody.disabled = false;
    draw();
    const dir = shift > 0 ? 'up' : 'down';
    toast(`Transposed ${dir} ${Math.abs(shift)} semitone${Math.abs(shift) === 1 ? '' : 's'}`);
  });

  updateRunBtn();

  async function start() {
    if (!state.audioUrl) { alert('Choose a backing track first.'); return; }
    try {
      if (!audio.isRunning) await audio.start();
    } catch (e) { alert(e.message || e); return; }

    state.detections = [];
    state.score = null;
    els.scoreArea.style.display = 'none';
    state.playing = true;
    state.startTime = performance.now();

    audio.startRecording();
    unsubscribe = audio.subscribe((frame) => {
      const t = audioEl.currentTime;
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
    els.kStart.textContent = '■ Stop';
    els.kStart.classList.remove('primary');
    els.kStart.classList.add('danger');
    const loop = () => { if (state.playing) { draw(); requestAnimationFrame(loop); } };
    loop();
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
      const cents = Math.abs(1200 * Math.log2(d.freq / refFreq));
      const adj = Math.min(cents % 1200, 1200 - (cents % 1200));
      const noteScore = Math.max(0, 100 - (adj / 2));
      sum += noteScore;
      scored++;
    }
    state.score = scored ? Math.round(sum / scored) : 0;
    update('karaoke', { lastScore: state.score });

    els.scoreArea.style.display = '';
    els.scoreArea.querySelector('.score-ring').style.setProperty('--val', state.score);
    els.scoreVal.textContent = state.score + '%';
    els.scoreLabel.textContent = grade(state.score);

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
    els.kStart.textContent = '▶ Start & record';
    els.kStart.classList.add('primary');
    els.kStart.classList.remove('danger');
    draw();
  }

  function grade(s) {
    if (s >= 90) return 'Studio-ready. Great pitch control.';
    if (s >= 75) return 'Solid — you\'re landing most notes.';
    if (s >= 60) return 'Getting there — focus on sustained notes.';
    if (s >= 40) return 'Keep working the drills.';
    return 'Check your key — try transposing.';
  }

  function formatBytes(n) {
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    return (n / 1024 / 1024).toFixed(1) + ' MB';
  }

  return {};
}
