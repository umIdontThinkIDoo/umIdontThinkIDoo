/**
 * Offline audio → melody JSON pipeline.
 *
 * Takes an audio file (mp3/m4a/wav), runs YIN over sliding windows, segments
 * consecutive same-note frames into notes, and produces melody JSON compatible
 * with the karaoke grader.
 *
 * Works best on: isolated vocals, acapella, solo instrument. Polyphonic music
 * (full songs with bass + drums + vocal) will produce approximate results —
 * the detector tracks whatever pitched source is loudest/clearest at each moment.
 */

import { yinPitch, rms } from './pitch.js';
import { freqToMidi, midiToNoteName } from './notes.js';

const WINDOW = 2048;
const HOP = 512;           // ~11.6 ms @ 44.1kHz
const YIELD_EVERY = 40;    // yield to UI every N frames
const SILENCE_RMS = 0.005; // below this the segment is considered silence
const MIN_NOTE_SEC = 0.12; // merge anything shorter than this

export async function analyzeFile(file, { onProgress, signal } = {}) {
  const ac = new (window.OfflineAudioContext || window.webkitOfflineAudioContext
    || window.AudioContext || window.webkitAudioContext)(1, 44100, 44100);
  // OfflineAudioContext needs concrete length before decode; use a regular ctx just for decodeAudioData
  const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
  const ab = await file.arrayBuffer();
  const audioBuf = await decodeCtx.decodeAudioData(ab.slice(0));
  decodeCtx.close();

  const sr = audioBuf.sampleRate;
  const mono = toMono(audioBuf);
  const totalFrames = Math.max(0, Math.floor((mono.length - WINDOW) / HOP));

  const rawFrames = []; // { t, freq, midi, rms }
  for (let f = 0; f < totalFrames; f++) {
    if (signal && signal.aborted) throw new Error('aborted');
    const start = f * HOP;
    const chunk = mono.subarray(start, start + WINDOW);
    const level = rms(chunk);
    let freq = -1;
    if (level >= SILENCE_RMS) {
      freq = yinPitch(chunk, sr, 0.15);
    }
    const t = (start + WINDOW / 2) / sr;
    rawFrames.push({
      t,
      freq: freq > 0 ? freq : 0,
      midi: freq > 0 ? Math.round(freqToMidi(freq)) : null,
      rms: level
    });
    if (f % YIELD_EVERY === 0) {
      if (onProgress) onProgress(f / totalFrames);
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  if (onProgress) onProgress(1);

  // Median filter MIDI to kill isolated detection errors
  const smoothed = medianFilterMidi(rawFrames, 5);

  // Segment into notes
  const notes = segmentNotes(smoothed, HOP, sr);

  const melody = notes.map((n) => ({
    t: +n.t.toFixed(3),
    note: midiToNoteName(n.midi),
    dur: +n.dur.toFixed(3)
  }));

  return {
    melody,
    frames: smoothed,
    duration: audioBuf.duration,
    sampleRate: sr
  };
}

function toMono(buf) {
  if (buf.numberOfChannels === 1) return buf.getChannelData(0);
  const out = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const data = buf.getChannelData(c);
    for (let i = 0; i < data.length; i++) out[i] += data[i];
  }
  const scale = 1 / buf.numberOfChannels;
  for (let i = 0; i < out.length; i++) out[i] *= scale;
  return out;
}

function medianFilterMidi(frames, k) {
  const half = Math.floor(k / 2);
  const out = frames.map((f) => ({ ...f }));
  for (let i = 0; i < frames.length; i++) {
    const window = [];
    for (let j = Math.max(0, i - half); j <= Math.min(frames.length - 1, i + half); j++) {
      if (frames[j].midi != null) window.push(frames[j].midi);
    }
    if (!window.length) { out[i].midi = null; continue; }
    window.sort((a, b) => a - b);
    out[i].midi = window[Math.floor(window.length / 2)];
  }
  return out;
}

function segmentNotes(frames, hop, sr) {
  const minFrames = Math.ceil((MIN_NOTE_SEC * sr) / hop);
  const notes = [];
  let cur = null;
  for (let i = 0; i < frames.length; i++) {
    const m = frames[i].midi;
    if (m == null) {
      if (cur) { finalize(cur); cur = null; }
      continue;
    }
    if (!cur || cur.midi !== m) {
      if (cur) finalize(cur);
      cur = { midi: m, startIdx: i, endIdx: i };
    } else {
      cur.endIdx = i;
    }
  }
  if (cur) finalize(cur);

  function finalize(c) {
    const count = c.endIdx - c.startIdx + 1;
    if (count < minFrames) return;
    const t = frames[c.startIdx].t;
    const endT = frames[c.endIdx].t;
    notes.push({ midi: c.midi, t, dur: Math.max(MIN_NOTE_SEC, endT - t) });
  }
  return notes;
}

export function exportMelodyJson(melody) {
  return JSON.stringify(melody, null, 2);
}
