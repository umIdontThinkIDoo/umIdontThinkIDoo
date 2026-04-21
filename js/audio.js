import { yinPitch, rms } from './pitch.js';
import { freqToNote } from './notes.js';

const BUFFER_SIZE = 2048;
const MIN_RMS = 0.01;

class AudioEngine {
  constructor() {
    this.ctx = null;
    this.stream = null;
    this.source = null;
    this.analyser = null;
    this.timeBuf = null;
    this.listeners = new Set();
    this.running = false;
    this.rafId = null;
    this.recorder = null;
    this.recordChunks = [];
    this.recordStartedAt = 0;
    this.onStatusChange = null;
  }

  get isRunning() { return this.running; }
  get sampleRate() { return this.ctx ? this.ctx.sampleRate : 44100; }

  async start() {
    if (this.running) return;
    try {
      // Request raw mic — critical for accurate pitch
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
          channelCount: 1
        },
        video: false
      });
    } catch (err) {
      throw new Error('Microphone permission denied or unavailable. On iPhone: Settings → Safari → Microphone.');
    }

    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === 'suspended') await this.ctx.resume();

    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = BUFFER_SIZE;
    this.analyser.smoothingTimeConstant = 0;
    this.source.connect(this.analyser);
    this.timeBuf = new Float32Array(this.analyser.fftSize);

    this.running = true;
    this._tick();
    this._emitStatus();
  }

  stop() {
    if (!this.running) return;
    this.running = false;
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = null;
    if (this.recorder && this.recorder.state !== 'inactive') {
      try { this.recorder.stop(); } catch (_) {}
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.ctx) {
      this.ctx.close();
      this.ctx = null;
    }
    this.source = null;
    this.analyser = null;
    this._emitStatus();
  }

  _tick = () => {
    if (!this.running || !this.analyser) return;
    this.analyser.getFloatTimeDomainData(this.timeBuf);
    const level = rms(this.timeBuf);
    let freq = -1;
    let note = null;
    if (level >= MIN_RMS) {
      freq = yinPitch(this.timeBuf, this.ctx.sampleRate);
      if (freq > 0) note = freqToNote(freq);
    }
    const frame = {
      time: this.ctx.currentTime,
      level,
      freq,
      note,
      buffer: this.timeBuf
    };
    this.listeners.forEach((fn) => {
      try { fn(frame); } catch (e) { console.error(e); }
    });
    this.rafId = requestAnimationFrame(this._tick);
  };

  subscribe(fn) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  _emitStatus() {
    if (this.onStatusChange) this.onStatusChange(this.running);
  }

  // --- Recording ---
  startRecording() {
    if (!this.stream) throw new Error('Start microphone first.');
    if (this.recorder && this.recorder.state === 'recording') return;
    const mime = pickSupportedMime();
    this.recordChunks = [];
    this.recorder = new MediaRecorder(this.stream, mime ? { mimeType: mime } : undefined);
    this.recorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) this.recordChunks.push(e.data); };
    this.recordStartedAt = this.ctx.currentTime;
    this.recorder.start(250);
  }

  stopRecording() {
    return new Promise((resolve, reject) => {
      if (!this.recorder || this.recorder.state === 'inactive') {
        resolve(null);
        return;
      }
      this.recorder.onstop = () => {
        const blob = new Blob(this.recordChunks, { type: this.recorder.mimeType || 'audio/webm' });
        resolve({ blob, mimeType: this.recorder.mimeType, durationSec: this.ctx.currentTime - this.recordStartedAt });
      };
      this.recorder.onerror = (e) => reject(e.error || new Error('Recorder error'));
      this.recorder.stop();
    });
  }

  get isRecording() {
    return !!this.recorder && this.recorder.state === 'recording';
  }

  // --- Play a pure tone for drill prompts ---
  playTone(freq, durationMs = 900, gain = 0.12) {
    if (!this.ctx) return;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = freq;
    const now = this.ctx.currentTime;
    g.gain.setValueAtTime(0, now);
    g.gain.linearRampToValueAtTime(gain, now + 0.02);
    g.gain.linearRampToValueAtTime(gain, now + (durationMs / 1000) - 0.05);
    g.gain.linearRampToValueAtTime(0, now + (durationMs / 1000));
    osc.connect(g).connect(this.ctx.destination);
    osc.start(now);
    osc.stop(now + (durationMs / 1000) + 0.02);
  }
}

function pickSupportedMime() {
  if (typeof MediaRecorder === 'undefined') return null;
  const candidates = ['audio/mp4', 'audio/webm;codecs=opus', 'audio/webm', 'audio/aac'];
  for (const m of candidates) {
    if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(m)) return m;
  }
  return null;
}

export const audio = new AudioEngine();
