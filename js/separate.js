/**
 * Zero-dependency stereo source separation via frequency-domain center-channel
 * extraction. Works because lead vocals in commercial mixes are panned center
 * with high phase coherence; stereo-spread instruments (guitars, pads, reverb)
 * are not.
 *
 * Algorithm per STFT frame:
 *   1. FFT left and right channels
 *   2. For each bin, compute:
 *      - panning similarity:    1 - |‖L‖-‖R‖| / (‖L‖+‖R‖)    → 1 when centered
 *      - phase coherence:       Re(L · conj(R)) / (‖L‖‖R‖)    → 1 when in-phase
 *   3. Soft mask = (pan · coh)^exponent clamped to [0,1]
 *   4. Apply mask to mid signal (L+R)/2
 *   5. Optional high-pass (zero low bins) to remove bass/kick rumble
 *   6. Inverse FFT + overlap-add
 */

import { fft, ifft, hannWindow } from './fft.js';

const FFT_SIZE = 4096;
const HOP = 1024;

export async function isolateCenter(audioBuf, { onProgress, highPassHz = 120, exponent = 2, signal } = {}) {
  const sr = audioBuf.sampleRate;
  const L = audioBuf.getChannelData(0);
  const R = audioBuf.numberOfChannels > 1 ? audioBuf.getChannelData(1) : L;
  const n = Math.min(L.length, R.length);

  if (audioBuf.numberOfChannels < 2) {
    // Nothing to separate; return a copy of the mono signal
    return L.slice(0, n);
  }

  const win = hannWindow(FFT_SIZE);
  // Precompute window^2 for overlap-add weight normalization
  const win2 = new Float32Array(FFT_SIZE);
  for (let i = 0; i < FFT_SIZE; i++) win2[i] = win[i] * win[i];

  const out = new Float32Array(n);
  const weight = new Float32Array(n);

  // Buffers reused per frame to avoid allocation pressure
  const lRe = new Float32Array(FFT_SIZE);
  const lIm = new Float32Array(FFT_SIZE);
  const rRe = new Float32Array(FFT_SIZE);
  const rIm = new Float32Array(FFT_SIZE);
  const mRe = new Float32Array(FFT_SIZE);
  const mIm = new Float32Array(FFT_SIZE);

  // High-pass cutoff in bin index
  const hpBin = Math.max(1, Math.floor((highPassHz * FFT_SIZE) / sr));

  const totalFrames = Math.max(1, Math.floor((n - FFT_SIZE) / HOP) + 1);
  let frameIdx = 0;

  for (let pos = 0; pos + FFT_SIZE <= n; pos += HOP, frameIdx++) {
    if (signal && signal.aborted) throw new Error('aborted');

    // Window + load into FFT buffers
    for (let i = 0; i < FFT_SIZE; i++) {
      lRe[i] = L[pos + i] * win[i];
      lIm[i] = 0;
      rRe[i] = R[pos + i] * win[i];
      rIm[i] = 0;
    }
    fft(lRe, lIm);
    fft(rRe, rIm);

    // Per-bin mask, apply to mid = (L+R)/2
    for (let k = 0; k < FFT_SIZE; k++) {
      // High-pass: zero out low and mirror-high bins below cutoff
      if (k < hpBin || k > FFT_SIZE - hpBin) {
        mRe[k] = 0; mIm[k] = 0; continue;
      }
      const lMag = Math.hypot(lRe[k], lIm[k]);
      const rMag = Math.hypot(rRe[k], rIm[k]);
      const sumMag = lMag + rMag;
      if (sumMag < 1e-9) { mRe[k] = 0; mIm[k] = 0; continue; }
      const pan = 1 - Math.abs(lMag - rMag) / sumMag;
      // Re(L * conj(R)) = lRe*rRe + lIm*rIm
      const dot = lRe[k] * rRe[k] + lIm[k] * rIm[k];
      let coh = dot / (lMag * rMag);
      if (coh < 0) coh = 0;
      let mask = Math.pow(pan * coh, exponent);
      if (mask > 1) mask = 1;
      // Mid channel
      const midRe = (lRe[k] + rRe[k]) * 0.5;
      const midIm = (lIm[k] + rIm[k]) * 0.5;
      mRe[k] = mask * midRe;
      mIm[k] = mask * midIm;
    }

    ifft(mRe, mIm);

    // Overlap-add with windowing
    for (let i = 0; i < FFT_SIZE; i++) {
      out[pos + i] += mRe[i] * win[i];
      weight[pos + i] += win2[i];
    }

    if ((frameIdx & 15) === 0) {
      if (onProgress) onProgress(frameIdx / totalFrames);
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  // Normalize by overlap-add weight
  for (let i = 0; i < n; i++) {
    if (weight[i] > 1e-9) out[i] /= weight[i];
  }
  if (onProgress) onProgress(1);
  return out;
}
