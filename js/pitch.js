/**
 * YIN pitch detection (Alain de Cheveigné & Hideki Kawahara, 2002).
 * Returns frequency in Hz, or -1 if no clear pitch is detected.
 * Monophonic, designed for human voice. Threshold 0.1-0.15 tuned for singing.
 */
export function yinPitch(buffer, sampleRate, threshold = 0.1) {
  const N = buffer.length;
  const maxTau = Math.floor(N / 2);
  const yin = new Float32Array(maxTau);
  yin[0] = 1;

  // Step 1: difference function
  for (let tau = 1; tau < maxTau; tau++) {
    let sum = 0;
    for (let i = 0; i < maxTau; i++) {
      const d = buffer[i] - buffer[i + tau];
      sum += d * d;
    }
    yin[tau] = sum;
  }

  // Step 2: cumulative mean normalized difference
  let running = 0;
  for (let tau = 1; tau < maxTau; tau++) {
    running += yin[tau];
    yin[tau] = yin[tau] * tau / running;
  }

  // Step 3: absolute threshold - find first dip below threshold
  let tau = -1;
  for (let t = 2; t < maxTau; t++) {
    if (yin[t] < threshold) {
      while (t + 1 < maxTau && yin[t + 1] < yin[t]) t++;
      tau = t;
      break;
    }
  }
  if (tau === -1) return -1;

  // Step 4: parabolic interpolation around tau
  const x0 = tau > 0 ? yin[tau - 1] : yin[tau];
  const x2 = tau + 1 < maxTau ? yin[tau + 1] : yin[tau];
  const a = (x0 + x2 - 2 * yin[tau]) / 2;
  const b = (x2 - x0) / 2;
  const refinedTau = a !== 0 ? tau - b / (2 * a) : tau;

  const freq = sampleRate / refinedTau;
  if (freq < 55 || freq > 1500) return -1; // ignore subharmonics / noise outside singing range
  return freq;
}

/**
 * Compute RMS of a buffer to detect silence / gate low signal.
 */
export function rms(buffer) {
  let sum = 0;
  for (let i = 0; i < buffer.length; i++) sum += buffer[i] * buffer[i];
  return Math.sqrt(sum / buffer.length);
}
