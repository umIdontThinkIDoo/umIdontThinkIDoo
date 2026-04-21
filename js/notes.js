const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const A4 = 440;
const A4_MIDI = 69;

export function freqToMidi(freq) {
  if (!freq || freq <= 0) return NaN;
  return A4_MIDI + 12 * Math.log2(freq / A4);
}

export function midiToFreq(midi) {
  return A4 * Math.pow(2, (midi - A4_MIDI) / 12);
}

export function midiToNoteName(midi) {
  const rounded = Math.round(midi);
  const name = NOTE_NAMES[((rounded % 12) + 12) % 12];
  const octave = Math.floor(rounded / 12) - 1;
  return `${name}${octave}`;
}

export function freqToNote(freq) {
  const midi = freqToMidi(freq);
  if (!isFinite(midi)) return null;
  const rounded = Math.round(midi);
  const cents = Math.round((midi - rounded) * 100);
  return {
    midi: rounded,
    name: midiToNoteName(rounded),
    cents,
    exactFreq: midiToFreq(rounded)
  };
}

export function noteNameToMidi(name) {
  const m = /^([A-Ga-g])([#b]?)(-?\d+)$/.exec(name.trim());
  if (!m) return NaN;
  const step = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }[m[1].toUpperCase()];
  const acc = m[2] === '#' ? 1 : m[2] === 'b' ? -1 : 0;
  const oct = parseInt(m[3], 10);
  return (oct + 1) * 12 + step + acc;
}

/**
 * Classify voice type from measured min/max MIDI notes.
 * Approximate — singing teachers use tessitura too, but range is a decent start.
 */
export function classifyVoiceType(minMidi, maxMidi) {
  if (!isFinite(minMidi) || !isFinite(maxMidi)) return null;
  const span = maxMidi - minMidi;
  if (span < 10) return 'Developing range';

  const candidates = [
    { name: 'Bass',          low: 40, high: 64 },  // E2-E4
    { name: 'Baritone',      low: 43, high: 67 },  // G2-G4
    { name: 'Tenor',         low: 48, high: 72 },  // C3-C5
    { name: 'Countertenor',  low: 52, high: 76 },  // E3-E5
    { name: 'Contralto',     low: 55, high: 77 },  // G3-F5
    { name: 'Mezzo-soprano', low: 57, high: 81 },  // A3-A5
    { name: 'Soprano',       low: 60, high: 84 }   // C4-C6
  ];
  let best = null;
  let bestScore = Infinity;
  for (const c of candidates) {
    const score = Math.abs(minMidi - c.low) + Math.abs(maxMidi - c.high);
    if (score < bestScore) { bestScore = score; best = c.name; }
  }
  return best;
}

/**
 * Shift a melody so it sits comfortably inside a user's measured range.
 * Strategy:
 *   - If the melody's span fits inside the user's span, center the melody's
 *     mid-pitch on the user's mid-pitch.
 *   - Otherwise, the melody is wider than the user — anchor the highest note
 *     just below the user's ceiling with 2 semitones of headroom so the peak
 *     notes are still reachable.
 * Returns the shifted melody (array of {t, note, dur}) and the shift in
 * semitones (negative = transposed down).
 */
export function transposeForRange(melody, userLowMidi, userHighMidi) {
  if (!Array.isArray(melody) || !melody.length) return { melody, shift: 0 };
  if (!isFinite(userLowMidi) || !isFinite(userHighMidi)) return { melody, shift: 0 };

  const midis = melody.map((n) => noteNameToMidi(n.note)).filter((m) => isFinite(m));
  if (!midis.length) return { melody, shift: 0 };
  const melLow = Math.min(...midis);
  const melHigh = Math.max(...midis);
  const melCenter = (melLow + melHigh) / 2;
  const userCenter = (userLowMidi + userHighMidi) / 2;
  const userSpan = userHighMidi - userLowMidi;
  const melSpan = melHigh - melLow;

  let shift;
  if (melSpan <= userSpan) {
    shift = Math.round(userCenter - melCenter);
  } else {
    // Wider than user — park the peak 2 semitones below their ceiling
    shift = userHighMidi - melHigh - 2;
  }

  const transposed = melody.map((n) => {
    const midi = noteNameToMidi(n.note);
    if (!isFinite(midi)) return n;
    return { ...n, note: midiToNoteName(midi + shift) };
  });
  return { melody: transposed, shift };
}

export { NOTE_NAMES };
