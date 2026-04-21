import { audio } from './audio.js';
import { settings } from './storage.js';

export function renderTuner(root) {
  root.innerHTML = `
    <div class="card">
      <div class="tuner">
        <div class="tuner-note" id="tuner-note">—</div>
        <div class="tuner-freq" id="tuner-freq">0.00 Hz</div>
        <div class="tuner-cents" id="tuner-cents">0 cents</div>
        <div class="meter" id="tuner-meter" aria-label="Cents meter">
          <div class="tick" style="left:10%"></div>
          <div class="tick" style="left:30%"></div>
          <div class="tick center" style="left:50%"></div>
          <div class="tick" style="left:70%"></div>
          <div class="tick" style="left:90%"></div>
          <div class="needle" id="tuner-needle"></div>
        </div>
        <canvas class="wave" id="tuner-wave" width="600" height="120"></canvas>
      </div>
    </div>

    <div class="card">
      <div class="row between">
        <div>
          <div style="font-weight:600;">Live range tracking</div>
          <div class="hint">Your lowest / highest detected notes this session and all-time.</div>
        </div>
        <button class="btn ghost" id="reset-range">Reset</button>
      </div>
      <div class="kv"><span>Session low</span><span id="s-low">—</span></div>
      <div class="kv"><span>Session high</span><span id="s-high">—</span></div>
      <div class="kv"><span>All-time low</span><span id="a-low">—</span></div>
      <div class="kv"><span>All-time high</span><span id="a-high">—</span></div>
    </div>

    <div class="card">
      <div class="row" style="gap:10px; flex-wrap:wrap;">
        <button class="btn primary" id="mic-toggle">Start mic</button>
        <span class="hint">iOS: tap once to grant mic permission. Use headphones to avoid feedback.</span>
      </div>
    </div>
  `;

  const noteEl = root.querySelector('#tuner-note');
  const freqEl = root.querySelector('#tuner-freq');
  const centsEl = root.querySelector('#tuner-cents');
  const needle = root.querySelector('#tuner-needle');
  const canvas = root.querySelector('#tuner-wave');
  const ctx2d = canvas.getContext('2d');
  const btn = root.querySelector('#mic-toggle');
  const sLow = root.querySelector('#s-low');
  const sHigh = root.querySelector('#s-high');
  const aLow = root.querySelector('#a-low');
  const aHigh = root.querySelector('#a-high');
  const resetBtn = root.querySelector('#reset-range');

  let sessionLow = null, sessionHigh = null;
  let allLow = settings.get('range.low', null);
  let allHigh = settings.get('range.high', null);

  const fmt = (n) => (n ? `${n.name} (${n.freq.toFixed(1)} Hz)` : '—');
  const refreshRange = () => {
    sLow.textContent = fmt(sessionLow);
    sHigh.textContent = fmt(sessionHigh);
    aLow.textContent = fmt(allLow);
    aHigh.textContent = fmt(allHigh);
  };
  refreshRange();

  const updateBtn = () => {
    btn.textContent = audio.isRunning ? 'Stop mic' : 'Start mic';
    btn.classList.toggle('danger', audio.isRunning);
    btn.classList.toggle('primary', !audio.isRunning);
  };
  updateBtn();

  btn.addEventListener('click', async () => {
    try {
      if (audio.isRunning) {
        audio.stop();
      } else {
        await audio.start();
      }
      updateBtn();
    } catch (e) {
      alert(e.message || e);
    }
  });

  resetBtn.addEventListener('click', () => {
    sessionLow = null; sessionHigh = null;
    allLow = null; allHigh = null;
    settings.set('range.low', null);
    settings.set('range.high', null);
    refreshRange();
  });

  // Subscribe to audio frames
  const unsubscribe = audio.subscribe((frame) => {
    // Draw waveform
    drawWave(ctx2d, canvas, frame.buffer);

    if (frame.note && frame.freq > 0) {
      const { name, cents } = frame.note;
      noteEl.textContent = name;
      freqEl.textContent = `${frame.freq.toFixed(2)} Hz`;
      centsEl.textContent = `${cents > 0 ? '+' : ''}${cents} cents`;
      const absCents = Math.abs(cents);
      noteEl.classList.toggle('intune', absCents < 8);
      noteEl.classList.toggle('flat', cents < -8);
      noteEl.classList.toggle('sharp', cents > 8);
      // needle: map cents (-50..+50) → 0..100%
      const pct = Math.max(0, Math.min(100, 50 + cents));
      needle.style.left = pct + '%';
      needle.style.background = absCents < 8 ? 'var(--ok)' : absCents < 20 ? 'var(--warn)' : 'var(--bad)';

      // Track range
      const midi = frame.note.midi;
      if (!sessionLow || midi < sessionLow.midi) sessionLow = { midi, name, freq: frame.freq };
      if (!sessionHigh || midi > sessionHigh.midi) sessionHigh = { midi, name, freq: frame.freq };
      if (!allLow || midi < allLow.midi) { allLow = { midi, name, freq: frame.freq }; settings.set('range.low', allLow); }
      if (!allHigh || midi > allHigh.midi) { allHigh = { midi, name, freq: frame.freq }; settings.set('range.high', allHigh); }
      refreshRange();
    } else {
      noteEl.textContent = '—';
      freqEl.textContent = '0.00 Hz';
      centsEl.textContent = '0 cents';
      noteEl.classList.remove('intune', 'flat', 'sharp');
      needle.style.left = '50%';
      needle.style.background = 'var(--accent)';
    }
  });

  // Cleanup when view is replaced
  return () => unsubscribe();
}

function drawWave(ctx, canvas, buffer) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = '#15151d';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#7c5cff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  const slice = buffer.length / w;
  for (let x = 0; x < w; x++) {
    const v = buffer[Math.floor(x * slice)] || 0;
    const y = (1 - v) * (h / 2);
    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
