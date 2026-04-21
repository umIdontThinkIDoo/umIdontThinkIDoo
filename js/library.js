import { listRecordings, deleteRecording, listSessions, settings } from './storage.js';
import { midiToNoteName, classifyVoiceType } from './notes.js';
import { renderSpotify } from './spotify.js';
import { toast } from './app.js';

export function renderLibrary(root) {
  root.innerHTML = `
    <div class="card">
      <h2>Voice profile</h2>
      <div id="voice-profile"></div>
    </div>

    <div class="card">
      <h2>Progress</h2>
      <div id="progress-stats"></div>
    </div>

    <div class="card">
      <h2>Recordings</h2>
      <ul class="list" id="recording-list"><li class="empty">Loading…</li></ul>
    </div>

    <div class="card">
      <h2>Drill sessions</h2>
      <ul class="list" id="session-list"><li class="empty">Loading…</li></ul>
    </div>

    <div class="card" id="spotify-card">
      <h2>Spotify</h2>
      <div id="spotify-root"></div>
    </div>
  `;

  refresh(root);
  renderSpotify(root.querySelector('#spotify-root'));

  return {
    onActivate() { refresh(root); }
  };
}

async function refresh(root) {
  const [recs, sessions] = await Promise.all([listRecordings(), listSessions()]);

  // Voice profile
  const allLow = settings.get('range.low', null);
  const allHigh = settings.get('range.high', null);
  const voice = (allLow && allHigh) ? classifyVoiceType(allLow.midi, allHigh.midi) : null;
  const semitones = (allLow && allHigh) ? (allHigh.midi - allLow.midi) : 0;

  const vp = root.querySelector('#voice-profile');
  vp.innerHTML = `
    <div class="kv"><span>Detected voice type</span><span>${voice || 'Need more data'}</span></div>
    <div class="kv"><span>All-time low</span><span>${allLow ? allLow.name : '—'}</span></div>
    <div class="kv"><span>All-time high</span><span>${allHigh ? allHigh.name : '—'}</span></div>
    <div class="kv"><span>Range</span><span>${semitones} semitones (${(semitones / 12).toFixed(1)} oct)</span></div>
    <p class="hint" style="margin-top:8px;">Voice type is estimated from detected min/max notes across all sessions. Hit your extremes in the Tuner to improve the estimate. Classifying as Bryson Tiller / Chris Brown tenor range means comfortably reaching ~D4–A4 in chest and mix.</p>
  `;

  // Progress stats
  const drills = sessions.filter((s) => s.kind === 'drill');
  const last = drills.slice(0, 5);
  const avgRecent = last.length ? last.reduce((a, b) => a + (b.avgAccuracy || 0), 0) / last.length : 0;
  const allAvg = drills.length ? drills.reduce((a, b) => a + (b.avgAccuracy || 0), 0) / drills.length : 0;
  root.querySelector('#progress-stats').innerHTML = `
    <div class="row" style="gap:16px; align-items:center;">
      <div class="score-ring" style="--val:${Math.round(avgRecent)}"><div>${Math.round(avgRecent)}%</div></div>
      <div class="col" style="flex:1;">
        <div class="kv"><span>Recent drill accuracy</span><span>${Math.round(avgRecent)}%</span></div>
        <div class="kv"><span>All-time drill accuracy</span><span>${Math.round(allAvg)}%</span></div>
        <div class="kv"><span>Total drills</span><span>${drills.length}</span></div>
        <div class="kv"><span>Total recordings</span><span>${recs.length}</span></div>
      </div>
    </div>
  `;

  // Recordings
  const recList = root.querySelector('#recording-list');
  if (!recs.length) {
    recList.innerHTML = '<li class="empty">No recordings yet. Use the Record tab.</li>';
  } else {
    recList.innerHTML = '';
    for (const r of recs) {
      const li = document.createElement('li');
      const date = new Date(r.createdAt).toLocaleString();
      const lowHigh = (r.lowMidi != null && r.highMidi != null)
        ? `${midiToNoteName(r.lowMidi)} – ${midiToNoteName(r.highMidi)}`
        : '—';
      const dur = r.durationSec ? `${r.durationSec.toFixed(1)}s` : '';
      const url = URL.createObjectURL(r.blob);
      li.innerHTML = `
        <div class="row between">
          <strong>${labelFor(r)}</strong>
          <span class="meta">${date}</span>
        </div>
        <audio controls src="${url}" preload="none" style="width:100%;"></audio>
        <div class="row between">
          <span class="meta">${dur} · range ${lowHigh}${r.score != null ? ' · ' + r.score + '%' : ''}</span>
          <button class="btn ghost" data-del="${r.id}">Delete</button>
        </div>
      `;
      recList.appendChild(li);
    }
    recList.querySelectorAll('[data-del]').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const id = Number(btn.getAttribute('data-del'));
        await deleteRecording(id);
        toast('Deleted');
        refresh(root);
      });
    });
  }

  // Sessions
  const sessList = root.querySelector('#session-list');
  if (!drills.length) {
    sessList.innerHTML = '<li class="empty">No drill sessions yet.</li>';
  } else {
    sessList.innerHTML = '';
    for (const s of drills.slice(0, 20)) {
      const li = document.createElement('li');
      const date = new Date(s.createdAt).toLocaleString();
      li.innerHTML = `
        <div class="row between">
          <strong>${s.drillId}</strong>
          <span class="badge ${accuracyClass(s.avgAccuracy)}">${Math.round(s.avgAccuracy || 0)}%</span>
        </div>
        <div class="meta">${date} · tonic ${s.tonic}</div>
      `;
      sessList.appendChild(li);
    }
  }
}

function labelFor(r) {
  if (r.kind === 'karaoke') return `Karaoke test${r.songName ? ' — ' + r.songName : ''}`;
  if (r.kind === 'drill') return `Drill — ${r.drillId || ''}`;
  return 'Freeform recording';
}

function accuracyClass(val) {
  const v = val || 0;
  if (v >= 80) return 'ok';
  if (v >= 50) return 'warn';
  return 'bad';
}
