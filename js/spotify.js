const ARTIST_SUGGESTIONS = ['Bryson Tiller', 'Chris Brown', 'The Weeknd', 'Brent Faiyaz', 'PARTYNEXTDOOR', 'Summer Walker'];

export function renderSpotify(root) {
  root.innerHTML = `
    <p class="hint">Quick links to Spotify. Opens the app on iPhone if installed, otherwise the web player.</p>
    <div class="col" style="gap:8px;">
      <input type="search" id="spotify-q" placeholder="Search Spotify — song, artist, album" />
      <button class="btn primary full" id="spotify-go">Open in Spotify</button>
    </div>
    <div class="hint" style="margin-top:10px;">Suggested voices to study:</div>
    <div class="row" style="flex-wrap:wrap; gap:6px; margin-top:6px;">
      ${ARTIST_SUGGESTIONS.map((a) => `<button class="badge" data-artist="${a}">${a}</button>`).join('')}
    </div>
  `;

  const input = root.querySelector('#spotify-q');
  const go = () => {
    const q = input.value.trim();
    if (!q) return;
    openSpotifySearch(q);
  };
  root.querySelector('#spotify-go').addEventListener('click', go);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
  root.querySelectorAll('[data-artist]').forEach((b) => {
    b.addEventListener('click', () => { input.value = b.getAttribute('data-artist'); go(); });
  });
}

function openSpotifySearch(query) {
  const encoded = encodeURIComponent(query);
  // On iOS Safari, open.spotify.com links hand off to the app if installed.
  const webUrl = `https://open.spotify.com/search/${encoded}`;
  window.open(webUrl, '_blank', 'noopener');
}
