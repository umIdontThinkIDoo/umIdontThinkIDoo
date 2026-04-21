- 👋 Hi, I’m @umIdontThinkIDoo
- 👀 I’m interested in ... Web Development
- 🌱 I’m currently learning ... Html,Css,Js
- 💞️ I’m looking to collaborate on ... ANYTHING UNPAID
- 📫 How to reach me ... 863-368-3584
- ⚡ Fun fact: ... I'm cooler than ur mom 

<!---
umIdontThinkIDoo/umIdontThinkIDoo is a ✨ mid-level "special" ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->

---

## Range — iPhone singing practice web app

A mobile-first PWA for pitch training, vocal drills, recording, and karaoke testing. No build step — open `index.html` over HTTPS.

### Features
- **Tuner**: real-time pitch + cents meter with a live waveform, tracks your session / all-time range.
- **Drills**: scales, arpeggios, fifths, octave jumps, R&B runs — app plays the target, you match, get scored.
- **Record**: isolated raw-mic recording (echo cancel / noise suppression / AGC off) with pitch-track visualization.
- **Karaoke**: upload a backing track + reference melody JSON, sing along, get graded 0-100%.
- **Library**: saved recordings, drill progress over time, auto-detected voice type (bass / baritone / tenor / countertenor / contralto / mezzo / soprano) from your measured range.
- **Spotify**: quick search that opens the Spotify app on iPhone (or web player).

### Running locally
```bash
# Must be HTTPS (or localhost) for getUserMedia to work
python3 -m http.server 8000
# Then visit http://localhost:8000 on your phone (same Wi-Fi, use your laptop's LAN IP)
```

For iPhone Safari installation: open the site, tap **Share → Add to Home Screen**. Grants mic via the first tap.

### Deploying
Works on any static host (GitHub Pages, Netlify, Cloudflare Pages). HTTPS required for microphone access.

### Tech
Vanilla JS ES modules, Web Audio API, YIN pitch detection, MediaRecorder, IndexedDB, service worker for offline. No dependencies.

### Known limitations
- True live monitoring (hearing yourself through headphones with zero latency) is not possible on iOS Safari — the app shows visual pitch feedback instead.
- Karaoke scoring needs a reference melody JSON: `[{"t": 0, "note": "G4", "dur": 0.5}, ...]` with times/durations in seconds.
- Voice type classification needs you to actually sing your low and high extremes; use the Tuner to log them.
