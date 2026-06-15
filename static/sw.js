// static/sw.js — Odysseus PWA Service Worker
// Strategy:
//   - HTML (navigation): stale-while-revalidate. Instant open from cache,
//     background refresh so the next open has latest HTML.
//   - JS/CSS (/static/*.js|.css): stale-while-revalidate. Boot instantly from
//     cache (no network wait for ~45 modules + the 1.1 MB stylesheet), refresh
//     in the background. When a background fetch brings genuinely newer bytes
//     (ETag/Last-Modified/size differ from what we served), tell open clients so
//     they can surface a subtle "Update available — Reload" prompt. Net effect:
//     fast boot AND edits still reach the user — one reload later instead of
//     blocking every reload on the network.
//   - Other static assets (images/fonts/libs): cache-first with bg refresh.
//   - API / non-GET: never cached.
// Bump CACHE_NAME whenever the precache list or SW logic changes.
const CACHE_NAME = 'odysseus-v332';

// Has this SW instance already told clients about a background update? Debounced
// so a deploy that changes many modules yields ONE prompt, not one per file.
let _updateAnnounced = false;

// Did the served bytes differ from the freshly-fetched ones? Compare the cheap
// validators the server already sends (ETag, then Last-Modified, then size).
// Conservative: only report "changed" when a validator is present and differs,
// so we never nag on a normal unchanged reload.
function _assetChanged(cachedRes, networkRes) {
  if (!cachedRes || !networkRes) return false;
  const ce = cachedRes.headers.get('ETag'), ne = networkRes.headers.get('ETag');
  if (ce && ne) return ce !== ne;
  const cl = cachedRes.headers.get('Last-Modified'), nl = networkRes.headers.get('Last-Modified');
  if (cl && nl) return cl !== nl;
  const cs = cachedRes.headers.get('Content-Length'), ns = networkRes.headers.get('Content-Length');
  if (cs && ns) return cs !== ns;
  return false;
}

function _announceUpdate() {
  if (_updateAnnounced) return;
  _updateAnnounced = true;
  self.clients.matchAll({ includeUncontrolled: true }).then(clients => {
    clients.forEach(c => c.postMessage({ type: 'assets-updated' }));
  });
}

// Core shell precached on install so repeat opens are instant without any
// network wait. Keep this list in sync with the <script type="module"> tags
// and <link rel="stylesheet"> in index.html.
const PRECACHE = [
  '/',
  '/static/style.css',
  '/static/app.js',
  '/static/js/storage.js',
  '/static/js/ui.js',
  '/static/js/markdown.js',
  '/static/js/dragSort.js',
  '/static/js/sessions.js',
  '/static/js/memory.js',
  '/static/js/skills.js',
  '/static/js/tourHints.js',
  '/static/js/fileHandler.js',
  '/static/js/voiceRecorder.js',
  '/static/js/models.js',
  '/static/js/rag.js',
  '/static/js/presets.js',
  '/static/js/search.js',
  '/static/js/spinner.js',
  '/static/js/tts-ai.js',
  '/static/js/document.js',
  '/static/js/gallery.js',
  '/static/js/chatRenderer.js',
  '/static/js/codeRunner.js',
  '/static/js/chatStream.js',
  '/static/js/chat.js',
  '/static/js/cookbook.js',
  '/static/js/forge.js',
  '/static/js/search-chat.js',
  '/static/js/compare/index.js',
  '/static/js/theme.js',
  '/static/js/censor.js',
  '/static/js/settings.js',
  '/static/js/admin.js',
  '/static/js/init.js',
  '/static/js/slashCommands.js',
  '/static/js/emailInbox.js',
  '/static/js/emailLibrary/utils.js',
  '/static/js/emailLibrary/signatureFold.js',
  '/static/js/emailLibrary/state.js',
  '/static/js/notes.js',
  '/static/js/tasks.js',
  '/static/js/calendar.js',
  '/static/js/calendar/utils.js',
  '/static/js/calendar/reminders.js',
  '/static/js/group.js',
  '/static/js/keyboard-shortcuts.js',
  '/static/js/sidebar-layout.js',
  '/static/js/section-management.js',
  '/static/lib/highlight.min.js',
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then(cache =>
      // addAll is atomic — if any item fails, none are cached. Use individual
      // puts so a single 404 can't block the whole install.
      Promise.all(
        PRECACHE.map(url =>
          fetch(url, { cache: 'reload' })
            .then(res => res.ok ? cache.put(url, res) : null)
            .catch(() => null)
        )
      )
    )
  );
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // Never touch API calls or non-GET.
  if (url.pathname.startsWith('/api/') || e.request.method !== 'GET') return;

  // HTML navigation: stale-while-revalidate the app shell — but ONLY for the
  // SPA root. Other navigations (e.g. a deep-linked /static/*.html page) must
  // go to the network/static handlers below; otherwise every navigation was
  // served the app index, replacing the page the user actually asked for.
  if (e.request.mode === 'navigate' && url.pathname === '/') {
    e.respondWith(
      caches.open(CACHE_NAME).then(async cache => {
        const cached = await cache.match('/');
        const network = fetch(e.request).then(res => {
          if (res && res.ok) cache.put('/', res.clone());
          return res;
        }).catch(() => cached);
        return cached || network;
      })
    );
    return;
  }

  // JS/CSS: stale-while-revalidate — serve the cached copy instantly so the app
  // boots without waiting on the network, then refresh the cache in the
  // background. If the refresh brought newer bytes, tell the page so it can
  // offer a reload. First-ever load (cold cache) falls through to the network.
  if (url.pathname.startsWith('/static/') && /\.(js|css)(\?|$)/.test(url.pathname + url.search)) {
    e.respondWith(
      caches.open(CACHE_NAME).then(async cache => {
        const cached = await cache.match(e.request);
        const networkFetch = fetch(e.request).then(res => {
          if (res && res.ok) {
            if (_assetChanged(cached, res)) _announceUpdate();
            cache.put(e.request, res.clone());
          }
          return res;
        }).catch(() => cached);
        // Cached → instant. No cache yet → wait for the network this once.
        return cached || networkFetch;
      })
    );
    return;
  }

  // Other static assets (images, fonts, libs): cache-first with background refresh.
  if (url.pathname.startsWith('/static/')) {
    e.respondWith(
      caches.open(CACHE_NAME).then(async cache => {
        const cached = await cache.match(e.request);
        const fetching = fetch(e.request).then(res => {
          if (res && res.ok) cache.put(e.request, res.clone());
          return res;
        }).catch(() => cached);
        return cached || fetching;
      })
    );
    return;
  }
});
