const DB_NAME = 'range-db';
const DB_VERSION = 1;
const STORE_RECORDINGS = 'recordings';
const STORE_SESSIONS = 'sessions';

let dbPromise = null;

function openDB() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_RECORDINGS)) {
        const s = db.createObjectStore(STORE_RECORDINGS, { keyPath: 'id', autoIncrement: true });
        s.createIndex('byDate', 'createdAt');
        s.createIndex('byKind', 'kind');
      }
      if (!db.objectStoreNames.contains(STORE_SESSIONS)) {
        const s = db.createObjectStore(STORE_SESSIONS, { keyPath: 'id', autoIncrement: true });
        s.createIndex('byDate', 'createdAt');
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function tx(store, mode = 'readonly') {
  return openDB().then((db) => db.transaction(store, mode).objectStore(store));
}

export async function addRecording(entry) {
  const store = await tx(STORE_RECORDINGS, 'readwrite');
  return new Promise((resolve, reject) => {
    const req = store.add({ createdAt: Date.now(), ...entry });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function listRecordings() {
  const store = await tx(STORE_RECORDINGS);
  return new Promise((resolve, reject) => {
    const req = store.getAll();
    req.onsuccess = () => resolve((req.result || []).sort((a, b) => b.createdAt - a.createdAt));
    req.onerror = () => reject(req.error);
  });
}

export async function deleteRecording(id) {
  const store = await tx(STORE_RECORDINGS, 'readwrite');
  return new Promise((resolve, reject) => {
    const req = store.delete(id);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

export async function addSession(entry) {
  const store = await tx(STORE_SESSIONS, 'readwrite');
  return new Promise((resolve, reject) => {
    const req = store.add({ createdAt: Date.now(), ...entry });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function listSessions() {
  const store = await tx(STORE_SESSIONS);
  return new Promise((resolve, reject) => {
    const req = store.getAll();
    req.onsuccess = () => resolve((req.result || []).sort((a, b) => b.createdAt - a.createdAt));
    req.onerror = () => reject(req.error);
  });
}

/**
 * Lightweight key-value settings in localStorage.
 */
export const settings = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem(`range:${key}`);
      return raw != null ? JSON.parse(raw) : fallback;
    } catch (_) { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`range:${key}`, JSON.stringify(value)); } catch (_) {}
  }
};
