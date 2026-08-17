const DB_NAME = "dpo-labeler-v3";
const DB_VERSION = 1;
const META_STORE = "meta";
const PENDING_STORE = "pendingEvents";
const PAIR_STORE = "pairCache";

function openDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(META_STORE)) {
        database.createObjectStore(META_STORE, { keyPath: "key" });
      }
      if (!database.objectStoreNames.contains(PENDING_STORE)) {
        database.createObjectStore(PENDING_STORE, { keyPath: "event_id" });
      }
      if (!database.objectStoreNames.contains(PAIR_STORE)) {
        database.createObjectStore(PAIR_STORE, { keyPath: "pair_key" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function runTransaction(storeName, mode, callback) {
  return openDb().then(
    (database) =>
      new Promise((resolve, reject) => {
        const transaction = database.transaction(storeName, mode);
        const store = transaction.objectStore(storeName);
        const request = callback(store);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      })
  );
}

export const storage = {
  async getMeta(key) {
    const record = await runTransaction(META_STORE, "readonly", (store) => store.get(key));
    return record ? record.value : null;
  },
  async setMeta(key, value) {
    return runTransaction(META_STORE, "readwrite", (store) => store.put({ key, value }));
  },
  async deleteMeta(key) {
    return runTransaction(META_STORE, "readwrite", (store) => store.delete(key));
  },
  async getAllPendingEvents() {
    return runTransaction(PENDING_STORE, "readonly", (store) => store.getAll());
  },
  async putPendingEvent(value) {
    return runTransaction(PENDING_STORE, "readwrite", (store) => store.put(value));
  },
  async deletePendingEvent(key) {
    return runTransaction(PENDING_STORE, "readwrite", (store) => store.delete(key));
  },
  async putPair(pair) {
    return runTransaction(PAIR_STORE, "readwrite", (store) => store.put({ pair_key: pair.pair_key, value: pair }));
  },
  async getPair(pairKey) {
    const record = await runTransaction(PAIR_STORE, "readonly", (store) => store.get(pairKey));
    return record ? record.value : null;
  },
  async deletePair(pairKey) {
    return runTransaction(PAIR_STORE, "readwrite", (store) => store.delete(pairKey));
  },
};
