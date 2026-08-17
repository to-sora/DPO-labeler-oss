import { api, fetchWithTimeout, HttpError, MEDIA_REQUEST_TIMEOUT_MS } from "./api.mjs";
import { storage } from "./storage.mjs";
import {
  buildCompactImageFacts,
  buildEmptyFilter,
  buildImageSummaryRows,
  buildLatestLabelRows,
  buildPairSummaryRows,
  buildLabelPayloadFromDisplay,
  cloneFilter,
  createKeyedValueCache,
  datasetProgress,
  deriveTaskDisplayLabels,
  formatPromptText,
  mapLatestLabelToDisplay,
  mergeQueueItems,
  pairKey,
  randomizePairDisplay,
} from "./app_helpers.mjs";

const POLL_INTERVAL_MS = 30_000;
const CLIENT_INSTANCE_KEY = "dpo_labeler_client_instance_id";
const REVIEW_QUEUE_PAGE_SIZE = 32;
const REVIEW_PREFETCH_WINDOW = 3;
const PREFETCH_PREVIEW_LIMIT = 20;
const DATA_CACHE_PREFIX = "dpo-labeler-data-";
const X_TEXT_LIMIT = 280;

function createEmptyReviewState() {
  return {
    reviewId: null,
    taskKeys: [],
    mode: "sequence",
    queue: [],
    total: 0,
    nextCursor: null,
    firstUnlabeledIndex: 0,
    currentPairKey: null,
    currentIndexHint: 0,
    currentPair: null,
    submitting: false,
  };
}

function createPrefetchState() {
  return {
    reviewId: null,
    generation: 0,
    inFlight: new Map(),
    warming: new Map(),
    entries: createKeyedValueCache({
      maxWeight: PREFETCH_PREVIEW_LIMIT,
      getWeight: (entry) => Array.isArray(entry?.previewUrls) ? entry.previewUrls.length : 0,
    }),
  };
}

function createSyncRequest() {
  return {
    requested: false,
    force: false,
    refreshCatalog: false,
    refreshReview: false,
  };
}

const state = {
  online: navigator.onLine,
  syncing: false,
  syncFailed: false,
  syncBackoffMs: 0,
  nextSyncAllowedAt: 0,
  syncRequest: createSyncRequest(),
  syncTimer: null,
  syncTimerDueAt: 0,
  config: null,
  catalog: null,
  session: null,
  currentView: "auth",
  review: createEmptyReviewState(),
  prefetch: createPrefetchState(),
  exportState: {
    exportType: "dpo-pairs",
    filter: buildEmptyFilter(),
    previewCount: null,
  },
  pendingCount: 0,
  clientInstanceId: null,
  pollTimer: null,
  shareState: {
    message: "",
    isError: false,
    pendingSlot: null,
  },
};

const elements = {
  statusPill: document.querySelector("#status-pill"),
  progressText: document.querySelector("#progress-text"),
  pendingCount: document.querySelector("#pending-count"),
  reviewerName: document.querySelector("#reviewer-name"),
  navTasks: document.querySelector("#nav-tasks"),
  navReview: document.querySelector("#nav-review"),
  navExport: document.querySelector("#nav-export"),
  logoutButton: document.querySelector("#logout-button"),
  authView: document.querySelector("#auth-view"),
  authForm: document.querySelector("#auth-form"),
  inviteToken: document.querySelector("#invite-token"),
  username: document.querySelector("#username"),
  authMessage: document.querySelector("#auth-message"),
  tasksView: document.querySelector("#tasks-view"),
  refreshCatalog: document.querySelector("#refresh-catalog"),
  selectAllTasks: document.querySelector("#select-all-tasks"),
  clearTaskSelection: document.querySelector("#clear-task-selection"),
  taskGroups: document.querySelector("#task-groups"),
  selectedTaskCount: document.querySelector("#selected-task-count"),
  selectedPairCount: document.querySelector("#selected-pair-count"),
  reviewMode: document.querySelector("#review-mode"),
  startReview: document.querySelector("#start-review"),
  catalogWarnings: document.querySelector("#catalog-warnings"),
  reviewView: document.querySelector("#review-view"),
  reviewModeChip: document.querySelector("#review-mode-chip"),
  reviewSubsetSummary: document.querySelector("#review-subset-summary"),
  backToTasks: document.querySelector("#back-to-tasks"),
  pairHeadline: document.querySelector("#pair-headline"),
  pairSubheadline: document.querySelector("#pair-subheadline"),
  pairSummary: document.querySelector("#pair-summary"),
  pairLatestLabel: document.querySelector("#pair-latest-label"),
  pairImages: document.querySelector("#pair-images"),
  mobileReviewStack: document.querySelector("#mobile-review-stack"),
  sharePanel: document.querySelector("#share-panel"),
  shareA: document.querySelector("#share-a"),
  shareB: document.querySelector("#share-b"),
  shareMessage: document.querySelector("#share-message"),
  defectsA: document.querySelector("#defects-a"),
  defectsB: document.querySelector("#defects-b"),
  pairNote: document.querySelector("#pair-note"),
  nextPair: document.querySelector("#next-pair"),
  exportView: document.querySelector("#export-view"),
  exportType: document.querySelector("#export-type"),
  exportPreview: document.querySelector("#export-preview"),
  exportDownload: document.querySelector("#export-download"),
  exportMessage: document.querySelector("#export-message"),
  filterRoot: document.querySelector("#filter-root"),
  zoomDialog: document.querySelector("#zoom-dialog"),
  zoomImage: document.querySelector("#zoom-image"),
  zoomClose: document.querySelector("#zoom-close"),
  imageTemplate: document.querySelector("#image-card-template"),
};

init().catch((error) => {
  console.error(error);
  renderAuthMessage(String(error));
});

async function init() {
  state.clientInstanceId = loadClientInstanceId();
  bindUi();
  await registerServiceWorker();
  await loadCachedState();
  await refreshPendingCount();
  await bootstrapSession();
  startPolling();
  renderShell();
}

function bindUi() {
  window.addEventListener("online", () => {
    state.online = true;
    state.syncFailed = false;
    state.syncBackoffMs = 0;
    state.nextSyncAllowedAt = 0;
    requestSync({ force: true, refreshCatalog: shouldRefreshCatalogSync(), refreshReview: shouldRefreshReviewSync() });
  });
  window.addEventListener("offline", () => {
    state.online = false;
    state.syncFailed = false;
    renderShell();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && state.session) {
      requestSync({ force: true, refreshCatalog: shouldRefreshCatalogSync(), refreshReview: shouldRefreshReviewSync() });
    }
  });

  elements.navTasks.addEventListener("click", () => switchView("tasks"));
  elements.navReview.addEventListener("click", () => switchView("review"));
  elements.navExport.addEventListener("click", () => switchView("export"));
  elements.logoutButton.addEventListener("click", () => void handleLogout());
  elements.refreshCatalog.addEventListener("click", () => void syncAll({ force: true, refreshCatalog: true }));
  elements.selectAllTasks.addEventListener("click", () => {
    state.review.taskKeys = collectAllTaskKeys();
    renderCatalog();
  });
  elements.clearTaskSelection.addEventListener("click", () => {
    state.review.taskKeys = [];
    renderCatalog();
  });
  elements.startReview.addEventListener("click", () => void handleStartReview());
  elements.backToTasks.addEventListener("click", () => switchView("tasks"));
  elements.nextPair.addEventListener("click", () => void loadPairAtIndex(getCurrentReviewIndex() + 1));
  elements.shareA.addEventListener("click", () => void handleShareImage("a"));
  elements.shareB.addEventListener("click", () => void handleShareImage("b"));
  elements.zoomClose.addEventListener("click", () => elements.zoomDialog.close());
  elements.exportType.addEventListener("change", (event) => {
    state.exportState.exportType = event.target.value;
  });
  elements.exportPreview.addEventListener("click", () => void handleExportPreview());
  elements.exportDownload.addEventListener("click", () => void handleExportDownload());

  elements.authForm.addEventListener("submit", (event) => {
    event.preventDefault();
    void handleLogin();
  });

  document.querySelectorAll("[data-decision]").forEach((button) => {
    button.addEventListener("click", () => void handleDecision(button.dataset.decision));
  });
}

async function registerServiceWorker() {
  if ("serviceWorker" in navigator) {
    await navigator.serviceWorker.register("/service-worker.js");
  }
}

async function loadCachedState() {
  const cachedCatalog = await storage.getMeta("catalog");
  const cachedReview = await storage.getMeta("review");
  if (cachedCatalog) {
    state.catalog = cachedCatalog;
  }
  if (cachedReview) {
    state.review = normalizeCachedReviewState(cachedReview);
  }
}

function reviewStatePayload() {
  return {
    reviewId: state.review.reviewId,
    taskKeys: state.review.taskKeys,
    mode: state.review.mode,
    queue: state.review.queue,
    total: state.review.total,
    nextCursor: state.review.nextCursor,
    firstUnlabeledIndex: state.review.firstUnlabeledIndex,
    currentPairKey: state.review.currentPairKey,
    currentIndexHint: getCurrentReviewIndex(),
  };
}

async function persistReviewState() {
  if (!state.review.reviewId) {
    await storage.deleteMeta("review");
    return;
  }
  await storage.setMeta("review", reviewStatePayload());
}

async function clearReviewState({ preserveTaskKeys = false } = {}) {
  const taskKeys = preserveTaskKeys ? [...state.review.taskKeys] : [];
  const mode = state.review.mode;
  const currentPairKey = state.review.currentPairKey;
  const currentPreviewUrls = state.review.currentPair?.images?.map((image) => image.preview_url) || [];
  await clearPrefetchState();
  if (currentPairKey) {
    await storage.deletePair(currentPairKey);
  }
  await deletePreviewUrlsFromCaches(currentPreviewUrls);
  state.review = {
    ...createEmptyReviewState(),
    taskKeys,
    mode,
  };
  state.shareState.pendingSlot = null;
  clearShareMessage();
  await storage.deleteMeta("review");
}

async function bootstrapSession() {
  try {
    const [configResponse, sessionResponse] = await Promise.all([api.getConfig(), api.getSession()]);
    state.config = configResponse.data;
    state.session = sessionResponse.data.session;
    elements.username.value = state.session.reviewer_username;
    await syncAll({ force: true, refreshCatalog: true, refreshReview: shouldRefreshReviewSync() });
    if (state.review.reviewId) {
      switchView("review");
    } else {
      switchView("tasks");
    }
  } catch (error) {
    if (error instanceof HttpError && error.status === 401) {
      try {
        const configResponse = await api.getConfig();
        state.config = configResponse.data;
      } catch (configError) {
        console.error(configError);
      }
      state.session = null;
      switchView("auth");
      return;
    }
    throw error;
  }
}

function startPolling() {
  if (state.pollTimer) {
    return;
  }
  state.pollTimer = window.setInterval(() => {
    if (state.session) {
      requestSync({ refreshCatalog: shouldRefreshCatalogSync(), refreshReview: shouldRefreshReviewSync() });
    }
  }, POLL_INTERVAL_MS);
}

function mergeSyncRequest(target, source = {}) {
  target.requested = target.requested || Boolean(source.requested);
  target.force = target.force || Boolean(source.force);
  target.refreshCatalog = target.refreshCatalog || Boolean(source.refreshCatalog);
  target.refreshReview = target.refreshReview || Boolean(source.refreshReview);
  return target;
}

function clearScheduledSync() {
  if (state.syncTimer) {
    window.clearTimeout(state.syncTimer);
    state.syncTimer = null;
    state.syncTimerDueAt = 0;
  }
}

function hasQueuedSyncRequest() {
  return Boolean(state.syncRequest.requested);
}

function takeSyncRequest(overrides = {}) {
  const request = createSyncRequest();
  mergeSyncRequest(request, { requested: true, ...overrides });
  mergeSyncRequest(request, state.syncRequest);
  state.syncRequest = createSyncRequest();
  clearScheduledSync();
  return request;
}

function shouldRefreshCatalogSync() {
  return !state.review.reviewId || state.currentView === "tasks" || state.currentView === "export";
}

function shouldRefreshReviewSync() {
  return Boolean(state.review.reviewId);
}

function scheduleQueuedSync() {
  if (!state.session || state.syncing || !hasQueuedSyncRequest()) {
    return;
  }
  const delayMs = state.syncRequest.force ? 0 : Math.max(0, state.nextSyncAllowedAt - Date.now());
  const dueAt = Date.now() + delayMs;
  if (state.syncTimer && state.syncTimerDueAt <= dueAt) {
    return;
  }
  clearScheduledSync();
  state.syncTimerDueAt = dueAt;
  state.syncTimer = window.setTimeout(() => {
    state.syncTimer = null;
    state.syncTimerDueAt = 0;
    void syncAll();
  }, delayMs);
}

function requestSync(options = {}) {
  if (!state.session) {
    return;
  }
  mergeSyncRequest(state.syncRequest, { requested: true, ...options });
  scheduleQueuedSync();
}

async function syncAll(options = {}) {
  if (!state.session) {
    clearScheduledSync();
    renderShell();
    return;
  }
  if (state.syncing) {
    requestSync(options);
    return;
  }
  const request = takeSyncRequest(options);
  if (!request.force && state.nextSyncAllowedAt && Date.now() < state.nextSyncAllowedAt) {
    mergeSyncRequest(state.syncRequest, request);
    scheduleQueuedSync();
    renderShell();
    return;
  }
  state.syncing = true;
  state.syncFailed = false;
  renderShell();
  try {
    await refreshPendingCount();
    if (state.online) {
      await flushPendingEvents();
      if (request.refreshCatalog) {
        await loadCatalog();
      }
      if (request.refreshReview && state.review.reviewId) {
        const loaded = await loadQueuePage(0, { replace: true });
        if (loaded && state.review.reviewId) {
          await loadInitialReviewPair();
        }
      }
    }
    state.syncBackoffMs = 0;
    state.nextSyncAllowedAt = 0;
  } catch (error) {
    console.error(error);
    if (error instanceof HttpError && error.status === 401) {
      state.session = null;
      state.syncFailed = false;
      state.syncBackoffMs = 0;
      state.nextSyncAllowedAt = 0;
      switchView("auth");
    } else {
      state.syncFailed = true;
      scheduleSyncRetry(request);
    }
  } finally {
    state.syncing = false;
    renderShell();
    scheduleQueuedSync();
  }
}

async function loadCatalog({ forceRender = false } = {}) {
  try {
    const response = await api.getCatalog();
    state.catalog = response.data;
    await storage.setMeta("catalog", response.data);
  } catch (error) {
    if (!state.catalog) {
      throw error;
    }
  }
  if (forceRender || state.currentView === "tasks" || state.currentView === "export") {
    renderCatalog();
    renderExportBuilder();
  }
}

async function handleLogin() {
  try {
    const response = await api.startSession({
      invite_token: elements.inviteToken.value.trim(),
      reviewer_username: elements.username.value.trim(),
      client_instance_id: state.clientInstanceId,
    });
    state.session = response.data.session;
    renderAuthMessage("");
    elements.inviteToken.value = "";
    state.syncBackoffMs = 0;
    state.nextSyncAllowedAt = 0;
    await syncAll({ force: true, refreshCatalog: true });
    switchView("tasks");
  } catch (error) {
    renderAuthMessage(error instanceof HttpError ? error.message : String(error));
  }
}

async function handleLogout() {
  try {
    await api.endSession();
  } catch (error) {
    console.error(error);
  }
  state.session = null;
  state.syncRequest = createSyncRequest();
  clearScheduledSync();
  await clearReviewState();
  switchView("auth");
  renderShell();
}

function switchView(view) {
  state.currentView = view;
  renderShell();
  if (view === "tasks") {
    renderCatalog();
    requestSync({ force: true, refreshCatalog: true });
  }
  if (view === "export") {
    renderExportBuilder();
    requestSync({ force: true, refreshCatalog: true });
  }
  if (view === "review") {
    renderReview();
  }
}

function normalizeCachedReviewState(cachedReview) {
  const review = {
    ...createEmptyReviewState(),
    ...cachedReview,
    queue: Array.isArray(cachedReview?.queue) ? cachedReview.queue : [],
    currentPair: null,
    submitting: false,
  };
  const currentIndexHint = Number(cachedReview?.currentIndexHint ?? cachedReview?.currentIndex ?? 0);
  review.currentIndexHint = Number.isFinite(currentIndexHint) ? Math.max(0, currentIndexHint) : 0;
  if (!review.currentPairKey && review.queue[review.currentIndexHint]) {
    review.currentPairKey = pairKey(review.queue[review.currentIndexHint].dataset_id, review.queue[review.currentIndexHint].session_id);
  }
  return review;
}

function getCurrentReviewIndex() {
  if (state.review.currentPairKey) {
    const index = state.review.queue.findIndex((item) => pairKey(item.dataset_id, item.session_id) === state.review.currentPairKey);
    if (index >= 0) {
      return index;
    }
  }
  const fallback = Number(state.review.currentIndexHint || 0);
  return Number.isFinite(fallback) ? Math.max(0, fallback) : 0;
}

async function loadInitialReviewPair() {
  if (!state.review.reviewId) {
    return false;
  }
  if (state.review.currentPairKey) {
    const loaded = await loadPairByKey(state.review.currentPairKey, { indexHint: getCurrentReviewIndex() });
    if (loaded) {
      return true;
    }
  }
  return loadPairAtIndex(Math.min(getCurrentReviewIndex(), Math.max(state.review.total - 1, 0)));
}

async function loadPairByKey(targetPairKey, { indexHint = 0 } = {}) {
  if (!targetPairKey) {
    return loadPairAtIndex(Math.min(indexHint, Math.max(state.review.total - 1, 0)));
  }
  await ensureQueueContainsPairKey(targetPairKey, indexHint);
  const index = state.review.queue.findIndex((item) => pairKey(item.dataset_id, item.session_id) === targetPairKey);
  if (index >= 0) {
    return loadPairAtIndex(index);
  }
  return loadPairAtIndex(Math.min(indexHint, Math.max(state.review.total - 1, 0)));
}

async function ensureQueueContainsPairKey(targetPairKey, indexHint = 0) {
  while (
    state.review.reviewId
    && state.review.nextCursor !== null
    && !state.review.queue.some((item) => pairKey(item.dataset_id, item.session_id) === targetPairKey)
    && state.review.queue.length <= indexHint
  ) {
    const loaded = await loadQueuePage(state.review.queue.length);
    if (!loaded) {
      return false;
    }
  }
  return state.review.queue.some((item) => pairKey(item.dataset_id, item.session_id) === targetPairKey);
}

async function ensureQueueLoadedThroughIndex(targetIndex) {
  if (targetIndex < 0) {
    return true;
  }
  while (state.review.reviewId && targetIndex >= state.review.queue.length && state.review.nextCursor !== null) {
    const loaded = await loadQueuePage(state.review.queue.length);
    if (!loaded) {
      return false;
    }
  }
  return true;
}

function prefetchContextIsActive(reviewId, generation) {
  return state.review.reviewId === reviewId && state.prefetch.reviewId === reviewId && state.prefetch.generation === generation;
}

function consumePrefetchedPair(pairKeyValue) {
  const entry = state.prefetch.entries.peek(pairKeyValue);
  if (!entry || entry.reviewId !== state.review.reviewId || entry.status !== "ready" || !entry.pair) {
    return null;
  }
  state.prefetch.entries.delete(pairKeyValue);
  return entry.pair;
}

async function resolvePairForDisplay(item, itemPairKey) {
  const prefetched = consumePrefetchedPair(itemPairKey);
  if (prefetched) {
    return prefetched;
  }
  const inFlight = state.prefetch.inFlight.get(itemPairKey);
  if (inFlight) {
    try {
      await inFlight;
      const ready = consumePrefetchedPair(itemPairKey);
      if (ready) {
        return ready;
      }
    } catch (error) {
      console.error(error);
    }
  }
  const response = await api.getReviewPair(state.review.reviewId, item.dataset_id, item.session_id);
  return response.data;
}

async function prefetchUpcomingPairs() {
  if (!state.online || !state.review.reviewId || !state.review.currentPairKey) {
    return;
  }
  const reviewId = state.review.reviewId;
  const generation = state.prefetch.generation;
  state.prefetch.reviewId = reviewId;
  const startIndex = getCurrentReviewIndex();
  const targetIndex = Math.min(state.review.total - 1, startIndex + REVIEW_PREFETCH_WINDOW);
  const loaded = await ensureQueueLoadedThroughIndex(targetIndex);
  if (!loaded || !prefetchContextIsActive(reviewId, generation)) {
    return;
  }
  for (let index = startIndex + 1; index <= targetIndex; index += 1) {
    const item = state.review.queue[index];
    if (!item) {
      continue;
    }
    const itemPairKey = pairKey(item.dataset_id, item.session_id);
    if (itemPairKey === state.review.currentPairKey) {
      continue;
    }
    await prefetchPair(item, itemPairKey, reviewId, generation);
  }
  await trimPrefetchCache();
}

async function prefetchPair(item, itemPairKey, reviewId, generation) {
  const existing = state.prefetch.entries.peek(itemPairKey);
  if (existing?.reviewId === reviewId && existing.status === "ready") {
    state.prefetch.entries.touch(itemPairKey);
    return existing.pair;
  }
  if (state.prefetch.inFlight.has(itemPairKey)) {
    return state.prefetch.inFlight.get(itemPairKey);
  }
  state.prefetch.entries.set(itemPairKey, {
    reviewId,
    status: "pending",
    pair: null,
    previewUrls: [],
    prefetchedAt: Date.now(),
  });
  const promise = (async () => {
    try {
      const response = await api.getReviewPair(reviewId, item.dataset_id, item.session_id);
      const rawPair = response.data;
      if (!prefetchContextIsActive(reviewId, generation)) {
        return rawPair;
      }
      await storage.putPair(rawPair);
      const previewUrls = rawPair.images.map((image) => image.preview_url);
      await warmPreviewUrls(previewUrls);
      if (!prefetchContextIsActive(reviewId, generation)) {
        return rawPair;
      }
      state.prefetch.entries.set(itemPairKey, {
        reviewId,
        status: "ready",
        pair: rawPair,
        previewUrls,
        prefetchedAt: Date.now(),
      });
      await trimPrefetchCache();
      return rawPair;
    } catch (error) {
      state.prefetch.entries.delete(itemPairKey);
      if (!(error instanceof HttpError && error.status === 404)) {
        console.error(error);
      }
      return null;
    } finally {
      if (state.prefetch.inFlight.get(itemPairKey) === promise) {
        state.prefetch.inFlight.delete(itemPairKey);
      }
    }
  })();
  state.prefetch.inFlight.set(itemPairKey, promise);
  return promise;
}

async function trimPrefetchCache() {
  const protectedKeys = new Set();
  if (state.review.currentPairKey) {
    protectedKeys.add(state.review.currentPairKey);
  }
  const currentIndex = getCurrentReviewIndex();
  for (let index = currentIndex + 1; index <= Math.min(state.review.total - 1, currentIndex + REVIEW_PREFETCH_WINDOW); index += 1) {
    const item = state.review.queue[index];
    if (item) {
      protectedKeys.add(pairKey(item.dataset_id, item.session_id));
    }
  }
  const removed = state.prefetch.entries.evictToFit({ protectKeys: [...protectedKeys] });
  await cleanupPrefetchEntries(removed);
}

async function clearPrefetchState() {
  state.prefetch.generation += 1;
  state.prefetch.reviewId = null;
  const removed = state.prefetch.entries.clear();
  state.prefetch.inFlight = new Map();
  state.prefetch.warming = new Map();
  await cleanupPrefetchEntries(removed);
}

async function cleanupPrefetchEntries(entries) {
  if (!entries?.length) {
    return;
  }
  await Promise.all(entries.map((entry) => storage.deletePair(entry.key)));
  await deletePreviewUrlsFromCaches(entries.flatMap((entry) => entry.value.previewUrls || []));
}

async function warmPreviewUrls(previewUrls) {
  await Promise.all(uniqueValues(previewUrls.map((url) => toAbsoluteUrl(url))).map((url) => warmPreviewUrl(url)));
}

async function warmPreviewUrl(url) {
  const existing = state.prefetch.warming.get(url);
  if (existing) {
    return existing;
  }
  const promise = (async () => {
    try {
      await fetchWithTimeout(
        url,
        {
          credentials: "same-origin",
          cache: "no-store",
        },
        MEDIA_REQUEST_TIMEOUT_MS
      );
    } catch (error) {
      console.error(error);
    } finally {
      if (state.prefetch.warming.get(url) === promise) {
        state.prefetch.warming.delete(url);
      }
    }
  })();
  state.prefetch.warming.set(url, promise);
  return promise;
}

async function deletePreviewUrlsFromCaches(previewUrls) {
  if (!previewUrls?.length || !("caches" in globalThis)) {
    return;
  }
  const absoluteUrls = uniqueValues(previewUrls.map((url) => toAbsoluteUrl(url)));
  const cacheNames = (await caches.keys()).filter((name) => name.startsWith(DATA_CACHE_PREFIX));
  await Promise.all(
    cacheNames.map(async (cacheName) => {
      const cache = await caches.open(cacheName);
      await Promise.all(absoluteUrls.map((url) => cache.delete(url)));
    })
  );
}

function toAbsoluteUrl(url) {
  return new URL(url, window.location.origin).href;
}

function renderShell() {
  elements.statusPill.textContent = state.syncing ? "Syncing" : state.syncFailed ? "Sync issue" : state.online ? "Online" : "Offline";
  elements.statusPill.classList.toggle("syncing", state.syncing);
  elements.statusPill.classList.toggle("offline", !state.online && !state.syncFailed);
  elements.statusPill.classList.toggle("error", state.syncFailed);
  elements.pendingCount.textContent = String(state.pendingCount);
  elements.progressText.textContent = state.review.reviewId ? `${state.review.total} remaining` : "No active review";
  elements.reviewerName.textContent = state.session ? state.session.reviewer_username : "Signed out";
  elements.navTasks.disabled = !state.session;
  elements.navReview.disabled = !state.session || !state.review.reviewId;
  elements.navExport.disabled = !state.session;
  elements.logoutButton.disabled = !state.session;
  elements.navTasks.classList.toggle("active", state.currentView === "tasks");
  elements.navReview.classList.toggle("active", state.currentView === "review");
  elements.navExport.classList.toggle("active", state.currentView === "export");

  elements.authView.hidden = state.currentView !== "auth";
  elements.tasksView.hidden = state.currentView !== "tasks";
  elements.reviewView.hidden = state.currentView !== "review";
  elements.exportView.hidden = state.currentView !== "export";

  if (state.currentView === "tasks") {
    renderCatalog();
  } else if (state.currentView === "review") {
    renderReview();
  } else if (state.currentView === "export") {
    renderExportBuilder();
  }
}

function renderAuthMessage(message) {
  elements.authMessage.textContent = message || "";
}

function renderCatalog() {
  if (!state.catalog) {
    elements.taskGroups.innerHTML = "<p class='empty-state'>Catalog not available yet.</p>";
    elements.startReview.disabled = true;
    elements.selectAllTasks.disabled = true;
    elements.clearTaskSelection.disabled = true;
    return;
  }

  const allTaskKeys = collectAllTaskKeys();
  const availableTaskKeys = new Set(allTaskKeys);
  state.review.taskKeys = state.review.taskKeys.filter((taskKey) => availableTaskKeys.has(taskKey));
  const selected = new Set(state.review.taskKeys);
  let selectedPairCount = 0;
  let warningCount = state.catalog.warnings?.length || 0;
  elements.taskGroups.innerHTML = "";
  const fragment = document.createDocumentFragment();

  for (const dataset of state.catalog.datasets) {
    const section = document.createElement("section");
    section.className = "dataset-card card";
    const progress = datasetProgress(dataset);

    const header = document.createElement("div");
    header.className = "dataset-header";
    header.innerHTML = `
      <div>
        <p class="eyebrow">Dataset</p>
        <h2>${escapeHtml(dataset.display_name)}</h2>
      </div>
      <div class="dataset-progress">
        <div class="dataset-progress-meta">
          <span>${progress.reviewedPairs} / ${progress.totalPairs} reviewed</span>
          <strong>${progress.reviewedPercent}%</strong>
        </div>
        <div class="dataset-progress-bar" aria-hidden="true">
          <span class="dataset-progress-fill" style="width: ${progress.reviewedPercent}%"></span>
        </div>
      </div>
    `;
    section.appendChild(header);

    const taskList = document.createElement("div");
    taskList.className = "task-list";

    for (const task of dataset.tasks) {
      if (selected.has(task.task_key)) {
        selectedPairCount += task.total_pairs;
      }
      const label = document.createElement("label");
      label.className = "task-item";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = selected.has(task.task_key);
      checkbox.addEventListener("change", () => {
        state.review.taskKeys = updateSelection(state.review.taskKeys, task.task_key, checkbox.checked);
        renderCatalog();
      });
      const meta = document.createElement("div");
      meta.className = "task-meta";
      const labels = deriveTaskDisplayLabels(dataset.display_name, task);
      meta.innerHTML = `
        <strong class="task-title">${escapeHtml(labels.title)}</strong>
        <span class="task-subtitle">${escapeHtml(labels.subtitle)}</span>
        <span class="task-stats">${task.reviewed_percent}% reviewed · ${task.reviewed_pairs} / ${task.total_pairs} done · ${task.invalid_pair_count} invalid</span>
      `;
      label.append(checkbox, meta);
      taskList.appendChild(label);
    }

    section.appendChild(taskList);
    fragment.appendChild(section);
  }

  elements.taskGroups.appendChild(fragment);
  elements.selectedTaskCount.textContent = String(state.review.taskKeys.length);
  elements.selectedPairCount.textContent = String(selectedPairCount);
  elements.catalogWarnings.textContent = warningCount ? `${warningCount} invalid session rows were quarantined from review.` : "No catalog warnings.";
  elements.startReview.disabled = state.review.taskKeys.length === 0;
  elements.selectAllTasks.disabled = allTaskKeys.length === 0 || state.review.taskKeys.length === allTaskKeys.length;
  elements.clearTaskSelection.disabled = state.review.taskKeys.length === 0;
}

async function handleStartReview() {
  if (!state.review.taskKeys.length) {
    return;
  }
  await clearPrefetchState();
  const mode = elements.reviewMode.value;
  const response = await api.createReviewSession({
    task_keys: state.review.taskKeys,
    mode,
  });
  state.review = {
    ...createEmptyReviewState(),
    reviewId: response.data.review.review_id,
    taskKeys: [...response.data.review.task_keys],
    mode: response.data.review.mode,
    total: response.data.queue_total,
    nextCursor: 0,
    firstUnlabeledIndex: response.data.first_unlabeled_index,
  };
  await persistReviewState();
  await loadQueuePage(0, { replace: true });
  await loadPairAtIndex(0);
  switchView("review");
}

async function loadQueuePage(cursor, { replace = false } = {}) {
  if (!state.review.reviewId) {
    return false;
  }
  try {
    const response = await api.getReviewQueue(state.review.reviewId, cursor, REVIEW_QUEUE_PAGE_SIZE);
    state.review.queue = replace || cursor === 0 ? response.data.items : mergeQueueItems(state.review.queue, response.data.items);
    state.review.total = response.data.total;
    state.review.nextCursor = response.data.next_cursor;
    state.review.firstUnlabeledIndex = response.data.first_unlabeled_index;
    await persistReviewState();
    return true;
  } catch (error) {
    if (error instanceof HttpError && error.status === 404) {
      await clearReviewState({ preserveTaskKeys: true });
      switchView("tasks");
      return false;
    }
    const cached = await storage.getMeta("review");
    if (cached?.queue) {
      state.review = {
        ...state.review,
        ...normalizeCachedReviewState(cached),
      };
      return true;
    }
    throw error;
  }
}

async function loadPairAtIndex(index) {
  if (!state.review.reviewId) {
    return false;
  }
  if (index >= state.review.total) {
    state.review.currentPairKey = null;
    state.review.currentPair = null;
    await persistReviewState();
    renderReview();
    return false;
  }
  if (index >= state.review.queue.length && state.review.nextCursor !== null) {
    const loaded = await loadQueuePage(state.review.queue.length);
    if (!loaded || !state.review.reviewId) {
      renderReview();
      return false;
    }
  }
  const item = state.review.queue[index];
  if (!item) {
    state.review.currentPairKey = null;
    state.review.currentPair = null;
    await persistReviewState();
    renderReview();
    return false;
  }
  const itemPairKey = pairKey(item.dataset_id, item.session_id);

  try {
    const response = await resolvePairForDisplay(item, itemPairKey);
    const preferredDisplayOrder = state.review.currentPair
      && state.review.currentPair.pair_key === itemPairKey
      ? state.review.currentPair.display_order
      : null;
    state.review.currentPairKey = itemPairKey;
    state.review.currentPair = randomizePairDisplay(response, preferredDisplayOrder);
    clearShareMessage();
    await storage.putPair(response);
    await persistReviewState();
  } catch (error) {
    if (error instanceof HttpError && error.status === 404) {
      const refreshed = await loadQueuePage(0, { replace: true });
      if (refreshed && state.review.reviewId) {
        return loadPairAtIndex(index);
      } else {
        renderReview();
      }
      return false;
    }
    const cachedPair = await storage.getPair(itemPairKey);
    if (!cachedPair) {
      throw error;
    }
    state.review.currentPairKey = itemPairKey;
    state.review.currentPair = randomizePairDisplay(cachedPair);
    clearShareMessage();
  }
  renderReview();
  void prefetchUpcomingPairs();
  return true;
}

function renderReview() {
  const hasActivePair = Boolean(state.review.reviewId && state.review.currentPair);
  const currentIndex = getCurrentReviewIndex();
  elements.reviewModeChip.textContent = state.review.mode === "random" ? "Random mode" : "Sequence mode";
  elements.reviewSubsetSummary.textContent = state.review.taskKeys.length
    ? `${state.review.taskKeys.length} task${state.review.taskKeys.length === 1 ? "" : "s"} selected`
    : "No tasks selected";
  const controlsDisabled = !hasActivePair || state.review.submitting;
  elements.pairNote.disabled = controlsDisabled;
  elements.nextPair.disabled = controlsDisabled || currentIndex >= state.review.total - 1;
  document.querySelectorAll("[data-decision]").forEach((button) => {
    button.disabled = controlsDisabled;
  });
  const shareDisabled = controlsDisabled || Boolean(state.shareState.pendingSlot);
  elements.sharePanel.hidden = !hasActivePair;
  elements.shareA.disabled = shareDisabled;
  elements.shareB.disabled = shareDisabled;
  elements.shareMessage.hidden = !state.shareState.message;
  elements.shareMessage.textContent = state.shareState.message;
  elements.shareMessage.style.color = state.shareState.isError ? "var(--danger)" : "var(--muted)";

  if (!state.review.reviewId || !state.review.currentPair) {
    elements.pairHeadline.textContent = state.review.reviewId ? "Queue complete" : "Select tasks to begin";
    elements.pairSubheadline.textContent = state.review.reviewId
      ? "All currently available pairs in this subset have been reviewed."
      : "Open the task page, choose 1..N tasks, and start sequence or random review.";
    elements.pairSummary.innerHTML = "";
    elements.pairLatestLabel.innerHTML = "";
    elements.defectsA.innerHTML = "";
    elements.defectsB.innerHTML = "";
    elements.mobileReviewStack.innerHTML = state.review.reviewId
      ? "<section class='card panel'><p class='empty-state'>No more loaded pairs are waiting in this review subset.</p></section>"
      : "<section class='card panel'><p class='empty-state'>Image cards will appear here after you start a review session.</p></section>";
    elements.pairImages.innerHTML = state.review.reviewId
      ? "<section class='card panel'><p class='empty-state'>No more loaded pairs are waiting in this review subset.</p></section>"
      : "<section class='card panel'><p class='empty-state'>Image cards will appear here after you start a review session.</p></section>";
    elements.pairNote.value = "";
    state.shareState.pendingSlot = null;
    clearShareMessage();
    elements.sharePanel.hidden = true;
    return;
  }

  const pair = state.review.currentPair;
  const displayLabel = mapLatestLabelToDisplay(pair.latest_label, pair.display_order);
  const subheadline = [`Pair ${Math.min(currentIndex + 1, state.review.total)} of ${state.review.total}`].join(" · ");
  elements.pairHeadline.textContent = pair.task_name;
  elements.pairSubheadline.textContent = subheadline;
  renderSummaryRows(elements.pairSummary, buildPairSummaryRows(pair));
  renderSummaryRows(elements.pairLatestLabel, buildLatestLabelRows(displayLabel));
  elements.pairNote.value = displayLabel?.note ?? "";
  elements.pairImages.innerHTML = "";
  elements.mobileReviewStack.innerHTML = "";
  renderDefects(elements.defectsA, "a", displayLabel);
  renderDefects(elements.defectsB, "b", displayLabel);

  for (const [slot, image] of pair.images.entries()) {
    const node = elements.imageTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector("[data-slot='label']").textContent = slot === 0 ? "A" : "B";
    const imageTag = node.querySelector("[data-slot='image']");
    imageTag.loading = "lazy";
    imageTag.src = image.preview_url;
    imageTag.alt = `${slot === 0 ? "A" : "B"} preview`;
    node.querySelector("[data-slot='image-link']").addEventListener("click", () => openOriginalImage(image.original_url));
    renderSummaryRows(node.querySelector("[data-slot='facts']"), buildImageSummaryRows(image));
    renderPromptText(node.querySelector("[data-slot='positive-prompt']"), image.positive_prompt, image.positive_prompt_segments);
    elements.pairImages.appendChild(node);
  }

  renderMobileReviewStack(pair, displayLabel);
}

function renderSummaryRows(container, rows) {
  container.innerHTML = "";
  const fragment = document.createDocumentFragment();
  for (const row of rows) {
    const wrapper = document.createElement("div");
    wrapper.className = "summary-item";
    const term = document.createElement("dt");
    term.textContent = row.label;
    const description = document.createElement("dd");
    description.textContent = row.value;
    wrapper.append(term, description);
    fragment.appendChild(wrapper);
  }
  container.appendChild(fragment);
}

function renderDefects(container, side, latestLabel) {
  container.innerHTML = "";
  const selected = new Set(side === "a" ? latestLabel?.defects_a ?? [] : latestLabel?.defects_b ?? []);
  for (const defect of state.config?.defect_tags ?? []) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.dataset.side = side;
    input.dataset.defect = defect;
    input.checked = selected.has(defect);
    input.disabled = state.review.submitting || !state.review.currentPair;
    const text = document.createElement("span");
    text.textContent = defect.replace(/_/g, " ");
    label.append(input, text);
    container.appendChild(label);
  }
}

async function handleDecision(decision) {
  if (!state.review.currentPair || !state.session || state.review.submitting) {
    return;
  }
  const currentPairKey = pairKey(state.review.currentPair.dataset_id, state.review.currentPair.session_id);
  const currentIndex = getCurrentReviewIndex();
  const currentPreviewUrls = state.review.currentPair.images.map((image) => image.preview_url);
  const labelPayload = buildLabelPayloadFromDisplay(
    state.review.currentPair,
    decision,
    getCheckedDefects("a"),
    getCheckedDefects("b"),
  );
  const event = {
    event_id: createUuid(),
    review_id: state.review.reviewId,
    dataset_id: state.review.currentPair.dataset_id,
    session_id: state.review.currentPair.session_id,
    ...labelPayload,
    note: elements.pairNote.value.trim(),
    created_at: new Date().toISOString(),
    client_ts: new Date().toISOString(),
    app_version: state.config.app_version,
    reviewer_username: state.session.reviewer_username,
  };

  state.review.submitting = true;
  window.scrollTo(0, 0);
  renderReview();
  renderShell();

  try {
    await api.submitLabelEvent(event);
  } catch (error) {
    await storage.putPendingEvent(event);
    await refreshPendingCount();
    if (state.online) {
      requestSync();
    }
  }

  try {
    state.review.queue = state.review.queue.filter((item) => pairKey(item.dataset_id, item.session_id) !== currentPairKey);
    state.review.total = Math.max(state.review.total - 1, 0);
    state.review.nextCursor = state.review.nextCursor === null ? null : Math.max(state.review.nextCursor - 1, 0);
    const loaded = await loadPairAtIndex(currentIndex);
    if (!loaded) {
      state.review.currentPairKey = null;
      state.review.currentPair = null;
      await persistReviewState();
      renderReview();
    }
  } catch (error) {
    console.error(error);
    state.review.currentPairKey = null;
    state.review.currentPair = null;
    await persistReviewState();
    renderReview();
  } finally {
    await storage.deletePair(currentPairKey);
    await deletePreviewUrlsFromCaches(currentPreviewUrls);
    state.review.submitting = false;
    await persistReviewState();
    renderReview();
    renderShell();
  }
}

function renderMobileReviewStack(pair, displayLabel) {
  elements.mobileReviewStack.innerHTML = "";
  for (const [slot, image] of pair.images.entries()) {
    const side = slot === 0 ? "a" : "b";
    const label = side.toUpperCase();
    elements.mobileReviewStack.appendChild(buildMobileInfoCard(label, image));
    elements.mobileReviewStack.appendChild(buildMobileImageCard(label, image));
    elements.mobileReviewStack.appendChild(buildMobileDefectCard(label, side, displayLabel));
  }
}

function buildMobileInfoCard(label, image) {
  const card = document.createElement("article");
  card.className = "card mobile-review-card mobile-info-card";

  const eyebrow = document.createElement("p");
  eyebrow.className = "eyebrow";
  eyebrow.textContent = `${label} Info`;

  const facts = document.createElement("p");
  facts.className = "mobile-info-facts";
  facts.textContent = buildCompactImageFacts(image);

  const promptLabel = document.createElement("p");
  promptLabel.className = "eyebrow section-eyebrow";
  promptLabel.textContent = "Prompt";

  const prompt = document.createElement("p");
  prompt.className = "prompt-text";
  renderPromptText(prompt, image.positive_prompt, image.positive_prompt_segments);

  card.append(eyebrow, facts, promptLabel, prompt);
  return card;
}

function buildMobileImageCard(label, image) {
  const card = document.createElement("article");
  card.className = "card mobile-review-card mobile-image-card";

  const eyebrow = document.createElement("p");
  eyebrow.className = "eyebrow";
  eyebrow.textContent = `${label} Image`;

  const button = document.createElement("button");
  button.type = "button";
  button.className = "image-frame";
  button.addEventListener("click", () => openOriginalImage(image.original_url));

  const imageTag = document.createElement("img");
  imageTag.loading = "lazy";
  imageTag.src = image.preview_url;
  imageTag.alt = `${label} preview`;
  button.appendChild(imageTag);

  card.append(eyebrow, button);
  return card;
}

function buildMobileDefectCard(label, side, displayLabel) {
  const card = document.createElement("section");
  card.className = "defect-editor mobile-review-card";

  const eyebrow = document.createElement("p");
  eyebrow.className = "eyebrow section-eyebrow";
  eyebrow.textContent = `${label} Defects`;

  const list = document.createElement("div");
  list.className = "defect-list";
  renderDefects(list, side, displayLabel);

  card.append(eyebrow, list);
  return card;
}

function openOriginalImage(url) {
  window.open(url, "_blank", "noopener");
}

function clearShareMessage() {
  state.shareState.message = "";
  state.shareState.isError = false;
}

function setShareMessage(message, { isError = false } = {}) {
  state.shareState.message = message;
  state.shareState.isError = isError;
  renderReview();
}

function buildShareText(image) {
  const prefix = "#AIGenerated";
  const prompt = String(image.positive_prompt || "").normalize("NFC").trim();
  if (!prompt) {
    return truncateShareText(prefix, X_TEXT_LIMIT);
  }
  const divider = "\n\n";
  const reservedWeight = getXWeightedLength(`${prefix}${divider}`);
  const remainingWeight = Math.max(0, X_TEXT_LIMIT - reservedWeight);
  const trimmedPrompt = truncatePromptByWeight(prompt, remainingWeight);
  return trimmedPrompt ? `${prefix}${divider}${trimmedPrompt}` : prefix;
}

function truncateShareText(text, maxLength) {
  const chars = Array.from(String(text || ""));
  if (chars.length <= maxLength) {
    return chars.join("");
  }

  const limitedChars = chars.slice(0, maxLength);
  const nextChar = chars[maxLength] || "";
  const lastChar = limitedChars[limitedChars.length - 1] || "";
  if (!/\s/u.test(lastChar) && nextChar && !/\s/u.test(nextChar)) {
    for (let index = limitedChars.length - 1; index >= 0; index -= 1) {
      if (/\s/u.test(limitedChars[index])) {
        const rounded = limitedChars.slice(0, index).join("").trimEnd();
        if (rounded) {
          return rounded;
        }
        break;
      }
    }
  }
  return limitedChars.join("").trimEnd();
}

function truncatePromptByWeight(text, maxWeight) {
  if (maxWeight <= 0) {
    return "";
  }
  const parts = String(text || "").split(/(\s+)/u);
  let output = "";
  let outputWeight = 0;

  for (const part of parts) {
    if (!part) {
      continue;
    }
    const partWeight = getXWeightedLength(part);
    if (outputWeight + partWeight <= maxWeight) {
      output += part;
      outputWeight += partWeight;
      continue;
    }
    if (/^\s+$/u.test(part)) {
      break;
    }
    if (!output.trim()) {
      output = truncateTextByWeight(part, maxWeight);
    }
    break;
  }
  return output.trimEnd();
}

function truncateTextByWeight(text, maxWeight) {
  let output = "";
  let outputWeight = 0;
  for (const segment of segmentGraphemes(text)) {
    const weight = getWeightedGraphemeLength(segment);
    if (outputWeight + weight > maxWeight) {
      break;
    }
    output += segment;
    outputWeight += weight;
  }
  return output.trimEnd();
}

function getXWeightedLength(text) {
  if (!text) {
    return 0;
  }
  const normalized = String(text).normalize("NFC");
  let total = 0;
  for (const token of normalized.split(/(\s+)/u)) {
    if (!token) {
      continue;
    }
    if (!/^\s+$/u.test(token) && /^https?:\/\//iu.test(token)) {
      total += 23;
      continue;
    }
    for (const segment of segmentGraphemes(token)) {
      total += getWeightedGraphemeLength(segment);
    }
  }
  return total;
}

function segmentGraphemes(text) {
  if (globalThis.Intl && typeof Intl.Segmenter === "function") {
    const segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });
    return [...segmenter.segment(text)].map((part) => part.segment);
  }
  return Array.from(text);
}

function getWeightedGraphemeLength(segment) {
  if (!segment) {
    return 0;
  }
  if (/^\s+$/u.test(segment)) {
    return Array.from(segment).length;
  }
  if (/[\p{Extended_Pictographic}]/u.test(segment)) {
    return 2;
  }
  if (/^[\x00-\x7F]+$/u.test(segment)) {
    return Array.from(segment).length;
  }
  return 2;
}

function buildShareFilename(pair, side, blobType) {
  const extension = blobType === "image/png" ? "png" : "jpg";
  return `${pair.session_id || "pair"}_${side}.${extension}`;
}

function buildTwitterIntentUrl(image) {
  return `https://twitter.com/intent/tweet?text=${encodeURIComponent(buildShareText(image))}`;
}

function triggerDownload(blob, filename) {
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename;
  link.rel = "noopener";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}

async function handleShareImage(side) {
  if (!state.review.currentPair || state.review.submitting || state.shareState.pendingSlot) {
    return;
  }
  const slotIndex = side === "a" ? 0 : 1;
  const image = state.review.currentPair.images[slotIndex];
  if (!image) {
    return;
  }

  state.shareState.pendingSlot = side;
  clearShareMessage();
  renderReview();
  const composerWindow = window.open("", "_blank");
  try {
    const previewResponse = await fetchWithTimeout(
      image.preview_url,
      {
        credentials: "same-origin",
        cache: "no-store",
      },
      MEDIA_REQUEST_TIMEOUT_MS
    );
    if (!previewResponse.ok) {
      throw new Error(`Preview fetch failed: ${previewResponse.status}`);
    }
    const blob = await previewResponse.blob();
    triggerDownload(blob, buildShareFilename(state.review.currentPair, side, blob.type));
    if (composerWindow) {
      try {
        composerWindow.opener = null;
      } catch (error) {
        console.error(error);
      }
      composerWindow.location.replace(buildTwitterIntentUrl(image));
      setShareMessage("Image downloaded. Attach it in X/Twitter after the composer opens.");
    } else {
      setShareMessage("Image downloaded. Open X/Twitter manually and attach the file.");
    }
  } catch (error) {
    if (composerWindow && !composerWindow.closed) {
      composerWindow.close();
    }
    console.error(error);
    setShareMessage("Unable to download image for sharing.", { isError: true });
    return;
  } finally {
    state.shareState.pendingSlot = null;
    renderReview();
  }
}

function getCheckedDefects(side) {
  return [...document.querySelectorAll(`input[data-side='${side}']:checked`)].map((input) => input.dataset.defect);
}

async function flushPendingEvents() {
  const pending = await storage.getAllPendingEvents();
  for (const event of pending) {
    try {
      await api.submitLabelEvent(event);
      await storage.deletePendingEvent(event.event_id);
    } catch (error) {
      if (error instanceof HttpError && error.status === 401) {
        state.session = null;
        switchView("auth");
      }
      throw error;
    }
  }
  await refreshPendingCount();
}

async function refreshPendingCount() {
  state.pendingCount = (await storage.getAllPendingEvents()).length;
}

function renderExportBuilder() {
  if (!state.config || !state.catalog) {
    elements.filterRoot.innerHTML = "<p class='empty-state'>Sign in and load the catalog to build exports.</p>";
    return;
  }
  renderFilterNode(state.exportState.filter, elements.filterRoot, true);
}

function renderFilterNode(node, container, isRoot = false) {
  container.innerHTML = "";
  const wrapper = document.createElement("div");
  wrapper.className = "filter-group";

  const header = document.createElement("div");
  header.className = "filter-group-header";
  const operator = document.createElement("select");
  operator.innerHTML = `<option value="and">AND</option><option value="or">OR</option>`;
  operator.value = node.operator;
  operator.addEventListener("change", () => {
    node.operator = operator.value;
  });
  header.append(operator);

  const addRule = document.createElement("button");
  addRule.type = "button";
  addRule.className = "nav-btn";
  addRule.textContent = "Add Rule";
  addRule.addEventListener("click", () => {
    node.conditions.push(createRule());
    renderExportBuilder();
  });
  header.append(addRule);

  const addGroup = document.createElement("button");
  addGroup.type = "button";
  addGroup.className = "nav-btn";
  addGroup.textContent = "Add Group";
  addGroup.addEventListener("click", () => {
    node.conditions.push(buildEmptyFilter());
    renderExportBuilder();
  });
  header.append(addGroup);

  if (!isRoot) {
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "nav-btn ghost";
    remove.textContent = "Remove Group";
    remove.addEventListener("click", () => {
      removeNode(state.exportState.filter, node);
      renderExportBuilder();
    });
    header.append(remove);
  }

  wrapper.append(header);
  const body = document.createElement("div");
  body.className = "filter-group-body";
  for (const condition of node.conditions) {
    if (condition.type === "group") {
      const nested = document.createElement("div");
      renderFilterNode(condition, nested, false);
      body.appendChild(nested);
      continue;
    }
    body.appendChild(renderRule(condition));
  }
  wrapper.append(body);
  container.appendChild(wrapper);
}

function createRule() {
  return {
    type: "rule",
    field: "dataset_id",
    operator: "eq",
    value: "",
  };
}

function renderRule(rule) {
  const row = document.createElement("div");
  row.className = "filter-rule";

  const fieldSelect = document.createElement("select");
  for (const field of state.config.filter_schema.fields) {
    const option = document.createElement("option");
    option.value = field.field;
    option.textContent = field.label;
    fieldSelect.appendChild(option);
  }
  fieldSelect.value = rule.field;
  fieldSelect.addEventListener("change", () => {
    rule.field = fieldSelect.value;
    const metadata = getFieldMetadata(rule.field);
    rule.operator = metadata.operators[0];
    rule.value = defaultRuleValue(metadata, rule.operator);
    renderExportBuilder();
  });
  row.appendChild(fieldSelect);

  const operatorSelect = document.createElement("select");
  const metadata = getFieldMetadata(rule.field);
  for (const operator of metadata.operators) {
    const option = document.createElement("option");
    option.value = operator;
    option.textContent = operator;
    operatorSelect.appendChild(option);
  }
  operatorSelect.value = rule.operator;
  operatorSelect.addEventListener("change", () => {
    rule.operator = operatorSelect.value;
    rule.value = defaultRuleValue(metadata, rule.operator);
    renderExportBuilder();
  });
  row.appendChild(operatorSelect);

  row.appendChild(renderRuleValueEditor(rule, metadata));

  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "nav-btn ghost";
  remove.textContent = "Remove";
  remove.addEventListener("click", () => {
    removeNode(state.exportState.filter, rule);
    renderExportBuilder();
  });
  row.appendChild(remove);
  return row;
}

function renderRuleValueEditor(rule, metadata) {
  if (metadata.value_kind === "boolean") {
    const select = document.createElement("select");
    select.innerHTML = `<option value="true">true</option><option value="false">false</option>`;
    select.value = String(Boolean(rule.value));
    select.addEventListener("change", () => {
      rule.value = select.value === "true";
    });
    return select;
  }

  if (metadata.value_kind === "defects") {
    const wrapper = document.createElement("div");
    wrapper.className = "checkbox-grid";
    const selected = new Set(Array.isArray(rule.value) ? rule.value : []);
    for (const defect of state.config.defect_tags) {
      const label = document.createElement("label");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = selected.has(defect);
      checkbox.addEventListener("change", () => {
        const next = new Set(Array.isArray(rule.value) ? rule.value : []);
        if (checkbox.checked) {
          next.add(defect);
        } else {
          next.delete(defect);
        }
        rule.value = [...next];
      });
      const text = document.createElement("span");
      text.textContent = defect.replace(/_/g, " ");
      label.append(checkbox, text);
      wrapper.appendChild(label);
    }
    return wrapper;
  }

  if (metadata.field === "decision") {
    return renderMultiOptionInput(rule, state.config.decisions);
  }

  if (metadata.field === "dataset_id") {
    return renderMultiOptionInput(rule, state.catalog.datasets.map((dataset) => dataset.dataset_id));
  }

  if (metadata.field === "task_key") {
    return renderMultiOptionInput(
      rule,
      state.catalog.datasets.flatMap((dataset) => dataset.tasks.map((task) => task.task_key)),
      (value) => taskDisplayName(value)
    );
  }

  if (metadata.field === "reviewer_username") {
    return renderMultiOptionInput(rule, collectReviewerOptions());
  }

  if (metadata.field === "task_name") {
    return renderMultiOptionInput(rule, uniqueValues(state.catalog.datasets.flatMap((dataset) => dataset.tasks.map((task) => task.task_name))));
  }

  if (metadata.field === "task_yaml_name") {
    return renderMultiOptionInput(rule, uniqueValues(state.catalog.datasets.flatMap((dataset) => dataset.tasks.map((task) => task.task_yaml_name))));
  }

  if (metadata.value_kind === "datetime") {
    if (rule.operator === "between") {
      const wrapper = document.createElement("div");
      wrapper.className = "datetime-range";
      const start = document.createElement("input");
      start.type = "datetime-local";
      start.value = isoToLocal(rule.value?.start);
      start.addEventListener("change", () => {
        rule.value = {
          start: localToIso(start.value),
          end: localToIso(end.value),
        };
      });
      const end = document.createElement("input");
      end.type = "datetime-local";
      end.value = isoToLocal(rule.value?.end);
      end.addEventListener("change", () => {
        rule.value = {
          start: localToIso(start.value),
          end: localToIso(end.value),
        };
      });
      wrapper.append(start, end);
      return wrapper;
    }
    const input = document.createElement("input");
    input.type = "datetime-local";
    input.value = isoToLocal(rule.value);
    input.addEventListener("change", () => {
      rule.value = localToIso(input.value);
    });
    return input;
  }

  const input = document.createElement("input");
  input.type = "text";
  input.value = rule.value ?? "";
  input.placeholder = metadata.field === "note" ? "Search note text" : "Value";
  input.addEventListener("input", () => {
    rule.value = input.value;
  });
  return input;
}

function renderMultiOptionInput(rule, options, labelFormatter = (value) => value) {
  if (rule.operator === "eq") {
    const select = document.createElement("select");
    for (const value of options) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = labelFormatter(value);
      select.appendChild(option);
    }
    if (options.length && !options.includes(rule.value)) {
      rule.value = options[0];
    }
    select.value = rule.value ?? "";
    select.addEventListener("change", () => {
      rule.value = select.value;
    });
    return select;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "checkbox-grid";
  const selected = new Set(Array.isArray(rule.value) ? rule.value : []);
  for (const value of options) {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selected.has(value);
    checkbox.addEventListener("change", () => {
      const next = new Set(Array.isArray(rule.value) ? rule.value : []);
      if (checkbox.checked) {
        next.add(value);
      } else {
        next.delete(value);
      }
      rule.value = [...next];
    });
    const text = document.createElement("span");
    text.textContent = labelFormatter(value);
    label.append(checkbox, text);
    wrapper.appendChild(label);
  }
  return wrapper;
}

async function handleExportPreview() {
  try {
    const response = await api.previewExport({
      export_type: state.exportState.exportType,
      filter: cloneFilter(state.exportState.filter),
    });
    state.exportState.previewCount = response.data.count;
    elements.exportMessage.textContent = `${response.data.count} rows match the current filter.`;
  } catch (error) {
    elements.exportMessage.textContent = error instanceof HttpError ? error.message : String(error);
  }
}

async function handleExportDownload() {
  try {
    const { blob, filename } = await api.downloadExport({
      export_type: state.exportState.exportType,
      filter: cloneFilter(state.exportState.filter),
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
    elements.exportMessage.textContent = `Downloaded ${filename}.`;
  } catch (error) {
    elements.exportMessage.textContent = error instanceof HttpError ? error.message : String(error);
  }
}

function getFieldMetadata(fieldName) {
  return state.config.filter_schema.fields.find((field) => field.field === fieldName);
}

function defaultRuleValue(metadata, operator) {
  if (metadata.value_kind === "boolean") {
    return true;
  }
  if (metadata.value_kind === "datetime") {
    if (operator === "between") {
      return {
        start: new Date().toISOString(),
        end: new Date().toISOString(),
      };
    }
    return new Date().toISOString();
  }
  if (metadata.value_kind === "defects" || operator === "in") {
    return [];
  }
  return "";
}

function removeNode(root, target) {
  if (!root.conditions) {
    return false;
  }
  const nextConditions = [];
  let removed = false;
  for (const condition of root.conditions) {
    if (condition === target) {
      removed = true;
      continue;
    }
    if (condition.type === "group" && removeNode(condition, target)) {
      removed = true;
    }
    nextConditions.push(condition);
  }
  root.conditions = nextConditions;
  return removed;
}

function collectAllTaskKeys() {
  if (!state.catalog) {
    return [];
  }
  return state.catalog.datasets.flatMap((dataset) => dataset.tasks.map((task) => task.task_key));
}

function updateSelection(current, value, checked) {
  const next = new Set(current);
  if (checked) {
    next.add(value);
  } else {
    next.delete(value);
  }
  return [...next];
}

function collectReviewerOptions() {
  if (!state.catalog) {
    return [];
  }
  const names = [];
  for (const dataset of state.catalog.datasets) {
    for (const task of dataset.tasks) {
      for (const reviewer of Object.keys(task.reviewers || {})) {
        if (!names.includes(reviewer)) {
          names.push(reviewer);
        }
      }
    }
  }
  return names.sort();
}

function uniqueValues(values) {
  return [...new Set(values.filter(Boolean))].sort();
}

function taskDisplayName(taskKey) {
  if (!state.catalog) {
    return taskKey;
  }
  for (const dataset of state.catalog.datasets) {
    const task = dataset.tasks.find((item) => item.task_key === taskKey);
    if (task) {
      return `${task.task_yaml_name} · ${task.task_name}`;
    }
  }
  return taskKey;
}

function loadClientInstanceId() {
  let value = localStorage.getItem(CLIENT_INSTANCE_KEY);
  if (!value) {
    value = createUuid();
    localStorage.setItem(CLIENT_INSTANCE_KEY, value);
  }
  return value;
}

function createUuid() {
  const cryptoApi = globalThis.crypto;
  if (cryptoApi?.randomUUID) {
    return cryptoApi.randomUUID();
  }
  if (cryptoApi?.getRandomValues) {
    const bytes = new Uint8Array(16);
    cryptoApi.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20, 32)}`;
  }
  const seed = `${Date.now().toString(16)}${Math.floor(Math.random() * Number.MAX_SAFE_INTEGER).toString(16)}`.padEnd(32, "0").slice(0, 32);
  return `${seed.slice(0, 8)}-${seed.slice(8, 12)}-4${seed.slice(13, 16)}-a${seed.slice(17, 20)}-${seed.slice(20, 32)}`;
}

function scheduleSyncRetry(request = {}) {
  const currentBackoff = state.syncBackoffMs || 5_000;
  state.syncBackoffMs = Math.min(currentBackoff * 2, 30_000);
  state.nextSyncAllowedAt = Date.now() + currentBackoff;
  requestSync({
    refreshCatalog: Boolean(request.refreshCatalog),
    refreshReview: Boolean(request.refreshReview),
  });
}

function isoToLocal(value) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  const offset = date.getTimezoneOffset();
  const local = new Date(date.getTime() - offset * 60_000);
  return local.toISOString().slice(0, 16);
}

function localToIso(value) {
  return value ? new Date(value).toISOString() : "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderPromptText(container, text, segments) {
  container.textContent = "";
  const fallback = formatPromptText(text, "No prompt");
  const parts = Array.isArray(segments) && segments.length
    ? segments
    : [{ text: fallback, bold: false }];
  for (const part of parts) {
    const content = String(part?.text ?? "");
    if (!content) {
      continue;
    }
    if (part.bold) {
      const strong = document.createElement("strong");
      strong.textContent = content;
      container.appendChild(strong);
      continue;
    }
    container.append(document.createTextNode(content));
  }
}
