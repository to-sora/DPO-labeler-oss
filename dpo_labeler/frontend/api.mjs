export class HttpError extends Error {
  constructor(message, status, payload = null) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.payload = payload;
  }
}

export class RequestTimeoutError extends Error {
  constructor(message, timeoutMs) {
    super(message);
    this.name = "RequestTimeoutError";
    this.timeoutMs = timeoutMs;
  }
}

export const JSON_REQUEST_TIMEOUT_MS = 15_000;
export const BLOB_REQUEST_TIMEOUT_MS = 60_000;
export const MEDIA_REQUEST_TIMEOUT_MS = 15_000;

async function parseJson(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }
  return JSON.parse(text);
}

export async function fetchWithTimeout(url, options = {}, timeoutMs = JSON_REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(new RequestTimeoutError(`${url} timed out`, timeoutMs)), timeoutMs);
  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new RequestTimeoutError(`${url} timed out`, timeoutMs);
    }
    throw error;
  } finally {
    window.clearTimeout(timer);
  }
}

async function requestJson(url, options = {}) {
  const response = await fetchWithTimeout(
    url,
    {
      credentials: "same-origin",
      cache: "no-store",
      ...options,
    },
    JSON_REQUEST_TIMEOUT_MS
  );
  const payload = await parseJson(response);
  if (!response.ok) {
    throw new HttpError(payload?.error || `${url} failed`, response.status, payload);
  }
  return payload;
}

async function requestBlob(url, body) {
  const response = await fetchWithTimeout(
    url,
    {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    BLOB_REQUEST_TIMEOUT_MS
  );
  if (!response.ok) {
    const payload = await parseJson(response);
    throw new HttpError(payload?.error || `${url} failed`, response.status, payload);
  }
  const blob = await response.blob();
  const disposition = response.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename="([^"]+)"/);
  return {
    blob,
    filename: match ? match[1] : "export.jsonl",
  };
}

export const api = {
  async getConfig() {
    return requestJson("/api/v1/config");
  },
  async getSession() {
    return requestJson("/api/v1/session/me");
  },
  async startSession(payload) {
    return requestJson("/api/v1/session/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },
  async endSession() {
    return requestJson("/api/v1/session/end", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
  },
  async getCatalog() {
    return requestJson("/api/v1/catalog");
  },
  async createReviewSession(payload) {
    return requestJson("/api/v1/review-sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },
  async getReviewQueue(reviewId, cursor = 0, limit = 24) {
    return requestJson(`/api/v1/review-sessions/${encodeURIComponent(reviewId)}/queue?cursor=${cursor}&limit=${limit}`);
  },
  async getReviewPair(reviewId, datasetId, sessionId) {
    return requestJson(
      `/api/v1/review-sessions/${encodeURIComponent(reviewId)}/pairs/${encodeURIComponent(datasetId)}/${encodeURIComponent(sessionId)}`
    );
  },
  async submitLabelEvent(payload) {
    return requestJson("/api/v1/label-events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },
  async previewExport(payload) {
    return requestJson("/api/v1/exports/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  },
  async downloadExport(payload) {
    return requestBlob("/api/v1/exports/download", payload);
  },
};
