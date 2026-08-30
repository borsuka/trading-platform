/**
 * API client.
 *
 * Two things this file is careful about:
 *
 * 1. **The trading mode.** Every response carries `X-Trading-Mode`. It is captured on every
 *    call so the LIVE/PAPER banner can never go stale — a user must never be unsure which mode
 *    they are looking at.
 * 2. **Token storage.** Tokens live in `sessionStorage`, not `localStorage`, so they do not
 *    survive the tab closing. A trading dashboard left open on a shared machine is a real
 *    risk, and this bounds it.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const ACCESS_KEY = "tp_access_token";
const REFRESH_KEY = "tp_refresh_token";

export type TradingMode = "PAPER" | "LIVE" | "BACKTEST";

let cachedMode: TradingMode = "PAPER";
let cachedIsLive = false;
const modeListeners = new Set<(mode: TradingMode, isLive: boolean) => void>();

export function currentMode(): { mode: TradingMode; isLive: boolean } {
  return { mode: cachedMode, isLive: cachedIsLive };
}

export function onModeChange(
  listener: (mode: TradingMode, isLive: boolean) => void,
): () => void {
  modeListeners.add(listener);
  return () => modeListeners.delete(listener);
}

function recordMode(response: Response): void {
  const mode = response.headers.get("X-Trading-Mode") as TradingMode | null;
  const isLive = response.headers.get("X-Live-Trading") === "true";
  if (mode && (mode !== cachedMode || isLive !== cachedIsLive)) {
    cachedMode = mode;
    cachedIsLive = isLive;
    modeListeners.forEach((listener) => listener(mode, isLive));
  }
}

/** An error carrying the API's structured detail, so the UI can show something useful. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly context: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function readToken(key: string): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(key);
}

export function setTokens(access: string, refresh: string): void {
  window.sessionStorage.setItem(ACCESS_KEY, access);
  window.sessionStorage.setItem(REFRESH_KEY, refresh);
}

export function clearTokens(): void {
  if (typeof window === "undefined") return;
  window.sessionStorage.removeItem(ACCESS_KEY);
  window.sessionStorage.removeItem(REFRESH_KEY);
}

export function isAuthenticated(): boolean {
  return readToken(ACCESS_KEY) !== null;
}

let refreshInFlight: Promise<boolean> | null = null;

/** Exchange the refresh token for a new pair. Concurrent callers share one request. */
async function refreshTokens(): Promise<boolean> {
  if (refreshInFlight) return refreshInFlight;

  const refresh = readToken(REFRESH_KEY);
  if (!refresh) return false;

  refreshInFlight = (async () => {
    try {
      const response = await fetch(`${API_URL}/api/v1/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refresh }),
      });
      if (!response.ok) {
        clearTokens();
        return false;
      }
      const body = await response.json();
      setTokens(body.access_token, body.refresh_token);
      return true;
    } catch {
      return false;
    } finally {
      refreshInFlight = null;
    }
  })();

  return refreshInFlight;
}

export async function api<T = unknown>(
  path: string,
  options: RequestInit & { retryOnAuthFailure?: boolean } = {},
): Promise<T> {
  const { retryOnAuthFailure = true, ...init } = options;
  const token = readToken(ACCESS_KEY);

  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(`${API_URL}${path}`, { ...init, headers });
  recordMode(response);

  // A 401 usually means the short-lived access token expired. Refresh once and retry;
  // retrying more than once would loop when the refresh token is also dead.
  if (response.status === 401 && retryOnAuthFailure && readToken(REFRESH_KEY)) {
    if (await refreshTokens()) {
      return api<T>(path, { ...options, retryOnAuthFailure: false });
    }
  }

  if (response.status === 204) return undefined as T;

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new ApiError(
      response.status,
      body.error ?? "error",
      body.message ?? `Request failed (${response.status})`,
      body.context ?? {},
    );
  }
  return body as T;
}

export const get = <T>(path: string) => api<T>(path);
export const post = <T>(path: string, body?: unknown) =>
  api<T>(path, { method: "POST", body: body ? JSON.stringify(body) : undefined });
export const put = <T>(path: string, body: unknown) =>
  api<T>(path, { method: "PUT", body: JSON.stringify(body) });
export const patch = <T>(path: string, body: unknown) =>
  api<T>(path, { method: "PATCH", body: JSON.stringify(body) });
export const del = <T>(path: string) => api<T>(path, { method: "DELETE" });

/** Fetch public build info, including the trading mode, before login. */
export async function fetchInfo(): Promise<{
  name: string;
  version: string;
  mode: TradingMode;
  is_live: boolean;
  environment: string;
  disclaimer: string;
}> {
  const response = await fetch(`${API_URL}/info`);
  recordMode(response);
  return response.json();
}

export async function login(email: string, password: string): Promise<void> {
  const response = await fetch(`${API_URL}/api/v1/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  recordMode(response);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new ApiError(response.status, body.error ?? "error", body.message ?? "Sign-in failed");
  }
  setTokens(body.access_token, body.refresh_token);
}

export async function register(
  email: string,
  password: string,
  fullName?: string,
): Promise<void> {
  const response = await fetch(`${API_URL}/api/v1/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, full_name: fullName || null }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new ApiError(
      response.status,
      body.error ?? "error",
      body.message ?? "Registration failed",
      body.context ?? {},
    );
  }
}

export async function logout(): Promise<void> {
  const refresh = readToken(REFRESH_KEY);
  if (refresh) {
    await post("/api/v1/auth/logout", { refresh_token: refresh }).catch(() => undefined);
  }
  clearTokens();
}
