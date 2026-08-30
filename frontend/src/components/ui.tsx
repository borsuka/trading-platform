"use client";

/**
 * Shared presentation components.
 *
 * `Money` and `Percent` always render an explicit sign. Colour alone is not an accessible way
 * to convey profit and loss, and a P&L figure is exactly the wrong place to rely on hue.
 */

import { useEffect, useState, type ReactNode } from "react";

import { ApiError } from "@/lib/api";

export function Money({
  value,
  currency = "",
  signed = true,
  decimals = 2,
}: {
  value: number | null | undefined;
  currency?: string;
  signed?: boolean;
  decimals?: number;
}) {
  if (value == null || Number.isNaN(value)) return <span className="muted">—</span>;
  const sign = signed && value !== 0 ? (value > 0 ? "+" : "−") : "";
  const cls = !signed ? "mono" : value > 0 ? "mono profit" : value < 0 ? "mono loss" : "mono";
  return (
    <span className={cls}>
      {sign}
      {currency}
      {Math.abs(value).toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })}
    </span>
  );
}

export function Percent({
  value,
  signed = true,
  decimals = 2,
}: {
  value: number | null | undefined;
  signed?: boolean;
  decimals?: number;
}) {
  if (value == null || Number.isNaN(value)) return <span className="muted">—</span>;
  const pct = value * 100;
  const sign = signed && pct !== 0 ? (pct > 0 ? "+" : "−") : "";
  const cls = !signed ? "mono" : pct > 0 ? "mono profit" : pct < 0 ? "mono loss" : "mono";
  return (
    <span className={cls}>
      {sign}
      {Math.abs(pct).toFixed(decimals)}%
    </span>
  );
}

export function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
}) {
  return (
    <div className="card">
      <div className="stat__label">{label}</div>
      <div className="stat__value">{value}</div>
      {hint ? <div className="stat__hint">{hint}</div> : null}
    </div>
  );
}

const STATUS_TONE: Record<string, string> = {
  running: "ok",
  active: "ok",
  filled: "ok",
  completed: "ok",
  ok: "ok",
  paused: "warn",
  pending: "warn",
  partially_filled: "warn",
  open: "warn",
  submitted: "warn",
  degraded: "warn",
  halted: "danger",
  error: "danger",
  rejected: "danger",
  failed: "danger",
  unhealthy: "danger",
  cancelled: "muted",
  stopped: "muted",
  created: "muted",
  expired: "muted",
};

export function StatusBadge({ status }: { status: string }) {
  const tone = STATUS_TONE[status?.toLowerCase()] ?? "muted";
  return <span className={`badge badge--${tone}`}>{status?.replace(/_/g, " ")}</span>;
}

export function Notice({
  kind = "info",
  children,
}: {
  kind?: "info" | "warn" | "error";
  children: ReactNode;
}) {
  return <div className={`notice notice--${kind}`}>{children}</div>;
}

export function ErrorNotice({ error }: { error: unknown }) {
  if (!error) return null;
  const message =
    error instanceof ApiError
      ? error.message
      : error instanceof Error
        ? error.message
        : String(error);
  return <Notice kind="error">{message}</Notice>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="page-header">
      <div>
        <h1>{title}</h1>
        {description ? <p>{description}</p> : null}
      </div>
      {actions ? <div className="button-row">{actions}</div> : null}
    </div>
  );
}

export function Disclaimer() {
  return (
    <p className="disclaimer">
      Past performance and backtest results do not guarantee future performance. This software
      executes trades according to rules you configure; it does not provide investment advice
      and does not guarantee any return. You can lose money.
    </p>
  );
}

/**
 * A dependency-free equity sparkline.
 *
 * Deliberately not a charting library: one SVG path is all this needs, and a charting
 * dependency in a dashboard that holds trading data is surface area for no benefit.
 */
export function Sparkline({
  points,
  height = 60,
}: {
  points: number[];
  height?: number;
}) {
  if (points.length < 2) return <div className="empty">Not enough data to plot</div>;

  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  const step = 100 / (points.length - 1);

  const path = points
    .map((value, index) => {
      const x = index * step;
      const y = height - ((value - min) / range) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");

  const rising = points[points.length - 1] >= points[0];

  return (
    <svg
      className="spark"
      viewBox={`0 0 100 ${height}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={`Equity from ${points[0].toFixed(2)} to ${points[points.length - 1].toFixed(2)}`}
    >
      <path
        d={path}
        fill="none"
        stroke={rising ? "var(--profit)" : "var(--loss)"}
        strokeWidth="1.5"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

/** Small helper for the many "fetch on mount, show error, allow refresh" pages. */
export function useAsync<T>(
  loader: () => Promise<T>,
  deps: unknown[] = [],
): { data: T | null; error: unknown; loading: boolean; reload: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    loader()
      .then((result) => {
        if (!cancelled) {
          setData(result);
          setError(null);
        }
      })
      .catch((caught) => {
        if (!cancelled) setError(caught);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  return { data, error, loading, reload: () => setNonce((n) => n + 1) };
}

/**
 * Confirmation gate for irreversible actions.
 *
 * Requires the user to type an exact phrase. Used for anything that moves money or stops a
 * running bot — a single click is too low a bar for those.
 */
export function ConfirmAction({
  phrase,
  label,
  description,
  onConfirm,
  danger = true,
}: {
  phrase: string;
  label: string;
  description: ReactNode;
  onConfirm: () => Promise<void> | void;
  danger?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);

  if (!open) {
    return (
      <button type="button" className={danger ? "danger" : ""} onClick={() => setOpen(true)}>
        {label}
      </button>
    );
  }

  return (
    <div className="card" style={{ marginTop: "0.5rem" }}>
      <div style={{ marginBottom: "0.6rem" }}>{description}</div>
      <div className="field">
        <label htmlFor="confirm-phrase">
          Type <code>{phrase}</code> to confirm
        </label>
        <input
          id="confirm-phrase"
          value={typed}
          onChange={(event) => setTyped(event.target.value)}
          autoComplete="off"
        />
      </div>
      <div className="button-row">
        <button
          type="button"
          className={danger ? "danger" : "primary"}
          disabled={typed !== phrase || busy}
          onClick={async () => {
            setBusy(true);
            try {
              await onConfirm();
              setOpen(false);
              setTyped("");
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? "Working…" : label}
        </button>
        <button type="button" onClick={() => setOpen(false)} disabled={busy}>
          Cancel
        </button>
      </div>
    </div>
  );
}
