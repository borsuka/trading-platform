"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";

import { Shell } from "@/components/Shell";
import {
  ConfirmAction,
  Disclaimer,
  Empty,
  ErrorNotice,
  Money,
  Notice,
  PageHeader,
  Percent,
  Stat,
  StatusBadge,
  useAsync,
} from "@/components/ui";
import { get, post } from "@/lib/api";

type Runtime = {
  bot_id: string;
  name: string;
  status: string;
  mode: string;
  symbols: string[];
  equity: number;
  cash: number;
  unrealized_pnl: number;
  realized_pnl: number;
  open_positions: number;
  drawdown: number;
  cycles: number;
  last_cycle_at: string | null;
  kill_switch: { active: boolean; reason?: string | null; message?: string | null };
  halt_reason: string | null;
  last_error: string | null;
};

type BotEvent = {
  event_type: string;
  message: string;
  occurred_at: string;
  severity: string;
};

export default function BotDetailPage() {
  return (
    <Shell>
      {/* `useSearchParams` suspends on first render, and a statically exported page has no
          server to stream a fallback from, so the boundary has to be here in the tree. */}
      <Suspense fallback={<Notice>Loading bot…</Notice>}>
        <BotDetail />
      </Suspense>
    </Shell>
  );
}

function BotDetail() {
  // The bot id travels as a query parameter rather than a path segment. A path segment would
  // make this a dynamic route, and a dynamic route cannot be statically exported without
  // knowing every id at build time - which, for user-created bots, is impossible.
  const id = useSearchParams().get("id") ?? "";

  const runtime = useAsync(
    () => get<Runtime>(`/api/v1/bots/${id}/runtime`).catch(() => null),
    [id],
  );
  const events = useAsync(
    () => get<BotEvent[]>(`/api/v1/bots/${id}/events?limit=80`).catch(() => []),
    [id],
  );
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  async function act(action: string, body?: unknown) {
    setBusy(true);
    setError(null);
    try {
      await post(`/api/v1/bots/${id}/${action}`, body);
      runtime.reload();
      events.reload();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  const state = runtime.data;

  return (
    <>
      <PageHeader
        title={state?.name ?? "Bot"}
        description={
          state
            ? `${state.mode} • ${state.symbols.join(", ")} • ${state.cycles} decision cycles`
            : "This bot is not currently running in this process."
        }
        actions={
          <>
            <button type="button" onClick={() => { runtime.reload(); events.reload(); }}>
              Refresh
            </button>
            {state ? (
              <button type="button" disabled={busy} onClick={() => act("cycle")}>
                Run one cycle
              </button>
            ) : null}
          </>
        }
      />

      <ErrorNotice error={error} />

      {!state ? (
        <Empty>
          This bot is not running. Start it from the bots page to see live runtime state.
        </Empty>
      ) : (
        <>
          {state.kill_switch.active ? (
            <Notice kind="error">
              <strong>Kill switch engaged</strong>
              {state.kill_switch.reason ? ` (${state.kill_switch.reason})` : ""}:{" "}
              {state.kill_switch.message}. No new orders will be placed. Open positions are
              untouched — clearing the switch does not resume trading, and the bot must be
              restarted separately.
            </Notice>
          ) : null}

          {state.halt_reason ? (
            <Notice kind="error">
              <strong>Halted:</strong> {state.halt_reason}
            </Notice>
          ) : null}

          <div className="grid grid--stats" style={{ marginBottom: "1rem" }}>
            <Stat label="Status" value={<StatusBadge status={state.status} />} />
            <Stat label="Equity" value={<Money value={state.equity} signed={false} />} />
            <Stat label="Realised P&L" value={<Money value={state.realized_pnl} />} />
            <Stat label="Unrealised P&L" value={<Money value={state.unrealized_pnl} />} />
            <Stat
              label="Drawdown"
              value={<Percent value={state.drawdown} signed={false} />}
            />
            <Stat label="Open positions" value={state.open_positions} />
          </div>

          <div className="button-row" style={{ marginBottom: "1rem" }}>
            {state.status === "running" ? (
              <button type="button" disabled={busy} onClick={() => act("pause")}>
                Pause
              </button>
            ) : null}
            {state.status === "paused" ? (
              <button type="button" disabled={busy} onClick={() => act("resume")}>
                Resume
              </button>
            ) : null}
            <button
              type="button"
              disabled={busy}
              onClick={() => act("stop", { close_positions: false })}
            >
              Stop (keep positions)
            </button>
            <ConfirmAction
              phrase="EMERGENCY STOP"
              label="Emergency stop"
              description={
                <>
                  This trips the kill switch and cancels resting orders.{" "}
                  <strong>Open positions are left open</strong> — an emergency is frequently the
                  worst moment to be a forced seller. Close them separately if that is what you
                  want.
                </>
              }
              onConfirm={() => act("emergency-stop", { close_positions: false, note: "manual" })}
            />
          </div>
        </>
      )}

      <div className="card">
        <h2>Activity</h2>
        {events.data && events.data.length > 0 ? (
          <div className="table-wrap" style={{ border: "none" }}>
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Event</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {events.data.map((event, index) => (
                  <tr key={`${event.occurred_at}-${index}`}>
                    <td className="mono muted">
                      {new Date(event.occurred_at).toLocaleTimeString()}
                    </td>
                    <td>
                      <span
                        className={`badge badge--${
                          event.severity === "critical"
                            ? "danger"
                            : event.severity === "warning"
                              ? "warn"
                              : "muted"
                        }`}
                      >
                        {event.event_type.replace(/_/g, " ")}
                      </span>
                    </td>
                    <td style={{ whiteSpace: "normal" }}>{event.message}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>No events yet.</Empty>
        )}
      </div>

      <Disclaimer />
    </>
  );
}
