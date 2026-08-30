"use client";

import { useState } from "react";

import { Shell } from "@/components/Shell";
import {
  ConfirmAction,
  Disclaimer,
  Empty,
  ErrorNotice,
  Notice,
  PageHeader,
  Percent,
  Stat,
  useAsync,
} from "@/components/ui";
import { get, post } from "@/lib/api";

type Bot = { id: string; name: string; status: string; kill_switch_active: boolean };

type RiskStatus = {
  limits: Record<string, number | boolean>;
  state: Record<string, number | string | null>;
  kill_switch: {
    active: boolean;
    reason?: string | null;
    message?: string | null;
    triggered_at?: string;
  };
  headroom: Record<string, number>;
};

type RiskEvent = {
  event_type: string;
  severity: string;
  symbol: string | null;
  message: string;
  occurred_at: string;
};

export default function RiskPage() {
  return (
    <Shell>
      <Risk />
    </Shell>
  );
}

function Risk() {
  const bots = useAsync(() => get<Bot[]>("/api/v1/bots"));
  const events = useAsync(() => get<RiskEvent[]>("/api/v1/risk/events?limit=100"));
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);

  const status = useAsync(
    () =>
      selected
        ? get<RiskStatus>(`/api/v1/risk/status/${selected}`).catch(() => null)
        : Promise.resolve(null),
    [selected],
  );

  async function resetKillSwitch(botId: string) {
    setError(null);
    try {
      await post(`/api/v1/risk/kill-switch/${botId}/reset`, { note: "cleared from dashboard" });
      status.reload();
      bots.reload();
      events.reload();
    } catch (caught) {
      setError(caught);
    }
  }

  return (
    <>
      <PageHeader
        title="Risk"
        description="Limits, remaining headroom, and every event where risk blocked a trade."
        actions={
          <button
            type="button"
            onClick={() => {
              bots.reload();
              events.reload();
              status.reload();
            }}
          >
            Refresh
          </button>
        }
      />

      <ErrorNotice error={error ?? events.error} />

      <div className="field" style={{ maxWidth: 320 }}>
        <label htmlFor="risk-bot">Bot</label>
        <select
          id="risk-bot"
          value={selected ?? ""}
          onChange={(e) => setSelected(e.target.value || null)}
        >
          <option value="">Select a running bot…</option>
          {(bots.data ?? []).map((bot) => (
            <option key={bot.id} value={bot.id}>
              {bot.name} ({bot.status})
            </option>
          ))}
        </select>
      </div>

      {status.data ? (
        <>
          {status.data.kill_switch.active ? (
            <div className="notice notice--error">
              <strong>Kill switch engaged</strong>
              {status.data.kill_switch.reason ? ` — ${status.data.kill_switch.reason}` : ""}:{" "}
              {status.data.kill_switch.message}
              <div style={{ marginTop: "0.6rem" }}>
                <ConfirmAction
                  phrase="RESET"
                  label="Reset kill switch"
                  description={
                    <>
                      Clearing the kill switch allows new orders again. Make sure you understand
                      why it tripped first — the reset is recorded in the audit log with your
                      identity. The bot must still be restarted separately.
                    </>
                  }
                  onConfirm={() => resetKillSwitch(selected!)}
                />
              </div>
            </div>
          ) : (
            <Notice kind="info">Kill switch is clear. New orders are permitted.</Notice>
          )}

          <div className="grid grid--stats" style={{ marginBottom: "1rem" }}>
            <Stat
              label="Daily loss headroom"
              value={
                <Percent value={status.data.headroom.daily_loss_remaining} signed={false} />
              }
              hint="Before trading stops for the day"
            />
            <Stat
              label="Drawdown headroom"
              value={<Percent value={status.data.headroom.drawdown_remaining} signed={false} />}
              hint="Before the kill switch trips"
            />
            <Stat
              label="Trades left today"
              value={status.data.headroom.trades_remaining_today}
            />
            <Stat
              label="Current drawdown"
              value={<Percent value={Number(status.data.state.drawdown ?? 0)} signed={false} />}
            />
          </div>

          <div className="card" style={{ marginBottom: "1rem" }}>
            <h2>Limits</h2>
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <tbody>
                  {Object.entries(status.data.limits)
                    .filter(([key]) => key.startsWith("max_") || key.startsWith("risk_"))
                    .map(([key, value]) => (
                      <tr key={key}>
                        <td className="muted mono">{key}</td>
                        <td className="num">{String(value)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : selected ? (
        <Empty>That bot is not running, so it has no live risk state.</Empty>
      ) : null}

      <div className="card">
        <h2>Risk events</h2>
        <p className="muted" style={{ fontSize: "0.82rem" }}>
          Every time risk blocked a trade or a limit was breached. These are the most useful
          records for answering &ldquo;why didn&apos;t it trade?&rdquo;
        </p>
        {events.data && events.data.length > 0 ? (
          <div className="table-wrap" style={{ border: "none" }}>
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Type</th>
                  <th>Symbol</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {events.data.map((event, index) => (
                  <tr key={`${event.occurred_at}-${index}`}>
                    <td className="mono muted">
                      {new Date(event.occurred_at).toLocaleString()}
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
                    <td className="mono">{event.symbol ?? "—"}</td>
                    <td style={{ whiteSpace: "normal" }}>{event.message}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>No risk events recorded.</Empty>
        )}
      </div>

      <Disclaimer />
    </>
  );
}
