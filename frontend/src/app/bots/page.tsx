"use client";

import Link from "next/link";
import { useState } from "react";

import { Shell } from "@/components/Shell";
import {
  Disclaimer,
  Empty,
  ErrorNotice,
  Notice,
  PageHeader,
  StatusBadge,
  useAsync,
} from "@/components/ui";
import { get, post } from "@/lib/api";

type Bot = {
  id: string;
  name: string;
  strategy_id: string;
  trading_mode: string;
  symbols: string[];
  interval: string;
  status: string;
  kill_switch_active: boolean;
  last_error: string | null;
};

type CatalogEntry = { name: string; description: string };

export default function BotsPage() {
  return (
    <Shell>
      <Bots />
    </Shell>
  );
}

function Bots() {
  const bots = useAsync(() => get<Bot[]>("/api/v1/bots"));
  const catalog = useAsync(() => get<CatalogEntry[]>("/api/v1/strategies/catalog"));
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  async function act(id: string, action: string, body?: unknown) {
    setBusyId(id);
    setError(null);
    try {
      await post(`/api/v1/bots/${id}/${action}`, body);
      bots.reload();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusyId(null);
    }
  }

  return (
    <>
      <PageHeader
        title="Bots"
        description="Every bot runs in paper mode. Live trading requires the separate activation flow on the exchange accounts page."
        actions={
          <button type="button" className="primary" onClick={() => setCreating(!creating)}>
            {creating ? "Cancel" : "New bot"}
          </button>
        }
      />

      <ErrorNotice error={error ?? bots.error} />

      {creating ? (
        <CreateBotForm
          strategies={catalog.data ?? []}
          onCreated={() => {
            setCreating(false);
            bots.reload();
          }}
        />
      ) : null}

      {bots.data && bots.data.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Symbols</th>
                <th>Interval</th>
                <th>Mode</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {bots.data.map((bot) => (
                <tr key={bot.id}>
                  <td>
                    <Link href={`/bots/view?id=${bot.id}`}>{bot.name}</Link>
                    {bot.last_error ? (
                      <div className="muted" style={{ fontSize: "0.74rem" }}>
                        {bot.last_error}
                      </div>
                    ) : null}
                  </td>
                  <td className="mono">{bot.symbols.join(", ")}</td>
                  <td className="mono">{bot.interval}</td>
                  <td>
                    <span className="badge badge--ok">{bot.trading_mode}</span>
                  </td>
                  <td>
                    {bot.kill_switch_active ? (
                      <span className="badge badge--danger">kill switch</span>
                    ) : (
                      <StatusBadge status={bot.status} />
                    )}
                  </td>
                  <td>
                    <div className="button-row">
                      {bot.status === "running" || bot.status === "paused" ? (
                        <button
                          type="button"
                          disabled={busyId === bot.id}
                          onClick={() => act(bot.id, "stop", { close_positions: false })}
                        >
                          Stop
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="primary"
                          disabled={busyId === bot.id}
                          onClick={() => act(bot.id, "start")}
                        >
                          Start
                        </button>
                      )}
                      {bot.status === "running" ? (
                        <button
                          type="button"
                          disabled={busyId === bot.id}
                          onClick={() => act(bot.id, "pause")}
                        >
                          Pause
                        </button>
                      ) : null}
                      {bot.status === "paused" ? (
                        <button
                          type="button"
                          disabled={busyId === bot.id}
                          onClick={() => act(bot.id, "resume")}
                        >
                          Resume
                        </button>
                      ) : null}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>No bots yet. Create one to start paper trading.</Empty>
      )}

      <Notice kind="info">
        Stopping a bot does <strong>not</strong> close its open positions. That is deliberate: a
        scheduled stop should not become a forced sale at whatever price happens to be
        available. Close positions explicitly from the positions page.
      </Notice>

      <Disclaimer />
    </>
  );
}

function CreateBotForm({
  strategies,
  onCreated,
}: {
  strategies: CatalogEntry[];
  onCreated: () => void;
}) {
  const [name, setName] = useState("");
  const [strategy, setStrategy] = useState("trend_following");
  const [symbols, setSymbols] = useState("BTCUSDT");
  const [interval, setInterval] = useState("1h");
  const [balance, setBalance] = useState(10000);
  const [riskPerTrade, setRiskPerTrade] = useState(0.005);
  const [maxPositions, setMaxPositions] = useState(3);
  const [maxDrawdown, setMaxDrawdown] = useState(0.1);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const simultaneousRisk = riskPerTrade * maxPositions;

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await post("/api/v1/bots", {
        name,
        strategy_type: strategy,
        symbols: symbols.split(",").map((s) => s.trim()).filter(Boolean),
        interval,
        starting_balance: balance,
        risk: {
          risk_per_trade: riskPerTrade,
          max_concurrent_positions: maxPositions,
          max_drawdown: maxDrawdown,
          max_daily_loss: Math.min(0.02, maxDrawdown / 2),
          max_weekly_loss: Math.min(0.06, maxDrawdown * 0.8),
        },
      });
      onCreated();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="card" onSubmit={submit} style={{ marginBottom: "1rem" }}>
      <h2>New bot</h2>
      <ErrorNotice error={error} />

      <div className="grid grid--cards">
        <div>
          <div className="field">
            <label htmlFor="bot-name">Name</label>
            <input id="bot-name" required value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="bot-strategy">Strategy</label>
            <select
              id="bot-strategy"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
            >
              {strategies.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name}
                </option>
              ))}
            </select>
            <div className="field__hint">
              {strategies.find((s) => s.name === strategy)?.description}
            </div>
          </div>
          <div className="field">
            <label htmlFor="bot-symbols">Symbols</label>
            <input
              id="bot-symbols"
              required
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
            />
            <div className="field__hint">Comma-separated, e.g. BTCUSDT, ETHUSDT</div>
          </div>
          <div className="field">
            <label htmlFor="bot-interval">Interval</label>
            <select
              id="bot-interval"
              value={interval}
              onChange={(e) => setInterval(e.target.value)}
            >
              {["5m", "15m", "30m", "1h", "4h", "1d"].map((i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div>
          <div className="field">
            <label htmlFor="bot-balance">Starting balance</label>
            <input
              id="bot-balance"
              type="number"
              min={100}
              step={100}
              value={balance}
              onChange={(e) => setBalance(Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label htmlFor="bot-risk">Risk per trade</label>
            <input
              id="bot-risk"
              type="number"
              min={0.001}
              max={0.05}
              step={0.001}
              value={riskPerTrade}
              onChange={(e) => setRiskPerTrade(Number(e.target.value))}
            />
            <div className="field__hint">
              Fraction of equity lost if the stop is hit. 0.005 = 0.5%.
            </div>
          </div>
          <div className="field">
            <label htmlFor="bot-positions">Max concurrent positions</label>
            <input
              id="bot-positions"
              type="number"
              min={1}
              max={20}
              value={maxPositions}
              onChange={(e) => setMaxPositions(Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label htmlFor="bot-drawdown">Max drawdown (kill switch)</label>
            <input
              id="bot-drawdown"
              type="number"
              min={0.02}
              max={0.5}
              step={0.01}
              value={maxDrawdown}
              onChange={(e) => setMaxDrawdown(Number(e.target.value))}
            />
          </div>

          <div
            className={
              simultaneousRisk > maxDrawdown ? "notice notice--warn" : "notice notice--info"
            }
          >
            With these settings, <strong>{(simultaneousRisk * 100).toFixed(1)}%</strong> of
            equity can be at risk simultaneously
            {simultaneousRisk > maxDrawdown
              ? ` — more than the ${(maxDrawdown * 100).toFixed(0)}% drawdown limit, so the kill switch could trip before your positions resolve.`
              : "."}
          </div>
        </div>
      </div>

      <button type="submit" className="primary" disabled={busy}>
        {busy ? "Creating…" : "Create bot"}
      </button>
    </form>
  );
}
