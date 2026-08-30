"use client";

import { useState } from "react";

import { Shell } from "@/components/Shell";
import {
  Disclaimer,
  Empty,
  ErrorNotice,
  Money,
  Notice,
  PageHeader,
  Percent,
  StatusBadge,
  useAsync,
} from "@/components/ui";
import { get, post } from "@/lib/api";

type Backtest = {
  id: string;
  name: string;
  strategy_type: string;
  symbol: string;
  interval: string;
  status: string;
  metrics: Record<string, number | string | boolean | string[]>;
  error_message: string | null;
  created_at: string;
};

type BacktestDetail = Backtest & {
  equity_curve: Array<{ timestamp: string; equity: number }>;
  monthly_returns: Record<string, number>;
  warnings: string[];
  disclaimer: string;
};

export default function BacktestsPage() {
  return (
    <Shell>
      <Backtests />
    </Shell>
  );
}

function Backtests() {
  const list = useAsync(() => get<Backtest[]>("/api/v1/backtests?limit=50"));
  const [selected, setSelected] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const [form, setForm] = useState({
    name: "",
    strategy_type: "trend_following",
    symbol: "BTCUSDT",
    interval: "1h",
    bars: 2000,
    seed: 42,
    run_validation: true,
  });

  const detail = useAsync(
    () =>
      selected
        ? get<BacktestDetail>(`/api/v1/backtests/${selected}`)
        : Promise.resolve(null),
    [selected],
  );

  async function run(event: React.FormEvent) {
    event.preventDefault();
    setRunning(true);
    setError(null);
    try {
      const created = await post<Backtest>("/api/v1/backtests", {
        ...form,
        name: form.name || `${form.strategy_type} ${form.symbol}`,
        initial_balance: 10000,
      });
      list.reload();
      setSelected(created.id);
    } catch (caught) {
      setError(caught);
    } finally {
      setRunning(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Backtests"
        description="Simulations over historical bars using the same decision path as live trading."
        actions={
          <button type="button" onClick={list.reload}>
            Refresh
          </button>
        }
      />

      <ErrorNotice error={error ?? list.error} />

      <Notice kind="warn">
        Backtests here use <strong>synthetic</strong> data — a deterministic random walk, not
        market data. They demonstrate that the engine and the strategy logic work. They are not
        a performance claim, and no result from them says anything about future returns.
      </Notice>

      <form className="card" onSubmit={run} style={{ marginBottom: "1rem" }}>
        <h2>Run a backtest</h2>
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))" }}>
          <div className="field">
            <label htmlFor="bt-strategy">Strategy</label>
            <select
              id="bt-strategy"
              value={form.strategy_type}
              onChange={(e) => setForm({ ...form, strategy_type: e.target.value })}
            >
              {["trend_following", "momentum_breakout", "mean_reversion", "multi_factor"].map(
                (s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ),
              )}
            </select>
          </div>
          <div className="field">
            <label htmlFor="bt-symbol">Symbol</label>
            <input
              id="bt-symbol"
              value={form.symbol}
              onChange={(e) => setForm({ ...form, symbol: e.target.value })}
            />
          </div>
          <div className="field">
            <label htmlFor="bt-interval">Interval</label>
            <select
              id="bt-interval"
              value={form.interval}
              onChange={(e) => setForm({ ...form, interval: e.target.value })}
            >
              {["15m", "1h", "4h", "1d"].map((i) => (
                <option key={i} value={i}>
                  {i}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="bt-bars">Bars</label>
            <input
              id="bt-bars"
              type="number"
              min={300}
              max={20000}
              step={100}
              value={form.bars}
              onChange={(e) => setForm({ ...form, bars: Number(e.target.value) })}
            />
          </div>
          <div className="field">
            <label htmlFor="bt-seed">Seed</label>
            <input
              id="bt-seed"
              type="number"
              value={form.seed}
              onChange={(e) => setForm({ ...form, seed: Number(e.target.value) })}
            />
            <div className="field__hint">Same seed, same data</div>
          </div>
        </div>

        <label style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
          <input
            type="checkbox"
            style={{ width: "auto" }}
            checked={form.run_validation}
            onChange={(e) => setForm({ ...form, run_validation: e.target.checked })}
          />
          Also run robustness analysis (walk-forward and Monte Carlo)
        </label>
        <div className="field__hint" style={{ marginBottom: "0.7rem" }}>
          Strongly recommended. A single in-sample backtest is weak evidence on its own.
        </div>

        <button type="submit" className="primary" disabled={running}>
          {running ? "Running…" : "Run backtest"}
        </button>
      </form>

      {list.data && list.data.length > 0 ? (
        <div className="table-wrap" style={{ marginBottom: "1rem" }}>
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Strategy</th>
                <th>Status</th>
                <th className="num">Return</th>
                <th className="num">Drawdown</th>
                <th className="num">Trades</th>
                <th className="num">Sharpe</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {list.data.map((item) => (
                <tr key={item.id}>
                  <td>{item.name}</td>
                  <td className="muted">{item.strategy_type}</td>
                  <td>
                    <StatusBadge status={item.status} />
                  </td>
                  <td className="num">
                    <Percent value={Number(item.metrics?.total_return ?? 0)} />
                  </td>
                  <td className="num">
                    <Percent value={Number(item.metrics?.max_drawdown ?? 0)} signed={false} />
                  </td>
                  <td className="num">{Number(item.metrics?.total_trades ?? 0)}</td>
                  <td className="num">{Number(item.metrics?.sharpe_ratio ?? 0).toFixed(2)}</td>
                  <td>
                    <button type="button" onClick={() => setSelected(item.id)}>
                      Details
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>No backtests yet.</Empty>
      )}

      {detail.data ? <BacktestDetailView detail={detail.data} /> : null}

      <Disclaimer />
    </>
  );
}

function BacktestDetailView({ detail }: { detail: BacktestDetail }) {
  const metrics = detail.metrics as Record<string, number>;
  const warnings = (detail.metrics.reliability_warnings as string[]) ?? [];
  const allWarnings = [...(detail.warnings ?? []), ...warnings];

  return (
    <div className="card">
      <h2>{detail.name}</h2>

      {allWarnings.length > 0 ? (
        <div className="notice notice--warn">
          <strong>Read before interpreting these numbers:</strong>
          <ul style={{ margin: "0.4rem 0 0", paddingLeft: "1.1rem" }}>
            {allWarnings.map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="table-wrap">
        <table>
          <tbody>
            {[
              ["Total return", <Percent key="r" value={metrics.total_return} />],
              ["CAGR", <Percent key="c" value={metrics.cagr} />],
              ["Max drawdown", <Percent key="d" value={metrics.max_drawdown} signed={false} />],
              ["Sharpe", metrics.sharpe_ratio?.toFixed(2)],
              ["Sortino", metrics.sortino_ratio?.toFixed(2)],
              ["Calmar", metrics.calmar_ratio?.toFixed(2)],
              ["Trades", metrics.total_trades],
              ["Win rate", <Percent key="w" value={metrics.win_rate} signed={false} />],
              ["Profit factor", metrics.profit_factor?.toFixed(2)],
              ["Expectancy", <Money key="e" value={metrics.expectancy} />],
              ["Max consecutive losses", metrics.max_consecutive_losses],
              ["Total fees", <Money key="f" value={metrics.total_fees} signed={false} />],
              [
                "Fees as % of gross profit",
                <Percent key="fp" value={metrics.fees_as_pct_of_gross_profit} signed={false} />,
              ],
            ].map(([label, value]) => (
              <tr key={String(label)}>
                <td className="muted">{label}</td>
                <td className="num">{value as React.ReactNode}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="muted" style={{ fontSize: "0.78rem", marginTop: "0.8rem" }}>
        {detail.disclaimer}
      </p>
    </div>
  );
}
