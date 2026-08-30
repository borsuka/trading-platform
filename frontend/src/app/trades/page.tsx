"use client";

import { Shell } from "@/components/Shell";
import {
  Disclaimer,
  Empty,
  ErrorNotice,
  Money,
  Notice,
  PageHeader,
  Percent,
  Stat,
  useAsync,
} from "@/components/ui";
import { get } from "@/lib/api";

type Trade = {
  id: string;
  symbol: string;
  side: string;
  quantity: number;
  entry_price: number;
  exit_price: number | null;
  entry_time: string;
  exit_time: string | null;
  net_pnl: number;
  fees: number;
  return_pct: number;
  r_multiple: number | null;
  exit_reason: string | null;
  strategy_name: string | null;
};

type Summary = {
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  net_pnl: number;
  fees: number;
};

export default function TradesPage() {
  return (
    <Shell>
      <Trades />
    </Shell>
  );
}

function Trades() {
  const trades = useAsync(() => get<Trade[]>("/api/v1/trades?limit=200"));
  const summary = useAsync(() => get<Summary>("/api/v1/trades/summary"));

  const meaningful = (summary.data?.total_trades ?? 0) >= 30;

  return (
    <>
      <PageHeader
        title="Trades"
        description="Completed round-trips, newest first."
        actions={
          <button
            type="button"
            onClick={() => {
              trades.reload();
              summary.reload();
            }}
          >
            Refresh
          </button>
        }
      />

      <ErrorNotice error={trades.error ?? summary.error} />

      <div className="grid grid--stats" style={{ marginBottom: "1rem" }}>
        <Stat label="Closed trades" value={summary.data?.total_trades ?? 0} />
        <Stat
          label="Win rate"
          value={
            summary.data && summary.data.total_trades > 0 ? (
              <Percent value={summary.data.win_rate} signed={false} />
            ) : (
              "—"
            )
          }
          hint={
            summary.data
              ? `${summary.data.wins} wins, ${summary.data.losses} losses`
              : undefined
          }
        />
        <Stat label="Net P&L" value={<Money value={summary.data?.net_pnl} />} />
        <Stat
          label="Fees paid"
          value={<Money value={summary.data?.fees} signed={false} />}
          hint="Already deducted from net P&L"
        />
      </div>

      {summary.data && summary.data.total_trades > 0 && !meaningful ? (
        <Notice kind="warn">
          Only {summary.data.total_trades} closed trades. Win rate and related ratios are not
          statistically reliable below about 30 — a run of luck in either direction dominates
          the number at this sample size.
        </Notice>
      ) : null}

      {trades.data && trades.data.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Closed</th>
                <th>Symbol</th>
                <th>Side</th>
                <th className="num">Entry</th>
                <th className="num">Exit</th>
                <th className="num">Net P&L</th>
                <th className="num">Return</th>
                <th className="num">R</th>
                <th>Reason</th>
                <th>Strategy</th>
              </tr>
            </thead>
            <tbody>
              {trades.data.map((trade) => (
                <tr key={trade.id}>
                  <td className="mono muted">
                    {trade.exit_time ? new Date(trade.exit_time).toLocaleString() : "open"}
                  </td>
                  <td className="mono">{trade.symbol}</td>
                  <td>
                    <span className={`badge badge--${trade.side === "long" ? "ok" : "warn"}`}>
                      {trade.side}
                    </span>
                  </td>
                  <td className="num">{trade.entry_price.toFixed(2)}</td>
                  <td className="num">{trade.exit_price?.toFixed(2) ?? "—"}</td>
                  <td className="num">
                    <Money value={trade.net_pnl} />
                  </td>
                  <td className="num">
                    <Percent value={trade.return_pct} />
                  </td>
                  <td className="num">
                    {trade.r_multiple != null ? (
                      <Money value={trade.r_multiple} decimals={2} />
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="muted">{trade.exit_reason?.replace(/_/g, " ") ?? "—"}</td>
                  <td className="muted">{trade.strategy_name ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>No completed trades yet.</Empty>
      )}

      <Notice kind="info">
        <strong>R</strong> is profit or loss measured in units of the trade&apos;s initial risk.
        It is the cleanest way to compare trades across symbols and position sizes: +2R means
        the trade made twice what it was risking.
      </Notice>

      <Disclaimer />
    </>
  );
}
