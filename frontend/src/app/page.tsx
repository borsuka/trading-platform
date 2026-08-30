"use client";

import Link from "next/link";

import { Shell } from "@/components/Shell";
import {
  Disclaimer,
  Empty,
  ErrorNotice,
  Money,
  Percent,
  PageHeader,
  Sparkline,
  Stat,
  StatusBadge,
  useAsync,
} from "@/components/ui";
import { get } from "@/lib/api";

type Aggregate = {
  equity: number;
  cash: number;
  realized_pnl: number;
  unrealized_pnl: number;
  fees_paid: number;
  total_return: number;
  drawdown: number;
  open_positions: number;
  total_exposure: number;
  closed_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  profit_factor: number;
};

type PortfolioResponse = {
  bots: Array<{ bot_id: string; name: string } & Record<string, number>>;
  aggregate: Aggregate;
};

type CurvePoint = { timestamp: string; equity: number };

type BotRow = {
  id: string;
  name: string;
  status: string;
  symbols: string[];
  interval: string;
  kill_switch_active: boolean;
};

export default function DashboardPage() {
  return (
    <Shell>
      <Dashboard />
    </Shell>
  );
}

function Dashboard() {
  const portfolio = useAsync(() => get<PortfolioResponse>("/api/v1/portfolio"));
  const curve = useAsync(() =>
    get<CurvePoint[]>("/api/v1/portfolio/equity-curve?limit=500"),
  );
  const bots = useAsync(() => get<BotRow[]>("/api/v1/bots"));

  const aggregate = portfolio.data?.aggregate;
  const running = bots.data?.filter((b) => b.status === "running") ?? [];
  const halted = bots.data?.filter((b) => ["halted", "error"].includes(b.status)) ?? [];

  return (
    <>
      <PageHeader
        title="Dashboard"
        description="Aggregate view across every bot currently running in this process."
        actions={
          <button type="button" onClick={() => { portfolio.reload(); curve.reload(); bots.reload(); }}>
            Refresh
          </button>
        }
      />

      <ErrorNotice error={portfolio.error ?? bots.error} />

      {halted.length > 0 ? (
        <div className="notice notice--error">
          <strong>{halted.length} bot(s) need attention:</strong>{" "}
          {halted.map((b) => b.name).join(", ")}. A halted bot has stopped opening positions —
          open positions are untouched. See the bot page for the cause.
        </div>
      ) : null}

      <div className="grid grid--stats" style={{ marginBottom: "1rem" }}>
        <Stat
          label="Equity"
          value={<Money value={aggregate?.equity} signed={false} />}
          hint={
            aggregate ? (
              <>
                Total return <Percent value={aggregate.total_return} />
              </>
            ) : null
          }
        />
        <Stat label="Realised P&L" value={<Money value={aggregate?.realized_pnl} />} />
        <Stat label="Unrealised P&L" value={<Money value={aggregate?.unrealized_pnl} />} />
        <Stat
          label="Drawdown"
          value={<Percent value={aggregate?.drawdown} signed={false} />}
          hint="From the equity high-water mark"
        />
        <Stat
          label="Open positions"
          value={aggregate?.open_positions ?? 0}
          hint={
            aggregate ? (
              <>
                Exposure <Money value={aggregate.total_exposure} signed={false} />
              </>
            ) : null
          }
        />
        <Stat
          label="Win rate"
          value={
            aggregate && aggregate.closed_trades > 0 ? (
              <Percent value={aggregate.win_rate} signed={false} />
            ) : (
              "—"
            )
          }
          hint={
            aggregate ? (
              <>
                {aggregate.closed_trades} closed
                {aggregate.closed_trades > 0 && aggregate.closed_trades < 30
                  ? " — too few to be meaningful"
                  : ""}
              </>
            ) : null
          }
        />
      </div>

      <div className="grid grid--cards">
        <div className="card">
          <h2>Equity curve</h2>
          {curve.data && curve.data.length > 1 ? (
            <>
              <Sparkline points={curve.data.map((p) => p.equity)} height={80} />
              <p className="muted" style={{ fontSize: "0.78rem", marginTop: "0.4rem" }}>
                {curve.data.length} snapshots
              </p>
            </>
          ) : (
            <Empty>Start a bot to record an equity curve.</Empty>
          )}
        </div>

        <div className="card">
          <h2>Bots</h2>
          {bots.data && bots.data.length > 0 ? (
            <div className="stack">
              {bots.data.slice(0, 6).map((bot) => (
                <div
                  key={bot.id}
                  style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem" }}
                >
                  <Link href={`/bots/view?id=${bot.id}`}>{bot.name}</Link>
                  <span>
                    {bot.kill_switch_active ? (
                      <span className="badge badge--danger">kill switch</span>
                    ) : null}{" "}
                    <StatusBadge status={bot.status} />
                  </span>
                </div>
              ))}
              <p className="muted" style={{ fontSize: "0.8rem" }}>
                {running.length} running of {bots.data.length}
              </p>
            </div>
          ) : (
            <Empty>
              No bots yet. <Link href="/bots">Create one</Link> to start paper trading.
            </Empty>
          )}
        </div>
      </div>

      {aggregate && aggregate.closed_trades > 0 && aggregate.closed_trades < 30 ? (
        <div className="notice notice--warn" style={{ marginTop: "1rem" }}>
          Only {aggregate.closed_trades} closed trades. Win rate and profit factor are not
          statistically reliable below about 30 trades — treat them as noise for now.
        </div>
      ) : null}

      <Disclaimer />
    </>
  );
}
