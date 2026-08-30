"use client";

import { useState } from "react";

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
  useAsync,
} from "@/components/ui";
import { get, post } from "@/lib/api";

type Position = {
  bot_id: string;
  bot_name?: string;
  symbol: string;
  side: string;
  quantity: number;
  entry_price: number;
  mark_price: number | null;
  unrealized_pnl: number;
  unrealized_pnl_pct?: number;
  stop_loss: number | null;
  take_profit: number | null;
  leverage: number;
  strategy_name: string | null;
  source: string;
};

export default function PositionsPage() {
  return (
    <Shell>
      <Positions />
    </Shell>
  );
}

function Positions() {
  const positions = useAsync(() => get<Position[]>("/api/v1/positions"));
  const [error, setError] = useState<unknown>(null);

  async function close(botId: string, symbol: string) {
    setError(null);
    try {
      await post(`/api/v1/positions/${botId}/${symbol}/close`);
      positions.reload();
    } catch (caught) {
      setError(caught);
    }
  }

  return (
    <>
      <PageHeader
        title="Positions"
        description="Open positions across every running bot, marked to the latest price."
        actions={
          <button type="button" onClick={positions.reload}>
            Refresh
          </button>
        }
      />

      <ErrorNotice error={error ?? positions.error} />

      {positions.data && positions.data.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Side</th>
                <th className="num">Quantity</th>
                <th className="num">Entry</th>
                <th className="num">Mark</th>
                <th className="num">Unrealised</th>
                <th className="num">Stop</th>
                <th className="num">Target</th>
                <th>Strategy</th>
                <th>Close</th>
              </tr>
            </thead>
            <tbody>
              {positions.data.map((position) => (
                <tr key={`${position.bot_id}-${position.symbol}`}>
                  <td className="mono">
                    <strong>{position.symbol}</strong>
                    {position.bot_name ? (
                      <div className="muted" style={{ fontSize: "0.74rem" }}>
                        {position.bot_name}
                      </div>
                    ) : null}
                  </td>
                  <td>
                    <span className={`badge badge--${position.side === "long" ? "ok" : "warn"}`}>
                      {position.side}
                    </span>
                  </td>
                  <td className="num">{position.quantity}</td>
                  <td className="num">{position.entry_price.toFixed(2)}</td>
                  <td className="num">{position.mark_price?.toFixed(2) ?? "—"}</td>
                  <td className="num">
                    <Money value={position.unrealized_pnl} />
                    {position.unrealized_pnl_pct != null ? (
                      <div style={{ fontSize: "0.74rem" }}>
                        <Percent value={position.unrealized_pnl_pct} />
                      </div>
                    ) : null}
                  </td>
                  <td className="num">{position.stop_loss?.toFixed(2) ?? "—"}</td>
                  <td className="num">{position.take_profit?.toFixed(2) ?? "—"}</td>
                  <td className="muted">{position.strategy_name ?? "—"}</td>
                  <td>
                    {position.source === "runtime" ? (
                      <ConfirmAction
                        phrase={position.symbol}
                        label="Close"
                        description={
                          <>
                            Close the {position.side} position in{" "}
                            <strong>{position.symbol}</strong> at the market price. Current
                            unrealised P&amp;L is <Money value={position.unrealized_pnl} />.
                          </>
                        }
                        onConfirm={() => close(position.bot_id, position.symbol)}
                      />
                    ) : (
                      <span className="muted">bot not running</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>No open positions.</Empty>
      )}

      <Notice kind="info">
        Closing a position is never blocked by a risk limit. Reducing exposure is not the risky
        direction, so it is always permitted — even when the kill switch is engaged.
      </Notice>

      <Disclaimer />
    </>
  );
}
