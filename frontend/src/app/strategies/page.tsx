"use client";

import { useState } from "react";

import { Shell } from "@/components/Shell";
import {
  Disclaimer,
  Empty,
  ErrorNotice,
  Notice,
  PageHeader,
  useAsync,
} from "@/components/ui";
import { get } from "@/lib/api";

type CatalogEntry = {
  name: string;
  version: string;
  description: string;
  allowed_regimes: string[];
  default_parameters: Record<string, unknown>;
};

const FAILURE_MODES: Record<string, string> = {
  trend_following:
    "Whipsaw in a range is the dominant loss mode. Mitigated by the ADX filter and by refusing to trade the RANGING regime at all.",
  momentum_breakout:
    "False breakouts are the dominant loss mode. Mitigated by requiring a minimum break size, volume expansion, and no recent failed break in the opposite direction.",
  mean_reversion:
    "Fading a trend is how this style blows up: many small wins, then one loss that erases them. Four independent brakes guard against it.",
  multi_factor:
    "Factor correlation (trend and momentum agreeing can be one signal counted twice) and parameter overfitting across six weights.",
};

export default function StrategiesPage() {
  return (
    <Shell>
      <Strategies />
    </Shell>
  );
}

function Strategies() {
  const catalog = useAsync(() => get<CatalogEntry[]>("/api/v1/strategies/catalog"));
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <>
      <PageHeader
        title="Strategies"
        description="The strategies this platform ships, with their parameters and the ways they fail."
      />

      <ErrorNotice error={catalog.error} />

      {catalog.data && catalog.data.length > 0 ? (
        <div className="stack">
          {catalog.data.map((entry) => (
            <div className="card" key={entry.name}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  gap: "1rem",
                }}
              >
                <div>
                  <h2 style={{ marginBottom: "0.15rem" }}>
                    {entry.name}{" "}
                    <span className="muted" style={{ fontSize: "0.78rem", fontWeight: 400 }}>
                      v{entry.version}
                    </span>
                  </h2>
                  <p className="muted" style={{ margin: 0, maxWidth: "70ch" }}>
                    {entry.description}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => setExpanded(expanded === entry.name ? null : entry.name)}
                >
                  {expanded === entry.name ? "Hide" : "Parameters"}
                </button>
              </div>

              <div style={{ marginTop: "0.6rem" }}>
                <span className="stat__label">Trades in</span>{" "}
                {entry.allowed_regimes.length > 0 ? (
                  entry.allowed_regimes.map((regime) => (
                    <span key={regime} className="badge badge--muted" style={{ marginRight: 4 }}>
                      {regime.replace(/_/g, " ")}
                    </span>
                  ))
                ) : (
                  <span className="badge badge--muted">any known regime</span>
                )}
              </div>

              {FAILURE_MODES[entry.name] ? (
                <div className="notice notice--warn" style={{ marginTop: "0.7rem" }}>
                  <strong>How it fails:</strong> {FAILURE_MODES[entry.name]}
                </div>
              ) : null}

              {expanded === entry.name ? (
                <div className="table-wrap" style={{ marginTop: "0.7rem" }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Parameter</th>
                        <th className="num">Default</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(entry.default_parameters).map(([key, value]) => (
                        <tr key={key}>
                          <td className="mono">{key}</td>
                          <td className="num">{String(value)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </div>
          ))}
        </div>
      ) : (
        <Empty>Loading the strategy catalogue…</Empty>
      )}

      <Notice kind="info">
        A strategy returns an opinion; it cannot place a trade. Every signal passes through the
        signal engine and the risk manager before any order exists, and there is no bypass.
      </Notice>

      <Disclaimer />
    </>
  );
}
