"use client";

import { Shell } from "@/components/Shell";
import {
  Disclaimer,
  Empty,
  ErrorNotice,
  PageHeader,
  Stat,
  StatusBadge,
  useAsync,
} from "@/components/ui";
import { get } from "@/lib/api";

type Health = {
  status: string;
  version: string;
  mode: string;
  uptime_seconds: number;
  checks: Record<
    string,
    { healthy: boolean; detail: string; latency_ms: number | null; required: boolean }
  >;
};

type Metrics = Record<string, string | number>;

type AuditEntry = {
  action: string;
  resource_type: string | null;
  success: boolean;
  ip_address: string | null;
  occurred_at: string;
};

export default function SystemPage() {
  return (
    <Shell>
      <System />
    </Shell>
  );
}

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

function System() {
  const health = useAsync(() => get<Health>("/health"));
  const metrics = useAsync(() => get<Metrics>("/api/v1/metrics").catch(() => ({})));
  const audit = useAsync(() => get<AuditEntry[]>("/api/v1/risk/audit?limit=50"));

  return (
    <>
      <PageHeader
        title="System"
        description="Health of every dependency, runtime metrics, and the account audit trail."
        actions={
          <button
            type="button"
            onClick={() => {
              health.reload();
              metrics.reload();
              audit.reload();
            }}
          >
            Refresh
          </button>
        }
      />

      <ErrorNotice error={health.error} />

      <div className="grid grid--stats" style={{ marginBottom: "1rem" }}>
        <Stat label="Overall" value={<StatusBadge status={health.data?.status ?? "unknown"} />} />
        <Stat label="Mode" value={health.data?.mode ?? "—"} />
        <Stat label="Version" value={health.data?.version ?? "—"} />
        <Stat
          label="Uptime"
          value={health.data ? formatUptime(health.data.uptime_seconds) : "—"}
        />
      </div>

      <div className="card" style={{ marginBottom: "1rem" }}>
        <h2>Dependencies</h2>
        {health.data ? (
          <div className="table-wrap" style={{ border: "none" }}>
            <table>
              <thead>
                <tr>
                  <th>Component</th>
                  <th>State</th>
                  <th>Detail</th>
                  <th className="num">Latency</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(health.data.checks).map(([name, check]) => (
                  <tr key={name}>
                    <td className="mono">{name}</td>
                    <td>
                      <span
                        className={`badge badge--${
                          check.healthy ? "ok" : check.required ? "danger" : "warn"
                        }`}
                      >
                        {check.healthy ? "ok" : check.required ? "failed" : "degraded"}
                      </span>
                    </td>
                    <td style={{ whiteSpace: "normal" }}>{check.detail}</td>
                    <td className="num">
                      {check.latency_ms != null ? `${check.latency_ms.toFixed(1)} ms` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>Loading…</Empty>
        )}
      </div>

      {metrics.data && Object.keys(metrics.data).length > 0 ? (
        <div className="card" style={{ marginBottom: "1rem" }}>
          <h2>Runtime</h2>
          <div className="table-wrap" style={{ border: "none" }}>
            <table>
              <tbody>
                {Object.entries(metrics.data).map(([key, value]) => (
                  <tr key={key}>
                    <td className="muted mono">{key}</td>
                    <td className="num">{String(value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      <div className="card">
        <h2>Audit trail</h2>
        <p className="muted" style={{ fontSize: "0.82rem" }}>
          Append-only. Records sign-ins, credential changes, risk changes, bot control and live
          activation.
        </p>
        {audit.data && audit.data.length > 0 ? (
          <div className="table-wrap" style={{ border: "none" }}>
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Action</th>
                  <th>Resource</th>
                  <th>Result</th>
                  <th>IP</th>
                </tr>
              </thead>
              <tbody>
                {audit.data.map((entry, index) => (
                  <tr key={index}>
                    <td className="mono muted">
                      {new Date(entry.occurred_at).toLocaleString()}
                    </td>
                    <td>{entry.action.replace(/_/g, " ")}</td>
                    <td className="muted">{entry.resource_type ?? "—"}</td>
                    <td>
                      <span className={`badge badge--${entry.success ? "ok" : "danger"}`}>
                        {entry.success ? "ok" : "failed"}
                      </span>
                    </td>
                    <td className="mono muted">{entry.ip_address ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty>No audit entries.</Empty>
        )}
      </div>

      <Disclaimer />
    </>
  );
}
