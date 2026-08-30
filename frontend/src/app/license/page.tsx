"use client";

import { useState } from "react";

import { Shell } from "@/components/Shell";
import {
  Disclaimer,
  Empty,
  ErrorNotice,
  Notice,
  PageHeader,
  Stat,
  StatusBadge,
  useAsync,
} from "@/components/ui";
import { get, post } from "@/lib/api";

type License = {
  id: string;
  license_key_masked: string;
  plan: string;
  status: string;
  expires_at: string | null;
  device_limit: number;
  activated_devices: number;
  days_remaining: number | null;
};

type Current = { license: License | null; enforcement: boolean; message: string };

type Subscription = {
  subscription: {
    plan: string;
    status: string;
    current_period_end: string | null;
    cancel_at_period_end: boolean;
  } | null;
  message: string;
};

export default function LicensePage() {
  return (
    <Shell>
      <LicenseView />
    </Shell>
  );
}

function LicenseView() {
  const current = useAsync(() => get<Current>("/api/v1/licenses/current"));
  const subscription = useAsync(() =>
    get<Subscription>("/api/v1/licenses/subscription/current"),
  );
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  async function startTrial() {
    setBusy(true);
    setError(null);
    try {
      await post("/api/v1/licenses/trial");
      current.reload();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  const license = current.data?.license;

  return (
    <>
      <PageHeader
        title="Licence"
        description="Licence status, device allocation and subscription."
        actions={
          <button
            type="button"
            onClick={() => {
              current.reload();
              subscription.reload();
            }}
          >
            Refresh
          </button>
        }
      />

      <ErrorNotice error={error ?? current.error} />

      {current.data ? (
        <Notice kind={license ? "info" : current.data.enforcement ? "error" : "warn"}>
          {current.data.message}
        </Notice>
      ) : null}

      {license ? (
        <div className="grid grid--stats" style={{ marginBottom: "1rem" }}>
          <Stat label="Plan" value={license.plan} />
          <Stat label="Status" value={<StatusBadge status={license.status} />} />
          <Stat
            label="Days remaining"
            value={license.days_remaining ?? "—"}
            hint={
              license.expires_at
                ? `Expires ${new Date(license.expires_at).toLocaleDateString()}`
                : "No expiry"
            }
          />
          <Stat
            label="Devices"
            value={`${license.activated_devices} / ${license.device_limit}`}
          />
          <Stat label="Key" value={<span className="mono">{license.license_key_masked}</span>} />
        </div>
      ) : (
        <div className="card">
          <Empty>
            No licence on this account.
            <div style={{ marginTop: "0.8rem" }}>
              <button type="button" className="primary" disabled={busy} onClick={startTrial}>
                {busy ? "Creating…" : "Start a 14-day trial"}
              </button>
            </div>
          </Empty>
        </div>
      )}

      {subscription.data ? (
        <div className="card" style={{ marginTop: "1rem" }}>
          <h2>Subscription</h2>
          <p className="muted">{subscription.data.message}</p>
          {subscription.data.subscription ? (
            <div className="table-wrap" style={{ border: "none" }}>
              <table>
                <tbody>
                  <tr>
                    <td className="muted">Plan</td>
                    <td>{subscription.data.subscription.plan}</td>
                  </tr>
                  <tr>
                    <td className="muted">Status</td>
                    <td>
                      <StatusBadge status={subscription.data.subscription.status} />
                    </td>
                  </tr>
                  <tr>
                    <td className="muted">Renews</td>
                    <td>
                      {subscription.data.subscription.current_period_end
                        ? new Date(
                            subscription.data.subscription.current_period_end,
                          ).toLocaleDateString()
                        : "—"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          ) : null}
        </div>
      ) : null}

      <Notice kind="info">
        Licence enforcement is off by default on a self-hosted installation. Turning a licensing
        outage into a trading halt — potentially while you hold open positions — would be a
        worse failure than a few days of unlicensed use.
      </Notice>

      <Disclaimer />
    </>
  );
}
