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
  useAsync,
} from "@/components/ui";
import { currentMode, del, get, post } from "@/lib/api";

type Account = {
  id: string;
  name: string;
  exchange: string;
  is_testnet: boolean;
  api_key_masked: string | null;
  api_key_fingerprint: string | null;
  can_withdraw: boolean;
  is_validated: boolean;
  validated_at: string | null;
  last_error: string | null;
  is_active: boolean;
};

export default function ExchangeAccountsPage() {
  return (
    <Shell>
      <ExchangeAccounts />
    </Shell>
  );
}

function ExchangeAccounts() {
  const accounts = useAsync(() => get<Account[]>("/api/v1/exchange-accounts"));
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [message, setMessage] = useState<string | null>(null);

  async function revalidate(id: string) {
    setError(null);
    setMessage(null);
    try {
      const result = await post<{ valid: boolean; safe: boolean | null; message: string }>(
        `/api/v1/exchange-accounts/${id}/validate`,
      );
      setMessage(result.message);
      accounts.reload();
    } catch (caught) {
      setError(caught);
    }
  }

  async function remove(id: string) {
    setError(null);
    try {
      await del(`/api/v1/exchange-accounts/${id}`);
      accounts.reload();
    } catch (caught) {
      setError(caught);
    }
  }

  return (
    <>
      <PageHeader
        title="Exchange accounts"
        description="Connect an exchange with a trade-only API key. Live trading additionally requires host configuration that cannot be changed from here."
        actions={
          <button type="button" className="primary" onClick={() => setAdding(!adding)}>
            {adding ? "Cancel" : "Connect account"}
          </button>
        }
      />

      <ErrorNotice error={error ?? accounts.error} />
      {message ? <Notice kind="info">{message}</Notice> : null}

      <Notice kind="warn">
        <strong>Your API key must not have withdrawal permission.</strong> The platform reads the
        key&apos;s permissions and refuses it outright if it can move funds — before storing it.
        A key that can withdraw turns any bug anywhere in the stack into a theft.
      </Notice>

      {adding ? (
        <ConnectForm
          onDone={() => {
            setAdding(false);
            accounts.reload();
          }}
        />
      ) : null}

      {accounts.data && accounts.data.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Exchange</th>
                <th>Network</th>
                <th>API key</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {accounts.data.map((account) => (
                <tr key={account.id}>
                  <td>{account.name}</td>
                  <td className="mono">{account.exchange}</td>
                  <td>
                    <span className={`badge badge--${account.is_testnet ? "ok" : "warn"}`}>
                      {account.is_testnet ? "testnet" : "mainnet"}
                    </span>
                  </td>
                  <td className="mono muted">
                    {account.api_key_masked ?? "—"}
                    {account.api_key_fingerprint ? (
                      <div style={{ fontSize: "0.7rem" }}>
                        fp {account.api_key_fingerprint}
                      </div>
                    ) : null}
                  </td>
                  <td>
                    {account.can_withdraw ? (
                      <span className="badge badge--danger">can withdraw — disabled</span>
                    ) : account.is_validated ? (
                      <span className="badge badge--ok">trade only</span>
                    ) : (
                      <span className="badge badge--warn">unvalidated</span>
                    )}
                    {account.last_error ? (
                      <div
                        className="muted"
                        style={{ fontSize: "0.72rem", whiteSpace: "normal" }}
                      >
                        {account.last_error}
                      </div>
                    ) : null}
                  </td>
                  <td>
                    <div className="button-row">
                      <button type="button" onClick={() => revalidate(account.id)}>
                        Re-validate
                      </button>
                      <ConfirmAction
                        phrase={account.name}
                        label="Remove"
                        description={
                          <>
                            Remove <strong>{account.name}</strong> and delete its stored
                            credentials. This cannot be undone; you would need to create a new
                            API key to reconnect.
                          </>
                        }
                        onConfirm={() => remove(account.id)}
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>No exchange accounts connected. Paper trading needs none.</Empty>
      )}

      <LiveActivationPanel accounts={accounts.data ?? []} />

      <Disclaimer />
    </>
  );
}

/**
 * What the platform knows about each venue before you connect to it.
 *
 * `testnet` is the load-bearing field. Coinbase Advanced Trade has no sandbox, and the backend
 * refuses to construct an adapter with testnet enabled rather than quietly routing "testnet"
 * orders to the production venue. Surfacing that here means the user learns it while choosing,
 * instead of as a rejected form submission.
 */
const VENUES = [
  { id: "bybit", label: "Bybit", testnet: true, symbols: "BTCUSDT" },
  { id: "binance", label: "Binance", testnet: true, symbols: "BTCUSDT" },
  { id: "coinbase", label: "Coinbase", testnet: false, symbols: "BTC-USD" },
  { id: "cryptocom", label: "Crypto.com", testnet: true, symbols: "BTC_USDT" },
] as const;

function venue(id: string) {
  return VENUES.find((v) => v.id === id) ?? VENUES[0];
}

function ConnectForm({ onDone }: { onDone: () => void }) {
  const [form, setForm] = useState({
    name: "",
    exchange: "bybit",
    api_key: "",
    api_secret: "",
    testnet: true,
  });
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const selected = venue(form.exchange);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await post("/api/v1/exchange-accounts", form);
      onDone();
    } catch (caught) {
      setError(caught);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="card" onSubmit={submit} style={{ marginBottom: "1rem" }}>
      <h2>Connect an exchange</h2>
      <ErrorNotice error={error} />

      <div className="field">
        <label htmlFor="acct-name">Name</label>
        <input
          id="acct-name"
          required
          value={form.name}
          onChange={(e) => setForm({ ...form, name: e.target.value })}
        />
      </div>
      <div className="field">
        <label htmlFor="acct-exchange">Exchange</label>
        <select
          id="acct-exchange"
          value={form.exchange}
          onChange={(e) => {
            const next = venue(e.target.value);
            // Selecting a venue without a sandbox must clear the flag, not leave it set and
            // let the request fail.
            setForm({
              ...form,
              exchange: e.target.value,
              testnet: next.testnet ? form.testnet : false,
            });
          }}
        >
          {VENUES.map((v) => (
            <option key={v.id} value={v.id}>
              {v.label}
            </option>
          ))}
        </select>
        <div className="field__hint">
          Symbols on this venue are written like <code>{selected.symbols}</code>.
        </div>
      </div>
      <div className="field">
        <label htmlFor="acct-key">API key</label>
        <input
          id="acct-key"
          required
          autoComplete="off"
          value={form.api_key}
          onChange={(e) => setForm({ ...form, api_key: e.target.value })}
        />
      </div>
      <div className="field">
        <label htmlFor="acct-secret">API secret</label>
        <input
          id="acct-secret"
          type="password"
          required
          autoComplete="off"
          value={form.api_secret}
          onChange={(e) => setForm({ ...form, api_secret: e.target.value })}
        />
        <div className="field__hint">
          Encrypted before storage and never returned by any endpoint.
        </div>
      </div>
      <label style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
        <input
          type="checkbox"
          style={{ width: "auto" }}
          checked={form.testnet}
          disabled={!selected.testnet}
          onChange={(e) => setForm({ ...form, testnet: e.target.checked })}
        />
        Testnet
      </label>
      <div className="field__hint" style={{ marginBottom: "0.7rem" }}>
        {selected.testnet ? (
          <>
            Start on testnet. Verify that fills and balances match what the platform believes
            before pointing this at real money.
          </>
        ) : (
          <>
            <strong>{selected.label} has no testnet.</strong> There is no way to rehearse
            against fake balances here, so any order this account places is a real one. Use
            paper mode to rehearse first.
          </>
        )}
      </div>

      <button type="submit" className="primary" disabled={busy}>
        {busy ? "Validating with the exchange…" : "Connect"}
      </button>
    </form>
  );
}

function LiveActivationPanel({ accounts }: { accounts: Account[] }) {
  const { isLive } = currentMode();

  if (accounts.length === 0) return null;

  if (!isLive) {
    return (
      <Notice kind="info">
        <strong>Live trading is disabled on this installation.</strong> It cannot be enabled
        from this interface. An operator must set <code>LIVE_TRADING_ENABLED=true</code> and{" "}
        <code>TRADING_MODE=live</code> in the host environment and restart, then complete the
        nine-point preflight. See <code>docs/live-trading.md</code>.
      </Notice>
    );
  }

  return (
    <Notice kind="error">
      <strong>This installation is configured for live trading.</strong> Activating an account
      for live use requires a passing preflight and the exact confirmation phrase. Start at the
      venue minimum size and verify the first fills by hand.
    </Notice>
  );
}
