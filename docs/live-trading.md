# Live trading

> Live trading places real orders with real money. Everything below exists because the cost of
> getting it wrong is measured in your capital, not in a stack trace.

## Before you consider it

Do these in order. Skipping steps is the whole reason people lose money on automated systems.

1. **Backtest**, then run `--validate`. An in-sample result on its own is not evidence — it is
   the output of a search over parameters, and a search always finds something.
2. **Read the robustness verdict.** If walk-forward says FAILED, the configuration does not
   survive out-of-sample testing. That is a result, not an obstacle to work around.
3. **Paper trade for long enough to see a drawdown.** Not a week of a favourable trend — long
   enough that the strategy has been wrong repeatedly and you have watched how that feels.
4. **Decide your loss limits before you can lose anything.** What daily loss makes you stop?
   What drawdown makes you turn it off entirely? Write those numbers down first; they are
   impossible to choose honestly while losing.
5. **Size for the worst case.** `risk_per_trade x max_concurrent_positions` is how much of your
   account is on the line simultaneously. Look at that number and ask whether you would accept
   losing it today.

## The nine preflight checks

Every one must pass. A check that cannot be performed counts as a failure — silence is never a
pass.

| # | Check | What it protects against |
|---|---|---|
| 1 | Platform configuration | Live being enabled by accident, or from the UI |
| 2 | User confirmation | Acting without an explicit, typed acknowledgement |
| 3 | Risk configuration | Limits that are incoherent, e.g. more simultaneous risk than the drawdown limit allows |
| 4 | API credentials | A key that does not work — discovered mid-session |
| 5 | **API permissions** | **A key that can withdraw your funds** |
| 6 | Account balance | Sizing against a balance that is not there |
| 7 | Exchange connectivity | A venue that is intermittently unreachable |
| 8 | Clock synchronisation | Signed orders rejected, or applied at the wrong moment |
| 9 | Market data | Trading on stale or missing prices |

Run the preflight without committing to anything:

```http
POST /api/v1/exchange-accounts/{id}/preflight
```

## API key setup

**The single most important step.** Create a key that can trade and cannot move funds.

### Bybit

1. API Management → Create New Key → **System-generated API Keys**.
2. Permissions: enable **Read-Write** for *Trade* only.
3. **Do not enable Withdraw.** Do not enable Transfer.
4. Set an **IP restriction** to your server's address.
5. Set an expiry date and diarise the renewal.

### Binance

1. API Management → Create API.
2. Enable **Enable Reading** and **Enable Spot & Margin Trading**.
3. **Leave "Enable Withdrawals" off.** Leave "Permits Universal Transfer" off.
4. Restrict access to a **trusted IP**.

### Coinbase

1. Settings → API → New API Key (Advanced Trade).
2. Permissions: **View** and **Trade**.
3. **Leave "Transfer" off.** On Coinbase, transfer *is* the withdrawal right — the platform
   reads `can_transfer` from `/key_permissions` and refuses the key if it is set.
4. Add an IP allow-list.

Two Coinbase-specific points:

* **There is no testnet.** The Advanced Trade sandbox was retired, so the rehearsal step below
  cannot be done on Coinbase. The platform refuses to accept a Coinbase account with testnet
  enabled rather than routing real orders while claiming otherwise. Rehearse in paper mode,
  then start live with the smallest size the venue allows.
* **Symbols are written `BTC-USD`.** `BTCUSD` is accepted and converted.

### Crypto.com

1. Dashboard → API Keys → Create.
2. Permissions: **Read** and **Trade**.
3. **Leave "Withdraw" off.**
4. Add an IP allow-list.

Crypto.com publishes no endpoint that reports a key's permissions, so unlike the other three
venues this cannot simply be read back. The platform establishes it by *probing*: it calls a
withdrawal-scoped endpoint, and a key that reaches it is refused. A key refused by that
endpoint passes. If the probe can neither succeed nor be cleanly refused — a network failure,
an unrecognised error code — the platform **refuses to connect**, because an unanswered
question about withdrawal rights is not the same as a safe answer.

Symbols are written `BTC_USDT`. `BTCUSDT` is accepted and converted.

The platform reads these permissions and **refuses** a key that can withdraw or transfer,
before it is stored. If you see that rejection, the key is genuinely dangerous — make a new one
rather than looking for a way around it.

## Enabling live mode

Two changes on the host, neither reachable from the interface:

```bash
# .env
TRADING_MODE=live
LIVE_TRADING_ENABLED=true
EXCHANGE=bybit          # or binance
```

Restart, then in the application:

1. Connect the exchange account (credentials are validated and encrypted before storage).
2. Run the preflight and read every line.
3. Activate live trading, typing the confirmation phrase exactly:

   ```
   I UNDERSTAND THE RISKS
   ```

4. Acknowledge that no return is guaranteed.

Activation is recorded in the audit log with your identity and the time.

## First live session

* **Start on testnet.** Both adapters support it. Verify that fills, balances and positions
  match what the platform believes before touching mainnet.
* **Start at the venue minimum.** The first live order's purpose is to prove the plumbing, not
  to make money.
* **Watch the first fills by hand.** Compare the venue's fill price and fee against the
  platform's record. A mismatch here means a calibration problem that would compound silently.
* **Keep the position count at 1** until you have seen a full cycle including a stop-loss.

## What stops trading automatically

| Trigger | Effect | Clears |
|---|---|---|
| Daily loss limit | Kill switch, no new orders | Manual reset |
| Weekly loss limit | Kill switch | Manual reset |
| Max drawdown | Kill switch | Manual reset |
| Reconciliation mismatch | Halt | Manual, after investigating |
| State corruption | Halt | Manual |
| Repeated API failures | Kill switch | Auto once the venue recovers |
| Stale market data | Kill switch | Auto once data resumes |
| Clock drift | Kill switch | Auto once the clock is corrected |
| Loss streak | Blocks new entries | Next winning trade |
| Manual emergency stop | Kill switch | Manual reset |

**None of these close your positions.** Stopping and liquidating are different decisions, and
an emergency is frequently the worst possible moment to be a forced seller. Closing positions
is always something you ask for explicitly.

## Emergency stop

```http
POST /api/v1/bots/{id}/emergency-stop
{"close_positions": false, "note": "why"}
```

Trips the kill switch and cancels resting orders. Set `close_positions: true` only when you
have decided that exiting now is better than holding through whatever is happening.

## Resuming after a halt

1. **Understand why it halted.** `GET /api/v1/risk/events` and the bot's event feed.
2. If it was a reconciliation mismatch, compare the platform's positions against the venue's
   directly, and work out where they diverged. Do not adopt one side without knowing why.
3. Reset the kill switch — it requires your identity and is audited:
   ```http
   POST /api/v1/risk/kill-switch/{bot_id}/reset
   ```
4. Restart the bot separately. Clearing the switch does not resume trading on its own.

## Ongoing operation

* **Re-validate keys periodically.** A user can widen a key's permissions on the venue after
  connecting it; `POST /api/v1/exchange-accounts/{id}/validate` is how the platform notices.
* **Keep the clock synchronised.** Run NTP. Drift causes rejected orders at best.
* **Back up the database and `ENCRYPTION_KEY` separately.** The key is not in the database on
  purpose; losing it means reconnecting every exchange account.
* **Watch fees as a share of gross profit.** A strategy that was marginal in backtest becomes
  unprofitable when the real fee tier is worse than assumed.

## If something goes wrong

1. Emergency stop the bot (positions stay open).
2. Check the venue's own interface for the real position and order state.
3. Read the audit log and the bot event feed.
4. Decide about the positions deliberately, from the venue if necessary.
5. Only then work out what to change.

The platform is designed so that its failure mode is *inaction* — no new orders — rather than
wrong action. That gives you time to look.

---

**Past performance and backtest results do not guarantee future performance. This software does
not provide investment advice. You are responsible for every order it places on your behalf.**
