# Risk disclosure — PLACEHOLDER

> Not reviewed by a lawyer. Must be reviewed before commercial use.

## The short version

**You can lose money using this software, including all of it.** Automated trading does not
reduce that risk; it changes how quickly and how consistently it can happen.

## Specific risks

**Market risk.** Prices move against positions. Stop-losses reduce but do not eliminate this: a
gap can jump straight through a stop, and the resulting fill can be materially worse than the
stop price.

**Leverage risk.** Where leverage is available, losses scale with it. A 10% adverse move at 10x
is a total loss of the margin committed.

**Execution risk.** Orders can be rejected, delayed, partially filled, or filled at a worse
price than expected. Exchanges have outages, and they have them most often when markets are
most volatile.

**Model risk.** Strategies are built on assumptions about market behaviour. Those assumptions
stop holding, sometimes permanently and usually without warning. A strategy that worked for
years can stop working and never resume.

**Backtest risk.** Backtests are simulations. They cannot fully capture slippage, latency,
liquidity or the psychological reality of holding a losing position. A profitable backtest is
weak evidence about the future, and an over-optimised one is no evidence at all.

**Technical risk.** Software has bugs. Servers fail. Networks partition. Clocks drift. This
platform is designed so its failure mode is inaction rather than wrong action, but that is a
design intent, not a guarantee.

**Counterparty risk.** Exchanges fail, get hacked, freeze withdrawals, or become inaccessible in
your jurisdiction. Funds held at an exchange are exposed to that exchange.

**Regulatory risk.** Rules governing crypto and automated trading change, sometimes abruptly,
and can affect your ability to trade or withdraw.

## What this software does not do

* It does **not** guarantee any return.
* It does **not** provide investment advice or personal recommendations.
* It does **not** predict prices.
* It does **not** hold, control or have access to your funds.
* It does **not** prevent losses. Its risk controls bound position size and stop trading at
  configured limits; they cannot stop the market moving.

## Before you use real money

* Understand every strategy you enable, including how it loses.
* Run backtests **and** the robustness analysis, and take a failed verdict seriously.
* Paper trade long enough to experience a drawdown.
* Never risk money you cannot afford to lose entirely.
* Set your loss limits before you can lose anything — they are impossible to choose honestly
  while losing.

**Past performance and backtest results do not guarantee future performance.**
