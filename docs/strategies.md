# Strategies

Each strategy documents its hypothesis, conditions, exits, parameters, restrictions and —
most importantly — the ways it fails. A strategy description that only lists entry rules is
marketing, not documentation.

Common properties, enforced by the framework rather than by each strategy remembering:

* A strategy returns an opinion. It has no reference to an exchange, a portfolio or an order
  manager, and cannot place a trade.
* An entry **must** carry a stop-loss. Without a defined invalidation level, risk-based sizing
  is impossible, so the result object rejects it.
* Insufficient history produces `NO_TRADE` with a reason, never an indicator computed on a
  short window.
* An `UNKNOWN` market regime is a hard no-trade for every strategy.
* Confidence can only ever *reduce* position size, never increase it beyond the configured
  risk per trade.

---

## Trend following

**Hypothesis.** Markets that have moved persistently in one direction continue for longer than
a random walk predicts, because information diffuses slowly and positioning unwinds gradually.
The edge is not in prediction accuracy — trend systems are wrong most of the time — but in the
payoff shape: many small losses funded by a few large wins.

That shape only survives if losers are cut mechanically, which is why the ATR stop is mandatory
rather than optional.

**Market conditions.** Trending markets with sufficient volatility to make a stop meaningful.

**Entry (long; short is the mirror).**

1. Fast EMA above slow EMA — the primary filter.
2. ADX above threshold with +DI above −DI — distinguishes a real trend from two EMAs that
   happen to be ordered inside a range.
3. Price structure making higher highs and higher lows — an independent confirmation that does
   not share inputs with the EMAs, so it catches structural breaks the averages lag.
4. Volume not abnormally thin — a trend advancing on collapsing volume is usually exhaustion.
5. ATR/price inside a band — too low and the stop sits inside the noise; too high and sizing
   collapses to nothing useful.
6. Not over-extended from the fast EMA — where trend entries have the worst expectancy.

**Exit.** ATR stop; take-profit at a reward/risk multiple; opposite EMA cross closes.

**Restrictions.** `TRENDING_BULL`, `TRENDING_BEAR`, `LOW_VOLATILITY`.

**Failure modes.**
* *Whipsaw in a range* — the dominant loss mode. Mitigated by the ADX filter and by refusing
  the `RANGING` regime entirely.
* *Volatility spikes* — a news gap can jump the stop. Mitigated by the volatility ceiling and
  by portfolio exposure caps, not by this strategy.
* *Late entries* — trend following is inherently late. The over-extension filter bounds how
  late, at the cost of missing the fastest moves.

**Key parameters.** `fast_ema_period` (21), `slow_ema_period` (55), `adx_threshold` (25),
`atr_stop_multiplier` (2.0), `risk_reward_ratio` (2.0), `max_extension_atr` (3.0).

---

## Momentum breakout

**Hypothesis.** When price escapes a well-defined range on expanding volume, the participants
who were selling into that range are gone, and the move continues while new participants chase
it. The edge is in the *quality* of the breakout, not in the breakout itself — most breaks of a
channel are noise.

The whole strategy is therefore a series of filters designed to discard low-quality breaks.

**Entry (long).**

1. Close above the Donchian upper channel. The channel **excludes the current bar**, so the
   breakout bar cannot form the level it is being tested against. Without that exclusion,
   "price broke above the channel" is tautological and the backtest looks far better than the
   strategy is.
2. The break exceeds the channel by a minimum fraction of ATR — a one-tick poke is noise.
3. Volume above a multiple of its average.
4. The pre-break channel was reasonably tight — breaking out of an already-wide channel means
   there was no range to break out of.
5. The longer EMA agrees with the direction.
6. Spread and book depth within limits.
7. No failed break in the opposite direction in the recent window, and the breakout bar has not
   given back most of its range by the close.

**Exit.** ATR stop placed beyond the broken level; take-profit at a reward/risk multiple; a
close back inside the shorter channel closes the position — the premise is dead.

**Restrictions.** Trending and volatility regimes; not `RANGING`.

**Failure modes.**
* *False breakouts* — the dominant loss mode, mitigated by filters 2, 3 and 7, and by the
  cooldown that stops repeated re-entry at the same failing level.
* *Gappy illiquid markets* — mitigated by the spread filter and minimum-volume floor.
* *Regime mismatch* — breakouts in a low-volatility drift are usually noise.

**Key parameters.** `channel_period` (20), `min_volume_expansion` (1.5), `min_break_atr` (0.10),
`max_channel_width_atr` (8.0), `max_close_retreat` (0.5).

---

## Mean reversion

**Hypothesis.** Inside a range, price oscillates around a central value because liquidity
providers lean against moves and no information is driving a sustained repricing. Buying
statistically cheap and selling statistically dear inside such a range has positive expectancy.

**The critical qualifier is *inside a range*.** The same logic applied during a trend is
catastrophic: it buys every step down of a decline, producing a long run of small wins followed
by one loss that erases them. This is the classic mean-reversion blow-up, and it is why the
regime filter here is not a refinement but the strategy's central safety mechanism.

**Four independent brakes against trading a trend.**

1. **Hard regime gate** — entries permitted only in `RANGING` and `LOW_VOLATILITY`.
2. **ADX ceiling** — even within a "ranging" classification, a high ADX vetoes entry.
3. **Slope check** — the mid-band must be roughly flat. A rising mean is a trend by another
   name.
4. **Band-walk detection** — several consecutive closes beyond the band means price is
   trending, not stretched.

**Entry (long).** All three of: close below the lower Bollinger band, RSI below oversold, and
z-score below the negative threshold. Requiring three correlated-but-not-identical measures
filters the single-indicator false positives that dominate this style.

**Exit.** Reversion to the mid-band; ATR stop beyond the band, capped so a runaway move cannot
produce an arbitrarily wide trade; RSI crossing back through neutral.

**Failure modes.**
* *Trend disguised as a range* — the four guards above exist for exactly this.
* *Volatility regime shift* — a compressing range that suddenly expands. Mitigated by the ATR
  stop and the drawdown controls.
* *Correlated entries* — many symbols oversold at once during a market-wide sell-off. Handled
  by portfolio exposure limits in the risk manager, not here.

**Key parameters.** `bollinger_period` (20), `bollinger_std` (2.0), `rsi_oversold` (30),
`zscore_threshold` (2.0), `max_adx` (22), `band_walk_bars` (3).

---

## Multi-factor

**Hypothesis.** No single indicator family is reliable across conditions. Trend measures fail
in ranges, oscillators fail in trends, volume confirms nothing alone. Combining weakly
correlated evidence into one score and requiring a high aggregate produces fewer but better
trades than any component alone.

**Factors**, each normalised to `[-1, +1]` where positive is bullish:

| Factor | Measures | Default weight |
|---|---|---|
| Trend | EMA alignment, separation, price vs slow EMA | 0.30 |
| Momentum | RSI displacement from 50, MACD histogram | 0.20 |
| Volume | Expansion, signed by the current bar's direction | 0.15 |
| Volatility | Whether ATR is in a tradable band — a **gate**, not a direction | 0.10 |
| Regime | Agreement between detected regime and candidate direction | 0.15 |
| News | External news score, clamped and weighted | 0.10 |

A signal is emitted only when `|final score| >= entry_threshold` (0.45 by default).

**Two rules stop one factor dominating.**

1. **Direction agreement** — at least `min_agreeing_factors` directional factors must share the
   sign. A 0.7 score built from one extreme factor and four neutral ones is not a consensus.
2. **Volatility is a gate, not a vote** — an untradable volatility environment vetoes the trade
   regardless of the other factors, and it is excluded from the consensus count because it
   carries no direction.

**News never trades alone.** With all technical factors neutral, the maximum score achievable
from news is below the entry threshold *by construction* — the parameter model refuses a
configuration where that is not true.

**Failure modes.**
* *Factor correlation* — trend and momentum agree more often than the weights assume, so a
  "consensus" can be one signal counted twice. Mitigated by requiring agreement across named
  families and keeping trend+momentum below half the total weight.
* *Parameter overfitting* — six weights plus a threshold is a large search space. The
  walk-forward and sensitivity tooling exists specifically for this strategy.

---

## Backtest methodology

Every strategy is evaluated the same way:

1. **Chronological** train/test split — never random. Shuffling time series leaks the future
   into the training set and is the most common way an overfit strategy passes validation.
2. **Walk-forward** across rolling windows, with non-overlapping test portions.
3. **Parameter sensitivity** over a grid, reported as *plateau* or *spike*.
4. **Monte Carlo**: permutation for ordering risk (drawdown distribution) and bootstrap for
   sampling risk (return distribution).

The bar for "robust" is deliberately demanding: a majority of out-of-sample windows profitable,
a positive average out-of-sample return, and at least 30 out-of-sample trades.

## Writing your own

```python
from app.strategies.base import Strategy, StrategyParameters, StrategyResult
from app.strategies.registry import register

class MyParameters(StrategyParameters):
    lookback: int = 20

@register
class MyStrategy(Strategy):
    name = "my_strategy"
    version = "1.0.0"
    description = "..."
    parameters_model = MyParameters
    allowed_regimes = frozenset({MarketRegime.RANGING})

    @property
    def required_history(self) -> int:
        return self.params.lookback + 50

    def _evaluate(self, context) -> StrategyResult:
        ...
```

The framework applies history, regime, cooldown, direction and confidence checks before
`_evaluate` runs, so an individual strategy cannot forget them.

---

**Past performance and backtest results do not guarantee future performance. No strategy
described here is a recommendation, and none guarantees a profit.**
