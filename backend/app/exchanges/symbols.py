"""Splitting a concatenated trading symbol into its base and quote assets.

Bybit and Binance write pairs as ``BTCUSDT``; Coinbase writes ``BTC-USD`` and Crypto.com writes
``BTC_USDT``. Bots configured in the concatenated style must keep working when pointed at a
venue that separates the two, which means guessing where the base ends.

The guess is made only from a known list of quote assets, longest first, and it refuses rather
than guesses when the split is ambiguous. Routing an order to the wrong instrument is a far
worse outcome than declining to route one, so anything uncertain returns ``(None, None)`` and
the caller raises.
"""

from __future__ import annotations

#: Quote assets, longest first so that ``USDT`` wins over ``USD`` for ``BTCUSDT``.
#: Order matters: ``BTCUSDT`` split on ``USD`` would leave a base of ``BTCT``.
QUOTE_ASSETS: tuple[str, ...] = (
    "USDT",
    "USDC",
    "TUSD",
    "BUSD",
    "FDUSD",
    "DAI",
    "USD",
    "EUR",
    "GBP",
    "BTC",
    "ETH",
)


def split_symbol(symbol: str) -> tuple[str | None, str | None]:
    """Split ``BTCUSDT`` into ``("BTC", "USDT")``.

    Returns ``(None, None)`` when the symbol carries an explicit separator (the caller should
    handle that itself), when no known quote asset matches, or when the remaining base would
    be empty.
    """
    text = symbol.strip().upper()
    if not text or "-" in text or "_" in text or "/" in text:
        return None, None

    for quote in QUOTE_ASSETS:
        if not text.endswith(quote):
            continue
        base = text[: -len(quote)]
        if base:
            return base, quote
    return None, None


def join_symbol(symbol: str, separator: str) -> str | None:
    """Rewrite ``BTCUSDT`` as ``BTC<separator>USDT``, or ``None`` if it cannot be split."""
    base, quote = split_symbol(symbol)
    if base is None or quote is None:
        return None
    return f"{base}{separator}{quote}"
