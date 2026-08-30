"""Recognising which assets a headline is about.

A news feed does not tag its articles with ticker symbols. It says "Bitcoin", and the platform
asks about ``BTC``. Without a translation between the two, filtering news by asset returns
almost nothing, and a bot trading BTC never sees the story that moved it.

Matching is on word boundaries, which matters more than it looks. ``TON`` appears inside
"button" and "Toronto"; ``ARB`` inside "arbitrage"; ``OP`` inside "open" and "operation". A
substring match would tag a routine market report with four coins it never mentions, and those
phantom tags propagate into the confidence adjustment of a real trade.
"""

from __future__ import annotations

import re
from functools import lru_cache

#: Ticker -> the words that mean it. The ticker itself is always included.
#:
#: Deliberately conservative: an asset is listed only when its name is distinctive enough to
#: match without a stream of false positives. Short, ambiguous tickers rely on their long name.
ASSET_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("bitcoin", "btc", "xbt"),
    "ETH": ("ethereum", "eth", "ether"),
    "SOL": ("solana", "sol"),
    "XRP": ("xrp", "ripple"),
    "BNB": ("bnb", "binance coin"),
    "ADA": ("cardano", "ada"),
    "DOGE": ("dogecoin", "doge"),
    "AVAX": ("avalanche", "avax"),
    "DOT": ("polkadot", "dot"),
    "POL": ("polygon", "matic", "pol"),
    "LINK": ("chainlink", "link"),
    "LTC": ("litecoin", "ltc"),
    "BCH": ("bitcoin cash", "bch"),
    "TRX": ("tron", "trx"),
    "SHIB": ("shiba inu", "shib"),
    "UNI": ("uniswap", "uni"),
    "ATOM": ("cosmos", "atom"),
    "XLM": ("stellar", "xlm"),
    "NEAR": ("near protocol", "near"),
    "APT": ("aptos", "apt"),
    "ARB": ("arbitrum", "arb"),
    "OP": ("optimism", "op"),
    "TON": ("toncoin", "ton"),
    "SUI": ("sui network", "sui"),
    "PEPE": ("pepe",),
    "USDT": ("tether", "usdt"),
    "USDC": ("usd coin", "usdc"),
}


@lru_cache(maxsize=1)
def _patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    """One compiled word-boundary pattern per asset, built once."""
    compiled = []
    for ticker, aliases in ASSET_ALIASES.items():
        # Longest alias first so 'bitcoin cash' is preferred over a bare 'bitcoin'.
        ordered = sorted(aliases, key=len, reverse=True)
        alternatives = "|".join(re.escape(alias) for alias in ordered)
        compiled.append((ticker, re.compile(rf"\b(?:{alternatives})\b", re.IGNORECASE)))
    return tuple(compiled)


def detect_assets(text: str) -> tuple[str, ...]:
    """Return the tickers a piece of text refers to, in a stable order."""
    if not text:
        return ()
    return tuple(ticker for ticker, pattern in _patterns() if pattern.search(text))


def mentions_asset(text: str, asset: str) -> bool:
    """Whether ``text`` refers to ``asset``, by ticker or by name.

    Falls back to a word-boundary search for the bare ticker when the asset is not one this
    module knows about, so an unlisted coin still matches its own symbol.
    """
    ticker = asset.strip().upper()
    if not ticker or not text:
        return False
    aliases = ASSET_ALIASES.get(ticker)
    if aliases is None:
        return bool(re.search(rf"\b{re.escape(ticker)}\b", text, re.IGNORECASE))
    for candidate, pattern in _patterns():
        if candidate == ticker:
            return bool(pattern.search(text))
    return False
