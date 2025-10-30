from __future__ import annotations

from typing import Dict, Any, Optional

import pandas as pd

from event_bus import EventBus, BUS


_PREV_COMBINED: Optional[pd.DataFrame] = None


def _keyify(df: pd.DataFrame) -> pd.Series:
    k = (
        df["expiration"].astype(str).str.strip()
        + "|" + df["strike"].astype(float).round(2).astype(str)
        + "|" + df["cp"].astype(str).str.upper().str[0]
    )
    return k


def compute_price_market_diffs(cur: pd.DataFrame, prev: Optional[pd.DataFrame]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"new": 0, "removed": 0, "price_moves": 0, "widened_spreads": 0}
    if cur is None or len(cur) == 0:
        return out

    cur = cur.copy()
    cur_key = _keyify(cur)
    cur = cur.assign(_k=cur_key)

    if prev is None or len(prev) == 0:
        out["new"] = len(cur)
        return out

    prev = prev.copy()
    prev_key = _keyify(prev)
    prev = prev.assign(_k=prev_key)

    cur_ix = cur.set_index("_k")
    prev_ix = prev.set_index("_k")

    added = cur_ix.index.difference(prev_ix.index)
    removed = prev_ix.index.difference(cur_ix.index)
    out["new"] = len(added)
    out["removed"] = len(removed)

    # For common keys, compare mdl_theo and market mid
    common = cur_ix.index.intersection(prev_ix.index)
    if len(common) > 0:
        c = cur_ix.loc[common]
        p = prev_ix.loc[common]
        # model theo move
        if {"mdl_theo"} <= set(c.columns) and {"mdl_theo"} <= set(p.columns):
            out["price_moves"] = int((c["mdl_theo"] - p["mdl_theo"]).abs().gt(0).sum())
        # spread widen
        have_spread = {"mkt_bid","mkt_ask"} <= set(c.columns) and {"mkt_bid","mkt_ask"} <= set(p.columns)
        if have_spread:
            c_spread = (c["mkt_ask"] - c["mkt_bid"]).astype(float)
            p_spread = (p["mkt_ask"] - p["mkt_bid"]).astype(float)
            out["widened_spreads"] = int((c_spread - p_spread).gt(0).sum())

    return out


def emit_diffs(bus: EventBus, combined: pd.DataFrame) -> Dict[str, Any]:
    global _PREV_COMBINED
    cur = combined
    prev = _PREV_COMBINED
    diffs = compute_price_market_diffs(cur, prev)
    payload = {"type": "price_market_diffs", "metrics": diffs}
    bus.publish("pricer/diffs", payload)
    _PREV_COMBINED = combined.copy()
    return diffs


