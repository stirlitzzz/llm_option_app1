from datetime import date
from os import forkpty
from numpy.random import f
import pandas as pd

import dotenv
import os
import ivolatility as ivol


def numify(df, col, dtype="float64"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
    else:
        df[col] = pd.Series(pd.NA, index=df.index, dtype=dtype)


def collect_market_and_resolve(
    symbol: str,
    trade_date: str,                    # 'YYYY-MM-DD'
    requested_legs: list[dict],         # [{"cp":"P","strike":250,"expiry":{"year":2026,"month":1,"iso":None}}, ...]
    getMarketData,                      # ivolatility method: setMethod('/equities/eod/stock-opts-by-param')
    dte_from: int = 0,
    dte_to: int = 760,
    mny_from: int = -90,
    mny_to: int = 90,
) -> dict:
    """
    Returns:
      {
        "ok": bool,
        "dataset": [ {optionId, symbol, cp, expiration, strike, bid, ask, iv, ...} ],
        "issues": [ ... ],
        "chain_ref": df  # cleaned DataFrame for charts
      }
    """
    issues = []
    sym = symbol.upper()

    # 1) Bulk fetch calls & puts
    calls = getMarketData(symbol=sym, tradeDate=trade_date, dteFrom=dte_from, dteTo=dte_to,
                          moneynessFrom=mny_from, moneynessTo=mny_to, cp='C')
    puts  = getMarketData(symbol=sym, tradeDate=trade_date, dteFrom=dte_from, dteTo=dte_to,
                          moneynessFrom=mny_from, moneynessTo=mny_to, cp='P')
    df = pd.concat([calls, puts], ignore_index=True) if not (calls.empty and puts.empty) else pd.DataFrame()
    if df.empty:
        return {"ok": False, "dataset": [], "issues": [{"code":"NO_CHAIN","msg":"no data returned"}], "chain_ref": df}

    # 2) Clean/normalize columns we’ll use
    colmap = {
        "expiration_date":"expiration", "call_put":"cp", "price_strike":"strike",
        "Bid":"bid", "Ask":"ask", "iv":"iv", "option_id":"optionId", "option_symbol":"osym",
        "openinterest":"openInterest", "volume":"volume"
    }
    df = df.rename(columns={k:v for k,v in colmap.items() if k in df.columns}).copy()
    df["cp"]        = df["cp"].str.upper().str[0]
    df["is_settlement"]= df.get("is_settlement", 0).fillna(0)

    for field in ["bid", "ask", "iv", "strike"]:
        numify(df, field)


    for field in ["openInterest", "volume"]:
        numify(df, field, dtype="Int64")
        #df[field] = df[field].fillna(0)

    # 3) Index for fast selection: per (exp, cp) keep a sorted strike list
    by_key = {}
    for (exp, cp), g in df.groupby(["expiration","cp"]):
        g = g.sort_values("strike")
        by_key[(exp, cp)] = g

    # 4) Selection helpers
    def pick_expiry(leg):
        exp = leg.get("expiry") or {}
        iso = exp.get("iso")
        if iso:
            return iso, False  # (chosen, snapped?)
        # fallback month/year → pick nearest listed expiry in that month, else closest overall
        y, m = exp.get("year"), exp.get("month")
        if y and m:
            month_listed = sorted({e for (e,cp) in by_key.keys() if int(e[:4])==y and int(e[5:7])==m})
            if month_listed:
                # prefer the earliest in-month expiry (often weeklies + monthly)
                return month_listed[0], False
        # closest overall to intended 3rd Friday if available
        listed = sorted({e for (e,cp) in by_key.keys()})
        if not listed:
            return None, False
        # crude “closest” by string distance proxy (lex order ≈ chronological for ISO)
        # better: convert to dates and abs delta; left as simple for MVP
        return listed[0], True

    def pick_strike(exp: str, cp: str, target_k: float):
        g = by_key.get((exp, cp))
        if g is None or g.empty:
            return None, None, None, False
        strikes = g["strike"].values
        # nearest by absolute distance
        import numpy as np
        idx = int(np.argmin(np.abs(strikes - target_k)))
        chosen = strikes[idx]
        below = strikes[max(idx-1, 0)]
        above = strikes[min(idx+1, len(strikes)-1)]
        exact = float(chosen) == float(target_k)
        return float(chosen), float(below), float(above), exact

    def best_row(exp: str, cp: str, strike: float):
        g = by_key.get((exp, cp))
        if g is None: return None
        sel = g[g["strike"] == strike]
        if sel.empty: return None
        # prefer non-settlement, then highest OI, then volume
        sel = sel.sort_values(by=["is_settlement","openInterest","volume"], ascending=[True, False, False])
        return sel.iloc[0].to_dict()

    # 5) Resolve each requested leg
    rows = []
    for i, leg in enumerate(requested_legs):
        cp  = str(leg["cp"]).upper()[0]
        k   = float(leg["strike"] if "strike" in leg else leg.get("k"))
        exp, month_snap = pick_expiry(leg)
        if not exp:
            issues.append({"code":"NO_EXP","msg":f"leg {i}: no expiry available"}); continue

        chosen, below, above, exact = pick_strike(exp, cp, k)
        if chosen is None:
            issues.append({"code":"NO_STRIKE","msg":f"leg {i}: no strikes for {cp} @ {exp}"}); continue
        if not exact:
            issues.append({"code":"STRIKE_ADJUSTED","msg":f"leg {i}: {k} -> {chosen} for {cp} @ {exp}", "details":{"below":below,"above":above}})

        r = best_row(exp, cp, chosen)
        if r is None:
            issues.append({"code":"ROW_MISSING","msg":f"leg {i}: missing row {cp} {chosen} @ {exp}"}); continue

        rows.append({
            "optionId": r.get("optionId") or r.get("osym") or f"{sym}:{exp}:{cp}:{int(round(chosen))}",
            "symbol": sym, "cp": cp, "expiration": exp, "strike": float(chosen),
            "bid": r.get("bid"), "ask": r.get("ask"), "iv": r.get("iv"),
            "openInterest": r.get("openInterest"), "volume": r.get("volume"),
            "delta":r.get("delta"),
            "gamma":r.get("gamma"),
            "vega":r.get("vega"),
            "underlying_price":r.get("underlying_price")
        })

    return {"ok": len(rows)==len(requested_legs), "dataset": rows, "issues": issues, "chain_ref": df}


def init_ivol_options_client():
    dotenv.load_dotenv()
    api_key = os.getenv("IVOL_API_KEY")
    if not api_key:
        raise RuntimeError("❌ IVOL_API_KEY not found in .env file or environment.")

    ivol.setLoginParams(apiKey=api_key)
    getMarketData = ivol.setMethod('/equities/eod/stock-opts-by-param')
    return getMarketData