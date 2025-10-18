"""
Drop-in data-loading module for your multiagent chat app.

- Polygon.io: OHLC aggregates (stocks, ETFs, indexes).
- iVolatility: Generic REST hook (you fill the actual endpoint + params).
- Yield curve: Load local CSV/JSON and provide simple linear interpolation.

Usage
-----
export POLYGON_API_KEY=...  # required for Polygon

python market_data_loader.py --ticker SPY --from 2024-01-01 --to 2024-02-01 \
    --yield-file ./data/ust_curve.csv

The iVolatility call is left as a TODO because orgs use different endpoints/packages.
Fill in IVOL_ENDPOINT / IVOL_PARAMS in `fetch_ivolatility`.
"""

from __future__ import annotations
import os
import asyncio
import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import httpx

# ----------------------------
# Small utilities
# ----------------------------

class ConfigError(RuntimeError):
    pass


def _env(key: str) -> str:
    v = os.getenv(key)
    if not v:
        raise ConfigError(f"Missing required env var: {key}")
    return v


async def _get_json(url: str, params: Dict[str, Any] | None = None, headers: Dict[str, str] | None = None, retries: int = 3, timeout: float = 30.0) -> Dict[str, Any]:
    """HTTP GET with tiny retry + backoff."""
    backoff = 1.0
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(retries):
            try:
                resp = await client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2
    return {}


# ----------------------------
# Polygon.io
# ----------------------------

@dataclass
class PolygonBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


async def fetch_polygon_agg(
    ticker: str,
    start: str,
    end: str,
    multiplier: int = 1,
    timespan: str = "day",
    adjusted: bool = True,
    limit: int = 50000,
) -> pd.DataFrame:
    """Fetch aggregate bars from Polygon.io.

    Docs: https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksTicker__range__multiplier____timespan___from___to
    """
    api_key = _env("POLYGON_API_KEY")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {
        "adjusted": str(adjusted).lower(),
        "sort": "asc",
        "limit": limit,
        "apiKey": api_key,
    }
    data = await _get_json(url, params=params)
    if data.get("status") != "OK":
        raise RuntimeError(f"Polygon error: {data}")

    rows = []
    for r in data.get("results", []):
        rows.append(
            {
                "timestamp": datetime.utcfromtimestamp(r["t"] / 1000.0),
                "open": r["o"],
                "high": r["h"],
                "low": r["l"],
                "close": r["c"],
                "volume": r.get("v", np.nan),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df.set_index("timestamp", inplace=True)
    return df

async def fetch_polygon_dividends(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 1000,
    client: Any | None = None,
) -> pd.DataFrame:
    """Fetch dividend history for a ticker from Polygon.

    If a Polygon SDK `client` (e.g., `polygon.RESTClient`) is provided, uses it.
    Otherwise falls back to a direct HTTPS call using `POLYGON_API_KEY`.

    Returns a tidy DataFrame sorted by `ex_dividend_date`.
    """
    cols = [
        "id",
        "cash_amount",
        "currency",
        "declaration_date",
        "dividend_type",
        "ex_dividend_date",
        "frequency",
        "pay_date",
        "record_date",
        "ticker",
    ]

    if client is not None:
        kwargs = dict(ticker=ticker, order="asc", sort="ex_dividend_date", limit=limit)
        if start:
            kwargs["ex_dividend_date_gte"] = start
        if end:
            kwargs["ex_dividend_date_lte"] = end
        try:
            rows = [d for d in client.list_dividends(**kwargs)]
            df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Polygon SDK list_dividends failed: {e}")
    else:
        api_key = _env("AWS_SECRET")
        url = "https://api.polygon.io/v3/reference/dividends"
        params = {
            "ticker": ticker,
            "order": "asc",
            "sort": "ex_dividend_date",
            "limit": limit,
            "apiKey": api_key,
        }
        if start:
            # v3 style filters use dotted field names
            params["ex_dividend_date.gte"] = start
        if end:
            params["ex_dividend_date.lte"] = end
        data = await _get_json(url, params=params)
        results = data.get("results", [])
        df = pd.DataFrame(results) if results else pd.DataFrame(columns=cols)

    # Normalize types
    for col in ["ex_dividend_date", "pay_date", "declaration_date", "record_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
    if "cash_amount" in df.columns:
        df["cash_amount"] = pd.to_numeric(df["cash_amount"], errors="coerce").astype(float)

    if not df.empty and "ex_dividend_date" in df.columns:
        df = df.sort_values("ex_dividend_date").reset_index(drop=True)

    # Return with consistent columns if possible
    return df[cols] if set(cols).issubset(df.columns) else df

# ----------------------------
# iVolatility (generic hook)
# ----------------------------

async def fetch_ivolatility(
    *,
    symbol: str,
    start: str,
    end: str,
    session: Optional[httpx.AsyncClient] = None,
    endpoint: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Generic iVolatility fetch.

    NOTE: iVolatility offers multiple products with different endpoints/auth methods
    (web service, CSV download, etc). To avoid guessing, this function is a thin
    adapter—set `endpoint` and `extra_params` per your contract. Example shape:

    endpoint = "https://api.ivolatility.com/svc/ivx/history"
    extra_params = {"symbol": symbol, "from": start, "to": end, "api_key": os.getenv("IVOL_API_KEY")}

    Expect a JSON or CSV; we auto-detect and return a tidy DataFrame with a datetime index.
    """
    url = endpoint or os.getenv("IVOL_ENDPOINT")
    if not url:
        raise ConfigError("Set IVOL_ENDPOINT env var, or pass endpoint=...")

    params = {"symbol": symbol, "from": start, "to": end}
    if extra_params:
        params.update(extra_params)

    # Try JSON first
    try:
        payload = await _get_json(url, params=params)
        # Heuristic normalization – adapt to your actual fields
        # Looking for fields like: date, ivx, iv30, iv60, iv90, etc.
        records = payload.get("data") or payload.get("results") or payload
        if isinstance(records, list) and records and isinstance(records[0], dict):
            df = pd.DataFrame(records)
        else:
            # Fallback to CSV if JSON shape is not as expected
            raise ValueError("Unexpected JSON shape; will try CSV parse")
    except Exception:
        # CSV fallback
        # Re-fetch as raw text
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            df = pd.read_csv(pd.compat.StringIO(resp.text))

    # Normalize common date column names
    for col in ["date", "Date", "trade_date", "timestamp"]:
        if col in df.columns:
            df.rename(columns={col: "date"}, inplace=True)
            break
    if "date" not in df.columns:
        raise RuntimeError("Could not find a date column in iVolatility response. Please map it explicitly.")

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    return df


# ----------------------------
# iVolatility (official SDK) adapter
# ----------------------------

# Optional dependency: ivolatility SDK. If not installed, these helpers stay inert.
try:
    import ivolatility as ivol  # type: ignore
    _IVOL_SDK = True
except Exception:  # pragma: no cover
    ivol = None
    _IVOL_SDK = False

try:
    import dotenv  # type: ignore
except Exception:  # pragma: no cover
    dotenv = None


def _numify(df: pd.DataFrame, col: str, dtype: str = "float64"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
    else:
        df[col] = pd.Series(pd.NA, index=df.index, dtype=dtype)


def init_ivol_options_client():
    """Initialize iVolatility SDK client and return a bound `getMarketData` function.

    Requires `IVOL_API_KEY` in env (optionally loaded via python-dotenv if present).
    """
    if not _IVOL_SDK:
        raise RuntimeError("ivolatility SDK not installed. `pip install ivolatility python-dotenv`")
    if dotenv:
        try:
            dotenv.load_dotenv()
        except Exception:
            pass
    api_key = os.getenv("IVOL_API_KEY")
    if not api_key:
        raise ConfigError("IVOL_API_KEY not found in environment.")
    ivol.setLoginParams(apiKey=api_key)
    return ivol.setMethod('/equities/eod/stock-opts-by-param')


def _third_friday(y: int, m: int) -> datetime:
    # Find the 3rd Friday of (y,m)
    d = datetime(y, m, 1)
    # weekday(): Monday=0 ... Sunday=6; we want Friday=5
    first_friday_delta = (4 - d.weekday()) % 7  # 0-based Friday index = 4
    first_friday = d.replace(day=1 + first_friday_delta)
    third_friday = first_friday.replace(day=first_friday.day + 14)
    return third_friday


async def collect_market_and_resolve(
    symbol: str,
    trade_date: str,                    # 'YYYY-MM-DD'
    requested_legs: list[dict],         # [{"cp":"P","strike":250,"expiry":{"year":2026,"month":1,"iso":None}}, ...]
    getMarketData=None,                 # ivolatility method: setMethod('/equities/eod/stock-opts-by-param')
    dte_from: int = 0,
    dte_to: int = 760,
    mny_from: int = -90,
    mny_to: int = 90,
) -> dict:
    """Fetch iVolatility option chains and resolve arbitrary legs to concrete rows.

    Returns
    -------
    dict with keys:
      ok: bool
      dataset: list[dict]   # simplified per-leg rows
      issues: list[dict]    # adjustments, missing data flags
      chain_ref: DataFrame  # cleaned chain for downstream charts
    """
    if getMarketData is None:
        getMarketData = init_ivol_options_client()

    sym = symbol.upper()

    def _fetch(cp: str) -> pd.DataFrame:
        return getMarketData(
            symbol=sym,
            tradeDate=trade_date,
            dteFrom=dte_from,
            dteTo=dte_to,
            moneynessFrom=mny_from,
            moneynessTo=mny_to,
            cp=cp,
        )

    # The SDK is sync; run in threads to parallelize C/P
    calls_df, puts_df = await asyncio.gather(
        asyncio.to_thread(_fetch, 'C'),
        asyncio.to_thread(_fetch, 'P'),
    )

    df = pd.concat([calls_df, puts_df], ignore_index=True) if not (calls_df.empty and puts_df.empty) else pd.DataFrame()
    if df.empty:
        return {"ok": False, "dataset": [], "issues": [{"code": "NO_CHAIN", "msg": "no data returned"}], "chain_ref": df}

    # ---- Clean / normalize ----
    colmap = {
        "expiration_date": "expiration",
        "call_put": "cp",
        "price_strike": "strike",
        "Bid": "bid",
        "Ask": "ask",
        "iv": "iv",
        "option_id": "optionId",
        "option_symbol": "osym",
        "openinterest": "openInterest",
        "volume": "volume",
        "underlying_price": "underlying_price",
        "delta": "delta",
        "gamma": "gamma",
        "vega": "vega",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns}).copy()

    # Normalize
    if "cp" in df.columns:
        df["cp"] = df["cp"].astype(str).str.upper().str[0]
    df["is_settlement"] = df.get("is_settlement", 0).fillna(0).astype("Int64")

    for field in ["bid", "ask", "iv", "strike", "delta", "gamma", "vega", "underlying_price"]:
        _numify(df, field)
    for field in ["openInterest", "volume"]:
        _numify(df, field, dtype="Int64")

    # Mid/mark price
    df["mark"] = np.where(np.isfinite(df["bid"]) & np.isfinite(df["ask"]), (df["bid"] + df["ask"]) / 2.0,
                           np.where(np.isfinite(df["bid"]), df["bid"], df["ask"]))

    # Ensure expiration is ISO date string
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.strftime("%Y-%m-%d")

    # Group for fast selection
    by_key: dict[tuple[str, str], pd.DataFrame] = {}
    for (exp, cp), g in df.groupby(["expiration", "cp"], dropna=True):
        by_key[(exp, cp)] = g.sort_values("strike")

    # Selection helpers
    def pick_expiry(leg: dict) -> tuple[Optional[str], bool]:
        exp = leg.get("expiry") or {}
        iso = exp.get("iso")
        if iso:
            return iso, False
        y, m = exp.get("year"), exp.get("month")
        listed = sorted({e for (e, _cp) in by_key.keys()})
        if not listed:
            return None, False
        if y and m:
            # choose expiry closest to the 3rd Friday of that month
            target = _third_friday(int(y), int(m))
            listed_dt = [datetime.strptime(e, "%Y-%m-%d") for e in listed]
            best = min(listed_dt, key=lambda d: abs((d - target).days))
            return best.strftime("%Y-%m-%d"), False
        # Fallback: earliest listed expiry
        return listed[0], True

    def pick_strike(exp: str, cp: str, target_k: float) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
        g = by_key.get((exp, cp))
        if g is None or g.empty:
            return None, None, None, False
        strikes = g["strike"].astype(float).values
        idx = int(np.argmin(np.abs(strikes - target_k)))
        chosen = float(strikes[idx])
        below = float(strikes[max(idx - 1, 0)])
        above = float(strikes[min(idx + 1, len(strikes) - 1)])
        exact = np.isclose(chosen, float(target_k))
        return chosen, below, above, exact

    def best_row(exp: str, cp: str, strike: float) -> Optional[dict]:
        g = by_key.get((exp, cp))
        if g is None:
            return None
        sel = g[g["strike"] == strike]
        if sel.empty:
            return None
        sel = sel.sort_values(by=["is_settlement", "openInterest", "volume"], ascending=[True, False, False])
        return sel.iloc[0].to_dict()

    issues: list[dict] = []
    rows: list[dict] = []

    for i, leg in enumerate(requested_legs):
        cp = str(leg.get("cp", "")).upper()[:1]
        k = float(leg.get("strike") if "strike" in leg else leg.get("k"))
        exp, month_snap = pick_expiry(leg)
        if not exp:
            issues.append({"code": "NO_EXP", "msg": f"leg {i}: no expiry available"})
            continue
        chosen, below, above, exact = pick_strike(exp, cp, k)
        if chosen is None:
            issues.append({"code": "NO_STRIKE", "msg": f"leg {i}: no strikes for {cp} @ {exp}"})
            continue
        if not exact:
            issues.append({"code": "STRIKE_ADJUSTED", "msg": f"leg {i}: {k} -> {chosen} for {cp} @ {exp}", "details": {"below": below, "above": above}})

        r = best_row(exp, cp, chosen)
        if r is None:
            issues.append({"code": "ROW_MISSING", "msg": f"leg {i}: missing row {cp} {chosen} @ {exp}"})
            continue

        rows.append({
            "optionId": r.get("optionId") or r.get("osym") or f"{sym}:{exp}:{cp}:{int(round(chosen))}",
            "symbol": sym,
            "cp": cp,
            "expiration": exp,
            "strike": float(chosen),
            "bid": r.get("bid"),
            "ask": r.get("ask"),
            "mark": r.get("mark"),
            "iv": r.get("iv"),
            "openInterest": r.get("openInterest"),
            "volume": r.get("volume"),
            "delta": r.get("delta"),
            "gamma": r.get("gamma"),
            "vega": r.get("vega"),
            "underlying_price": r.get("underlying_price"),
            "snapped_expiry": bool(month_snap),
            "snapped_strike": not exact,
        })

    return {
        "ok": len(rows) == len(requested_legs),
        "dataset": rows,
        "issues": issues,
        "chain_ref": df,
    }

# ----------------------------
# Yield curve loader + interpolator
# ----------------------------

TENOR_ORDER = [
    "1M", "2M", "3M", "4M", "6M",
    "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y",
]


def _tenor_to_years(label: str) -> float:
    label = label.upper().strip()
    if label.endswith("M"):
        return float(label[:-1]) / 12.0
    if label.endswith("Y"):
        return float(label[:-1])
    raise ValueError(f"Unrecognized tenor: {label}")


def load_yield_curve(path: str) -> pd.DataFrame:
    """Load a yield curve file.

    Accepted formats:
      - CSV with columns: date, tenor (e.g., 3M, 2Y, 10Y), rate (in % or decimal)
      - JSON as list of {date, tenor, rate}

    Returns a pivoted DataFrame indexed by date with tenor columns in years.
    """
    if path.lower().endswith(".csv"):
        raw = pd.read_csv(path)
    elif path.lower().endswith(".json"):
        raw = pd.read_json(path)
    else:
        raise ValueError("Only .csv or .json supported for yield curve")

    # Normalize columns
    cols = {c.lower(): c for c in raw.columns}
    need = {"date", "tenor", "rate"}
    if not need.issubset(set(k.lower() for k in raw.columns)):
        raise ValueError(f"Yield file must have columns {need}; saw {set(raw.columns)}")

    raw.rename(columns={cols.get("date"): "date", cols.get("tenor"): "tenor", cols.get("rate"): "rate"}, inplace=True)
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)

    # Convert rate to decimal if it looks like percent
    if raw["rate"].max() > 2.0:
        raw["rate"] = raw["rate"] / 100.0

    raw["years"] = raw["tenor"].apply(_tenor_to_years)
    pivot = raw.pivot_table(index="date", columns="years", values="rate", aggfunc="last").sort_index()
    pivot.columns.name = "years"
    return pivot


def make_yield_interpolator(curve_df: pd.DataFrame):
    """Return a function tenor_years -> rate using linear interpolation on the latest row."""
    if curve_df.empty:
        raise ValueError("Empty yield curve DataFrame")
    latest = curve_df.iloc[-1]
    xs = np.array(latest.index.tolist(), dtype=float)
    ys = latest.values.astype(float)

    def interp(tenor_years: float) -> float:
        return float(np.interp(tenor_years, xs, ys))

    return interp


# ----------------------------
# QuantLib helpers (dates & dividends)
# ----------------------------

try:
    import QuantLib as ql  # type: ignore
    _HAVE_QL = True
except Exception:  # pragma: no cover
    ql = None
    _HAVE_QL = False


def _ensure_ql_import():
    if not _HAVE_QL:
        raise ImportError("QuantLib not installed. `pip install QuantLib`")


def _to_ql_date(ts) -> "ql.Date":
    _ensure_ql_import()
    if isinstance(ts, pd.Timestamp):
        dt = ts.to_pydatetime().date()
    else:
        dt = pd.to_datetime(ts).to_pydatetime().date()
    return ql.Date(dt.day, dt.month, dt.year)


def _calendar_by_name(name: str | None) -> "ql.Calendar":
    _ensure_ql_import()
    if not name:
        return ql.NullCalendar()
    n = name.lower().strip()
    if n in {"target", "eur"}:
        return ql.TARGET()
    if n in {"us", "usa", "unitedstates", "nyse"}:
        return ql.UnitedStates()
    if n in {"uk", "unitedkingdom"}:
        return ql.UnitedKingdom()
    # Fallback
    return ql.NullCalendar()


def _convention_by_name(name: str | None) -> "ql.BusinessDayConvention":
    _ensure_ql_import()
    m = {
        None: ql.Following,
        "following": ql.Following,
        "modifiedfollowing": ql.ModifiedFollowing,
        "preceding": ql.Preceding,
        "modifiedpreceding": ql.ModifiedPreceding,
        "unadjusted": ql.Unadjusted,
    }
    key = None if name is None else name.lower().replace("_", "")
    return m.get(key, ql.Following)


def to_ql_dates(
    dates_like, *, calendar: str | None = None, adjust: bool = False, convention: str | None = "Following",
) -> list:
    """Convert an iterable of date-like values to a list[ql.Date].

    Parameters
    ----------
    dates_like : Iterable of strings / datetime / pandas Timestamps
    calendar : Optional calendar name ("TARGET", "UnitedStates", "Null") used when `adjust=True`.
    adjust : If True, business-day adjust each date using the given calendar & convention.
    convention : Business-day convention to use when `adjust=True`.
    """
    _ensure_ql_import()
    qds = [_to_ql_date(d) for d in dates_like]
    if adjust:
        cal = _calendar_by_name(calendar)
        conv = _convention_by_name(convention)
        qds = [cal.adjust(d, conv) for d in qds]
    # de-dup & sort by serial number
    seen = set()
    uniq = []
    for d in qds:
        sn = d.serialNumber()
        if sn not in seen:
            seen.add(sn)
            uniq.append(d)
    uniq.sort(key=lambda d: d.serialNumber())
    return uniq


def to_ql_fixed_dividends(
    df: pd.DataFrame,
    *,
    date_col: str = "ex_dividend_date",
    amount_col: str = "cash_amount",
    calendar: str | None = None,
    adjust: bool = False,
    convention: str | None = "Following",
) -> list:
    """Return a QuantLib DividendSchedule from a DataFrame (list[ql.FixedDividend])."""
    _ensure_ql_import()
    dates = to_ql_dates(df.dropna(subset=[date_col])[date_col], calendar=calendar, adjust=adjust, convention=convention)
    amounts = [float(x) for x in df.loc[df[date_col].notna(), amount_col].tolist()]
    # align lengths safely (skip if mismatch after roll/dup removal)
    n = min(len(dates), len(amounts))
    return [ql.FixedDividend(amounts[i], dates[i]) for i in range(n)]


# ----------------------------
# Pricing utilities (clean pipeline)
# ----------------------------
from typing import Optional


def set_eval_date(trade_date: str | pd.Timestamp):
    """Set QuantLib evaluation date once (QL has a global Settings state)."""
    _ensure_ql_import()
    ts = pd.Timestamp(trade_date)
    ql.Settings.instance().evaluationDate = ql.Date(ts.day, ts.month, ts.year)


def price_resolved_legs(
    resolved: dict,
    *,
    pricer,                             # callable: price(S,K,maturity,cp,vol,dividends)->dict
    trade_date: str | pd.Timestamp,
    dividends_sched_tuples: list[tuple],
    default_vol: Optional[float] = None,
) -> pd.DataFrame:
    """Price legs from `collect_market_and_resolve` output in a tidy, robust way.

    - Sets QL evaluation date once (thread-safety: keep one thread for QL).
    - Skips legs with missing/NaN vol unless `default_vol` is provided.
    - Adds dollar-scaled risk metrics with clear units.
    """
    import numpy as _np

    set_eval_date(trade_date)

    legs = resolved.get("dataset", [])
    out = []

    for leg in legs:
        S = leg.get("underlying_price")
        K = leg.get("strike")
        cp = str(leg.get("cp", "")).upper()[:1]
        vol = leg.get("iv")
        bid = leg.get("bid")
        ask = leg.get("ask")

        if vol is None or not _np.isfinite(vol):
            if default_vol is None:
                continue
            vol = float(default_vol)

        exp = pd.to_datetime(leg.get("expiration"))
        maturity = _to_ql_date(exp)

        res = pricer.price(S=S, K=K, maturity=maturity, cp=cp, vol=vol, dividends=dividends_sched_tuples)

        row = {
            "optionId": leg.get("optionId"),
            "cp": cp,
            "expiration": exp.strftime("%Y-%m-%d"),
            "strike": float(K),
            "underlying_price": float(S),
            "iv": float(vol),
            "bid": float(bid),
            "ask": float(ask),
            **res,
        }
        out.append(row)

    df = pd.DataFrame(out)
    if df.empty:
        return df

    # Clear, unit-aware scalings
    # - dollar_gamma_1pct: gamma PnL for a 1% move (0.01 * S); uses 0.5 * Γ * (ΔS)^2
    # - vega_1pt: vega per 1 vol point (QL vega is often per 1.00 = 100 vol pts)
    if {"gamma", "underlying_price"}.issubset(df.columns):
        df["dollar_gamma_1pct"] = 0.5 * df["gamma"] * (0.01 * df["underlying_price"]) ** 2
    if "vega" in df.columns:
        df["vega_1pt"] = df["vega"] * 0.01

    # Convenience: sorted output
    df = df.sort_values(["expiration", "cp", "strike"]).reset_index(drop=True)
    return df

# ----------------------------
# CLI demo
# ----------------------------

async def _demo(args):
    out: Dict[str, Any] = {}

    if args.ticker:
        print(f"Fetching Polygon aggregates for {args.ticker} {args.from_} → {args.to}...")
        df_poly = await fetch_polygon_agg(args.ticker, args.from_, args.to, multiplier=args.multiplier, timespan=args.timespan)
        print(df_poly.head())
        out["polygon_rows"] = int(df_poly.shape[0])

    if args.symbol:
        print(f"Fetching iVolatility for {args.symbol} {args.from_} → {args.to}...")
        df_iv = await fetch_ivolatility(symbol=args.symbol, start=args.from_, end=args.to)
        print(df_iv.head())
        out["ivolatility_rows"] = int(df_iv.shape[0])

    if args.yield_file:
        print(f"Loading yield curve from {args.yield_file}...")
        yc = load_yield_curve(args.yield_file)
        print(yc.tail(1))
        interp = make_yield_interpolator(yc)
        # Example spot checks
        for y in [0.25, 1, 2, 5, 10, 30]:
            print(f"Interpolated {y}Y: {interp(y):.4%}")
        out["yield_dates"] = yc.shape[0]
        out["yield_tenors"] = yc.shape[1]

    print("Summary:", json.dumps(out, indent=2))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Market data loader demo")
    p.add_argument("--ticker", type=str, help="Polygon ticker, e.g., SPY")
    p.add_argument("--symbol", type=str, help="iVolatility symbol, e.g., SPY")
    p.add_argument("--from", dest="from_", type=str, default="2024-01-01")
    p.add_argument("--to", dest="to", type=str, default="2024-02-01")
    p.add_argument("--multiplier", type=int, default=1)
    p.add_argument("--timespan", type=str, default="day", choices=["minute", "hour", "day", "week", "month"]) 
    p.add_argument("--yield-file", type=str, help="Path to local yield curve CSV/JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        asyncio.run(_demo(args))
    except ConfigError as ce:
        print(f"Config error: {ce}")
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
