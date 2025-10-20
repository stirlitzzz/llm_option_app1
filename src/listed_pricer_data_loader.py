import os
import dotenv
import pandas as pd
import asyncio
import pandas as pd, numpy as np
from datetime import datetime
# --- use your existing helpers if you have them; duplicated here for self-containment ---
try:
    import ivolatility as ivol  # type: ignore
    _IVOL_SDK = True
except Exception:
    ivol = None
    _IVOL_SDK = False

dotenv.load_dotenv()
api_key = os.getenv("IVOL_API_KEY")
if not api_key:
    raise RuntimeError("❌ IVOL_API_KEY not found in .env file or environment.")

ivol.setLoginParams(apiKey=api_key)


def get_ivol_yc(pd_date):
    getMarketData = ivol.setMethod('/equities/interest-rates')
    marketData = getMarketData(from_=pd_date.strftime('%Y-%m-%d'), till=pd_date.strftime('%Y-%m-%d'), currency='USD')
    return marketData

def apply_ivol_yc(marketData,df_options):
    df_results=df_options.copy()
    #df_results['yc'] = None
    for index, row in df_options.iterrows():
        period = int(row['Texp'] * 252)
        yc = marketData[marketData['period'] == period]['rate'].values[0]/100
        df_results.at[index, 'Rate'] = yc
    return df_results




def init_ivol_options_client():
    if not _IVOL_SDK:
        raise RuntimeError("ivolatility SDK not installed. `pip install ivolatility`")
    api_key = os.getenv("IVOL_API_KEY")
    if not api_key:
        raise RuntimeError("IVOL_API_KEY not found in environment.")
    ivol.setLoginParams(apiKey=api_key)
    return ivol.setMethod('/equities/eod/stock-opts-by-param')

def _numify(df: pd.DataFrame, col: str, dtype: str = "float64"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
    else:
        df[col] = pd.Series(pd.NA, index=df.index, dtype=dtype)

def _normalize_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Light rename + types + computed mark."""
    if df is None or df.empty:
        return pd.DataFrame()
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
    if "cp" in df.columns:
        df["cp"] = df["cp"].astype(str).str.upper().str[0]
    for field in ["bid","ask","iv","strike","delta","gamma","vega","underlying_price"]:
        _numify(df, field)
    for field in ["openInterest","volume"]:
        _numify(df, field, dtype="Int64")
    df["mark"] = np.where(np.isfinite(df["bid"]) & np.isfinite(df["ask"]),
                          (df["bid"] + df["ask"]) / 2.0,
                          np.where(np.isfinite(df["bid"]), df["bid"], df["ask"]))
    if "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"]).dt.strftime("%Y-%m-%d")
    return df

def _ensure_iso_date(v) -> str:
    """Accepts 'YYYY-MM-DD', 'M/D/YY', datetime, etc. -> 'YYYY-MM-DD'."""
    if isinstance(v, str):
        # try a couple formats
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
            try:
                return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    if isinstance(v, (pd.Timestamp, datetime)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    raise ValueError(f"Unrecognized date format: {v}")

def fetch_ivol_chain(symbol: str, trade_date: str, getMarketData=None,
                     dte_from: int = 0, dte_to: int = 1200,
                     mny_from: int = -200, mny_to: int = 200) -> pd.DataFrame:
    """Download C & P once and return a single normalized DataFrame."""
    if getMarketData is None:
        getMarketData = init_ivol_options_client()
    sym = symbol.upper()
    def _fetch(cp: str):
        return getMarketData(symbol=sym, tradeDate=trade_date,
                             dteFrom=dte_from, dteTo=dte_to,
                             moneynessFrom=mny_from, moneynessTo=mny_to, cp=cp)
    # SDK is sync; fetch serially for simplicity
    calls = _normalize_chain(_fetch('C'))
    puts  = _normalize_chain(_fetch('P'))
    if calls.empty and puts.empty:
        return pd.DataFrame()
    df = pd.concat([calls, puts], ignore_index=True)
    return df

def pick_by_maturity_and_strike(
    chain_df: pd.DataFrame,
    legs_df: pd.DataFrame,
    symbol: str,
    snap_to_nearest: bool = False
) -> dict:
    """
    legs_df must have columns: ['Expiration','Strike','CP'] (case-insensitive ok).
    Returns dict { ok, dataset, issues, matched_df }.
    """
    if chain_df.empty:
        return {"ok": False, "dataset": [], "issues": [{"code":"NO_CHAIN","msg":"empty chain"}], "matched_df": chain_df}

    # standardize leg column names
    cols = {c.lower(): c for c in legs_df.columns}
    need = {"expiration","strike","cp"}
    if not need.issubset(set(cols.keys())):
        raise ValueError("legs_df must have columns Expiration, Strike, CP")
    L = legs_df.rename(columns={
        cols["expiration"]: "Expiration",
        cols["strike"]: "Strike",
        cols["cp"]: "CP"
    }).copy()

    # normalize leg fields
    L["CP"] = L["CP"].astype(str).str.upper().str[0]
    L["Strike"] = pd.to_numeric(L["Strike"], errors="coerce").astype(float)
    L["ExpirationISO"] = L["Expiration"].apply(_ensure_iso_date)

    # fast lookup: (exp, cp) -> frame sorted by strike
    by_key = { (e, c): g.sort_values("strike").reset_index(drop=True)
               for (e, c), g in chain_df.groupby(["expiration","cp"], dropna=True) }

    issues, rows = [], []
    for i, leg in L.iterrows():
        exp = leg["ExpirationISO"]; cp = leg["CP"]; k = float(leg["Strike"])
        g = by_key.get((exp, cp))
        if g is None or g.empty:
            issues.append({"code":"NO_EXP_CP","msg":f"{symbol} {cp} @ {exp} not in chain"})
            continue
        # exact strike?
        exact = g[g["strike"] == k]
        if not exact.empty:
            sel = exact.iloc[0]
            snapped = False
        elif snap_to_nearest:
            # nearest strike fallback
            strikes = g["strike"].values
            idx = int(np.argmin(np.abs(strikes - k)))
            sel = g.iloc[idx]
            snapped = True
            issues.append({"code":"SNAP_STRIKE","msg":f"{symbol} {cp} {k} -> {float(sel['strike'])} @ {exp}"})
        else:
            issues.append({"code":"NO_STRIKE","msg":f"{symbol} {cp} {k} @ {exp} not listed"})
            continue

        rows.append({
            "symbol": symbol.upper(),
            "cp": cp,
            "expiration": exp,
            "strike": float(sel["strike"]),
            "snapped_strike": bool(snapped),
            "bid": float(sel.get("bid")) if pd.notna(sel.get("bid")) else np.nan,
            "ask": float(sel.get("ask")) if pd.notna(sel.get("ask")) else np.nan,
            "mark": float(sel.get("mark")) if pd.notna(sel.get("mark")) else np.nan,
            "iv": float(sel.get("iv")) if pd.notna(sel.get("iv")) else np.nan,
            "openInterest": sel.get("openInterest"),
            "volume": sel.get("volume"),
            "delta": float(sel.get("delta")) if pd.notna(sel.get("delta")) else np.nan,
            "gamma": float(sel.get("gamma")) if pd.notna(sel.get("gamma")) else np.nan,
            "vega": float(sel.get("vega")) if pd.notna(sel.get("vega")) else np.nan,
            "underlying_price": float(sel.get("underlying_price")) if pd.notna(sel.get("underlying_price")) else np.nan,
            "optionId": sel.get("optionId") or sel.get("osym"),
        })

    return {
        "ok": len(rows) == len(L),
        "dataset": rows,
        "issues": issues,
        "matched_df": pd.DataFrame(rows)
    }

# --- one-call convenience wrapper ---
def fetch_and_pick(symbol: str, trade_date: str, legs_df: pd.DataFrame, getMarketData=None, snap_to_nearest=False):
    chain = fetch_ivol_chain(symbol, trade_date, getMarketData=getMarketData)
    return pick_by_maturity_and_strike(chain, legs_df, symbol, snap_to_nearest=snap_to_nearest)