# src/agents_wireup.py
from __future__ import annotations
from typing import List, Literal, Optional, TypedDict
from agents import function_tool
from tracing import trace_tool

# ---------- Strict models ----------
class Issue(TypedDict, total=False):
    code: str
    msg: str
    severity: Literal["warning", "blocker"]

class LegSpec(TypedDict):
    cp: Literal["P", "C"]
    k: float

class Spec(TypedDict):
    symbol: str
    year: int
    month: int
    size: int
    structure: Literal["collar", "vertical", "calendar", "diagonal", "custom"]
    legs: List[LegSpec]

class IDRow(TypedDict):
    optionId: str
    symbol: str
    cp: Literal["P", "C"]
    expiration: str       # YYYY-MM-DD
    strike: float

class QuoteRow(TypedDict):
    optionId: str
    bid: Optional[float]
    ask: Optional[float]
    iv_bid: Optional[float]
    iv_ask: Optional[float]
    spot: Optional[float]

class CurveRow(TypedDict):
    T: float              # years
    r: float              # continuous-comp zero

class CurvePayload(TypedDict):
    rows: List[CurveRow]

class DivCash(TypedDict):
    date: str             # YYYY-MM-DD
    amount: float

class DivsForExpiry(TypedDict):
    expiration: str       # YYYY-MM-DD
    cash: List[DivCash]

class PricedRow(TypedDict, total=False):
    optionId: str
    symbol: str
    cp: Literal["P", "C"]
    expiration: str
    strike: float
    theo: float
    delta: Optional[float]
    gamma: Optional[float]
    vega: Optional[float]
    theta: Optional[float]

class UIMeta(TypedDict, total=False):
    cp: Literal["P", "C", ""]        # "" if absent
    K: Optional[float]
    exp: Optional[str]
    edge_vs_mid: Optional[float]

class UIRow(TypedDict, total=False):
    row: List[Optional[float]]       # [bid, theo, ask]
    vol_row: List[Optional[float]]   # [iv_bid, None|σ_input, iv_ask]
    meta: UIMeta

# ---------- Tool return types ----------
class TranslateStrategyOut(TypedDict):
    ok: bool
    spec: Spec
    issues: List[Issue]

class ResolveSeriesOut(TypedDict):
    ok: bool
    dataset: List[IDRow]
    expiration: str
    issues: List[Issue]

class LoadQuotesOut(TypedDict):
    ok: bool
    quotes: List[QuoteRow]
    issues: List[Issue]

class LoadCurveOut(TypedDict):
    ok: bool
    curve: CurvePayload
    issues: List[Issue]

class BuildDivsOut(TypedDict):
    ok: bool
    divs: List[DivsForExpiry]
    issues: List[Issue]

class PriceWithQLOut(TypedDict):
    ok: bool
    priced: List[PricedRow]
    issues: List[Issue]

class PresentRowsOut(TypedDict):
    ok: bool
    ui: List[UIRow]
    issues: List[Issue]

# ---------- 1) Intent → Spec ----------
@function_tool
@trace_tool("translate_strategy")
def translate_strategy(text: str) -> TranslateStrategyOut:
    # TODO: replace with your actual parser (regex/LLM). Hard-coded for demo.
    spec: Spec = {
        "symbol": "AAPL",
        "year": 2026,
        "month": 1,
        "size": 500,
        "structure": "collar",
        "legs": [{"cp": "P", "k": 250.0}, {"cp": "C", "k": 300.0}],
    }
    return {"ok": True, "spec": spec, "issues": []}

# ---------- 2) Series Resolver ----------
@function_tool
@trace_tool("resolve_series")
def resolve_series(spec: Spec) -> ResolveSeriesOut:
    print(f"agents_wireup.py: resolve_series: spec: {spec}")
    print(f"agents_wireup.py: resolve_series: spec['legs']: {spec['legs']}")
    symbol = spec["symbol"].upper()
    year   = int(spec["year"])
    month  = int(spec["month"])
    exp=spec["legs"][0]["expiry"]
    #exp    = f"{year}-{month:02d}-17"  # demo: “monthly” placeholder
    ds: List[IDRow] = []
    for leg in spec["legs"]:
        ds.append({
            "optionId": f"{symbol}:{exp}:{leg['cp']}:{int(leg['k'])}",
            "symbol": symbol,
            "cp": leg["cp"],
            "expiration": leg["expiry"]["iso"],
            #"expiration": exp,
            "strike": float(leg["k"]),
        })
    print(f"agents_wireup.py: resolve_series: ds: {ds}")
    return {"ok": True, "dataset": ds, "expiration": exp, "issues": []}

# ---------- 3) Quotes/IV Loader (stub) ----------
@function_tool
@trace_tool("load_quotes")
def load_quotes(ids: List[IDRow]) -> LoadQuotesOut:
    out: List[QuoteRow] = []
    for r in ids:
        out.append({
            "optionId": r["optionId"],
            "bid": 5.30,
            "ask": 5.90,
            "iv_bid": 0.245,
            "iv_ask": 0.255,
            "spot": 190.12,
        })
    return {"ok": True, "quotes": out, "issues": []}

# ---------- 4) Curve & Dividend Providers (stubs) ----------
@function_tool
@trace_tool("load_curve")
def load_curve(curve_path: str) -> LoadCurveOut:
    rows: List[CurveRow] = [{"T": 0.25, "r": 0.0530}, {"T": 0.50, "r": 0.0538}, {"T": 1.00, "r": 0.0542}]
    return {"ok": True, "curve": {"rows": rows}, "issues": []}

@function_tool
@trace_tool("build_div_forecast")
def build_div_forecast(symbol: str, expirations: List[str]) -> BuildDivsOut:
    exp = expirations[0] if expirations else "2026-01-17"
    divs: List[DivsForExpiry] = [{"expiration": exp, "cash": [{"date": "2025-11-15", "amount": 0.24}]}]
    return {"ok": True, "divs": divs, "issues": []}

# ---------- 5) Pricer (stub) ----------
@function_tool
@trace_tool("price_with_ql")
def price_with_ql(
    legs: List[IDRow],
    curve: CurvePayload,
    divs: List[DivsForExpiry],
    use_input_vols: bool = True,
) -> PriceWithQLOut:
    priced: List[PricedRow] = []
    for leg in legs:
        priced.append({
            "optionId": leg["optionId"],
            "symbol": leg["symbol"],
            "cp": leg["cp"],
            "expiration": leg["expiration"],
            "strike": float(leg["strike"]),
            "theo": 5.62,
            "delta": (-0.42 if leg["cp"] == "P" else 0.38),
            "gamma": 0.012,
            "vega": 0.080,
            "theta": -0.015,
        })
    return {"ok": True, "priced": priced, "issues": []}

# ---------- 6) Presenter ----------
@function_tool
@trace_tool("present_rows")
def present_rows(priced_rows: List[PricedRow], quotes: List[QuoteRow]) -> PresentRowsOut:
    qmap = {q["optionId"]: q for q in quotes}
    ui: List[UIRow] = []
    for r in priced_rows:
        q = qmap.get(r["optionId"])
        bid = q["bid"] if q else None
        ask = q["ask"] if q else None
        theo = r.get("theo")
        mid = ((bid + ask) / 2.0) if (bid is not None and ask is not None) else None
        edge = (round(float(theo - mid), 2) if (theo is not None and mid is not None) else None)
        ui.append({
            "row": [bid, theo, ask],
            "vol_row": [
                (q["iv_bid"] if q else None),
                None,  # σ_input editable in UI later
                (q["iv_ask"] if q else None),
            ],
            "meta": {"cp": r["cp"], "K": r["strike"], "exp": r["expiration"], "edge_vs_mid": edge},
        })
    return {"ok": True, "ui": ui, "issues": []}


