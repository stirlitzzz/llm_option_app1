# agents_wireup.py
from __future__ import annotations
from typing import TypedDict, Literal, List, Optional
from agents import function_tool

# ---------- Strict models ----------
class LegSpec(TypedDict):
    cp: Literal["P","C"]
    k: float

class Spec(TypedDict):
    symbol: str
    year: int
    month: int
    legs: List[LegSpec]
    structure: Literal["collar","vertical","calendar","diagonal","custom"]

class Issue(TypedDict, total=False):
    code: str
    msg: str
    severity: Literal["warning","blocker"]

class IDRow(TypedDict, total=False):
    optionId: str
    symbol: str
    cp: Literal["P","C"]
    expiration: str        # YYYY-MM-DD
    strike: float
    osym: Optional[str]

class QuoteRow(TypedDict):
    optionId: str
    bid: Optional[float]
    ask: Optional[float]
    iv_bid: Optional[float]
    iv_ask: Optional[float]
    spot: Optional[float]

class CurveRow(TypedDict):
    T: float               # years
    r: float               # cont. zero

class CurvePayload(TypedDict):
    rows: List[CurveRow]

class DivCash(TypedDict):
    date: str              # YYYY-MM-DD
    amount: float

class DivsForExpiry(TypedDict):
    expiration: str        # YYYY-MM-DD
    cash: List[DivCash]

class PricerInputRow(TypedDict):
    optionId: str
    symbol: str
    cp: Literal["P","C"]
    expiration: str
    strike: float
    # add sigma_input here if you want the pricer to take user vols

class PricedRow(TypedDict, total=False):
    optionId: str
    symbol: str
    cp: Literal["P","C"]
    expiration: str
    strike: float
    theo: float
    delta: Optional[float]
    gamma: Optional[float]
    vega: Optional[float]
    theta: Optional[float]

class UIRow(TypedDict, total=False):
    row: List[Optional[float]]         # [bid, theo, ask]
    vol_row: Optional[List[Optional[float]]]
    meta: dict                         # if you want, replace with a TypedDict too

# ---------- Tool-specific I/O types ----------
class ParseIntentOut(TypedDict):
    ok: bool
    spec: Spec
    issues: List[Issue]

class ResolveSeriesOut(TypedDict):
    ok: bool
    dataset: List[IDRow]
    expiration: Optional[str]
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

# ---------- Tools ----------
@function_tool
def parse_intent(text: str) -> ParseIntentOut:
    spec: Spec = {
        "symbol":"AAPL","year":2026,"month":1,
        "legs":[{"cp":"P","k":250.0},{"cp":"C","k":300.0}],
        "structure":"collar",
    }
    return {"ok": True, "spec": spec, "issues": []}

@function_tool
def resolve_series(spec: Spec) -> ResolveSeriesOut:
    ds: List[IDRow] = [
        {"optionId":"AAPL:2026-01-17:P:250","symbol":"AAPL","cp":"P","expiration":"2026-01-17","strike":250.0},
        {"optionId":"AAPL:2026-01-17:C:300","symbol":"AAPL","cp":"C","expiration":"2026-01-17","strike":300.0},
    ]
    return {"ok": True, "dataset": ds, "expiration":"2026-01-17", "issues": []}

@function_tool
def load_quotes(ids: List[IDRow]) -> LoadQuotesOut:
    out: List[QuoteRow] = []
    spot = 190.12
    for r in ids:
        out.append({"optionId": r["optionId"], "bid":5.30, "ask":5.90, "iv_bid":0.245, "iv_ask":0.255, "spot":spot})
    return {"ok": True, "quotes": out, "issues": []}

@function_tool
def load_curve(curve_path: str) -> LoadCurveOut:
    rows: List[CurveRow] = [{"T":0.25,"r":0.0530},{"T":0.50,"r":0.0538},{"T":1.00,"r":0.0542}]
    return {"ok": True, "curve": {"rows": rows}, "issues": []}

@function_tool
def build_div_forecast(symbol: str, expirations: List[str]) -> BuildDivsOut:
    divs: List[DivsForExpiry] = [{"expiration":"2026-01-17","cash":[{"date":"2025-11-15","amount":0.24}]}]
    return {"ok": True, "divs": divs, "issues": []}

@function_tool
def price_with_ql(
    legs: List[PricerInputRow],
    curve: CurvePayload,
    divs: List[DivsForExpiry],
    use_input_vols: bool = True
) -> PriceWithQLOut:
    priced: List[PricedRow] = []
    for leg in legs:
        priced.append({**leg, "theo":5.62, "delta":-0.42, "gamma":0.012, "vega":0.08, "theta":-0.015})
    return {"ok": True, "priced": priced, "issues": []}

@function_tool
def present_rows(priced_rows: List[PricedRow], quotes: List[QuoteRow]) -> PresentRowsOut:
    qmap = {q["optionId"]: q for q in quotes}
    ui: List[UIRow] = []
    for r in priced_rows:
        q = qmap.get(r["optionId"], {})
        bid, ask = q.get("bid"), q.get("ask")
        theo = r.get("theo")
        mid = (bid+ask)/2 if (bid is not None and ask is not None) else None
        ui.append({"row":[bid, theo, ask], "vol_row": None,
                   "meta":{"cp": r["cp"], "K": r["strike"], "exp": r["expiration"], "edge_vs_mid": (None if mid is None or theo is None else theo - mid)}})
    return {"ok": True, "ui": ui, "issues": []}
