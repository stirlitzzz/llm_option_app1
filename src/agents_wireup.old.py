# agents_wireup.py
from typing import Any, TypedDict, List, Dict, Optional
from agents import Agent, Runner, function_tool, RunContextWrapper

# ---------- Shared types ----------
class IDRow(TypedDict, total=False):
    optionId: str
    symbol: str
    cp: str
    expiration: str
    strike: float
    osym: Optional[str]

class StepResult(TypedDict, total=False):
    ok: bool
    dataset: List[IDRow]
    normalized: Dict[str, Any]
    issues: List[Dict[str, Any]]
    extras: Dict[str, Any]

# ---------- 1) Intent → Spec ----------
@function_tool
def parse_intent(text: str) -> StepResult:
    """Parse user text like 'AAPL Jan26 250/300 collar' into a spec dict."""
    # Your quick parser (regex/LLM), returning {"symbol","year","month","legs":[...],"structure":...}
    spec = {"symbol":"AAPL","year":2026,"month":1,"legs":[{"cp":"P","k":250},{"cp":"C","k":300}],"structure":"collar"}
    return {"ok": True, "dataset": [], "normalized": {"spec": spec}, "issues": []}

# ---------- 2) Series Resolver ----------
@function_tool
def resolve_series(spec: Dict[str, Any]) -> StepResult:
    """Hit /equities/eod/stock-opts-by-param, snap expiry/strikes, return option IDs."""
    # Call your resolve_option_rows[_multi](...) here and adapt to the format
    # For brevity, fake two IDs:
    ds = [
        {"optionId":"AAPL:2026-01-17:P:250","symbol":"AAPL","cp":"P","expiration":"2026-01-17","strike":250.0},
        {"optionId":"AAPL:2026-01-17:C:300","symbol":"AAPL","cp":"C","expiration":"2026-01-17","strike":300.0},
    ]
    return {"ok": True, "dataset": ds, "normalized": {"expiration":"2026-01-17"}, "issues": []}

# ---------- 3) Quotes/IV Loader ----------
@function_tool
def load_quotes(ids: List[IDRow]) -> StepResult:
    """Fetch bid/ask/iv/spot for the given option IDs (vectorized)."""
    # Call your ivolatility adapter, join in underlying spot.
    # Keep it minimal; presenter will use these fields.
    enriched = []
    spot = 190.12
    for r in ids:
        enriched.append({**r, "bid": 5.30, "ask": 5.90, "iv_bid": 0.245, "iv_ask": 0.255, "spot": spot})
    return {"ok": True, "dataset": enriched, "normalized": {}, "issues": []}

# ---------- 4) Curve / Dividend Providers ----------
@function_tool
def load_curve(curve_path: str) -> StepResult:
    """Load zero curve CSV → return a compact dict and a resolver to r(T)."""
    # Use your load_curve_csv() helper and stash the rows. For demo, inline rows:
    rows = [(0.25,0.0530),(0.50,0.0538),(1.00,0.0542),(2.00,0.0520)]
    return {"ok": True, "dataset": [], "normalized": {"curve_rows": rows}, "issues": []}

@function_tool
def build_div_forecast(symbol: str, expirations: List[str]) -> StepResult:
    """Return per-expiry cash dividend schedules."""
    # You already have this; demo returns one dividend before Jan-17-2026:
    divs = {"2026-01-17": [("2025-11-15", 0.24)]}
    return {"ok": True, "dataset": [], "normalized": {"divs": divs}, "issues": []}

# ---------- 5) Pricer (QuantLib) ----------
@function_tool
def price_with_ql(
    legs: List[Dict[str, Any]],
    curve_rows: List[tuple],
    divs_by_exp: Dict[str, List[tuple]],
    use_input_vols: bool = True
) -> StepResult:
    """Price each leg with your QLAmericanPricer; reuse SimpleQuotes; return theo+greeks."""
    # Instantiate (or look up cached) QLAmericanPricer from curve_rows; iterate legs.
    # Demo payload:
    out = []
    for leg in legs:
        theo = 5.62
        out.append({**leg, "theo": theo, "delta": -0.42, "gamma": 0.012, "vega": 0.08, "theta": -0.015})
    return {"ok": True, "dataset": out, "normalized": {}, "issues": []}

# ---------- 6) Presenter ----------
@function_tool
def present_rows(rows: List[Dict[str, Any]]) -> StepResult:
    """Return compact UI rows: [Bid | THEO | Ask] and [IV Bid | σ_input | IV Ask] per leg."""
    ui = []
    for r in rows:
        mid = (r.get("bid",0)+r.get("ask",0))/2 if "bid" in r else None
        edge_vs_mid = (r["theo"] - mid) if mid else None
        ui.append({
            "row": [r.get("bid"), r["theo"], r.get("ask")],
            "vol_row": [r.get("iv_bid"), r.get("iv_input", r.get("iv_mid")), r.get("iv_ask")],
            "meta": {"cp": r["cp"], "K": r["strike"], "exp": r["expiration"], "edge_vs_mid": edge_vs_mid}
        })
    return {"ok": True, "dataset": ui, "normalized": {}, "issues": []}
