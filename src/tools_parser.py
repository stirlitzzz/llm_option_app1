# src/tools_parser_py.py
from agents import function_tool
from tracing import trace_tool
from bus import BUS
from datetime import date

# --- your MVP parser pieces (paste your working funcs here) ---
from strategy_fast_parse import parse_strategy_mvp  # returns {ok, quantity, symbol, maturities, strikes, ratio, structure}
from strategy_normalize import build_legs              # optional: to expand into legs

def _bus_put_json(topic: str, key: str, payload: dict, producer: str):
    # wraps BUS.put with a small status envelope for idempotency/debug
    BUS.put(topic=topic, key=key, payload={
        "status":"done", "producer":producer, "payload":payload
    }, producer=producer)




def parse_and_store_mvp_impl(text: str, key_hint: str | None = None) -> dict:
    """
    Non-LLM: parse text -> MVP spec -> store on bus under topic 'spec'.
    Returns {ok, key, issues}.
    """
    mvp = parse_strategy_mvp(text)
    if not mvp.get("ok"):
        return {"ok": False, "key": None, "issues":[{"code":"PARSE_FAIL","msg":"MVP parser could not extract core fields."}]}

    # derive a key if not provided (SYMBOL-YYYY-MM-structure or SYMBOL-custom)
    sym = mvp["symbol"]
    # pick first maturity, if any, else 'NA'
    mat = (mvp["maturities"][0] if mvp["maturities"] else "NA")
    struct = (mvp["structure"]["cp_string"] if isinstance(mvp["structure"], dict) else mvp["structure"])
    key = key_hint or f"{sym}-{mat}-{struct}"

    _bus_put_json("spec", key, mvp, producer="parser_py")
    return {"ok": True, "key": key, "issues": []}

@trace_tool("parse_and_store_mvp")
@function_tool
def parse_and_store_mvp(text: str, key_hint: str | None = None) -> dict:
    return parse_and_store_mvp_impl(text, key_hint)



def normalize_and_store_legs_impl(key: str, ref_year: int, ref_month: int, ref_day: int) -> dict:
    """
    Load MVP spec from bus, expand to legs, store under 'legs'.
    This keeps LLM entirely out of the loop for normalization.
    """
    spec_wrap = BUS.get(topic="spec", key=key)
    if not spec_wrap or "payload" not in spec_wrap:
        return {"ok": False, "issues":[{"code":"NO_SPEC","msg":f"spec:{key} not found"}]}

    mvp = spec_wrap["payload"]
    ref_date = date(ref_year, ref_month, ref_day)

    legs_blob = build_legs(
        quantity=mvp["quantity"],
        symbol=mvp["symbol"],
        maturities=mvp["maturities"],
        strikes=mvp["strikes"],
        ratio=mvp.get("ratio"),
        structure=mvp["structure"],
        ref_date=ref_date
    )
    _bus_put_json("legs", key, legs_blob, producer="parser_py")

    return {"ok": True, key:key,"issues": legs_blob.get("issues", [])}


@function_tool
@trace_tool("normalize_and_store_legs")
def normalize_and_store_legs(key: str, ref_year: int, ref_month: int, ref_day: int) -> dict:
    return normalize_and_store_legs_impl(key, ref_year, ref_month, ref_day)