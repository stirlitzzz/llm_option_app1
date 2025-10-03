# src/tools_parse_py.py
from agents import function_tool
from tracing import trace_tool
import re

@function_tool
@trace_tool("parse_strategy_python")
def parse_strategy_python(text: str) -> dict:
    """
    Very rough parser. Returns { ok, spec?, issues }.
    Examples supported: 'trade 500 AAPL jan26 250/300 collar'
    """
    t = text.lower()
    m = re.search(r'(\d+)\s*([a-z]+)\s+(\w+)\s+(\d{3})/(\d{3})\s+(collar|vertical|calendar|diagonal)', t)
    if not m:
        return {"ok": False, "issues": [{"code":"PARSE_FAIL","msg":"Pattern not recognized"}]}
    size = int(m.group(1)); month_str = m.group(2); sym = m.group(3).upper()
    k1 = float(m.group(4)); k2 = float(m.group(5)); structure = m.group(6)
    mon_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
    month = mon_map.get(month_str[:3], None)
    year = 2026  # naive default; swap with your parse_expiry_mmmyy if present
    if not month:
        return {"ok": False, "issues": [{"code":"BAD_MONTH","msg":f"Month '{month_str}' not recognized"}]}
    spec = {"symbol": sym, "year": year, "month": month, "size": size,
            "structure": structure, "legs": [{"cp":"P","k":k1},{"cp":"C","k":k2}]}
    return {"ok": True, "spec": spec, "issues": []}