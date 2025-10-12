# strategy_normalize.py
from __future__ import annotations
from datetime import date, timedelta
from typing import List, Tuple, Optional, Dict, Any
import re

# ---------------- Dates & maturities ----------------

_MON = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
_MON_RE = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:uary|ch|il|e|y|ust|tember|ober|ember)?"

def third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)  # 0=Mon..4=Fri

def yy_to_yyyy(yy: int, pivot: int = 99) -> int:
    """Map 2-digit year to 4-digit (00..pivot -> 2000..20pivot; else 1900..19yy)."""
    return 2000 + yy if yy <= pivot else 1900 + yy

def parse_maturity_token(token: str, ref_date: date, pivot: int = 99) -> Optional[Tuple[int,int,Optional[str]]]:
    """
    Parse a single maturity string to (year, month, iso) where iso is:
      - exact date if user gave a full date (MM/DD/YYYY or MM/DD/YY)
      - None if only month/year (we'll fill 3rd Friday later)
    Supports: Jan26, Jan 2026, January 2026, YYYY-MM, MM/YYYY, MM/DD/YYYY, bare month (e.g., 'feb')
    """
    t = token.strip().lower()

    # MM/DD/YYYY or MM/DD/YY
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", t)
    if m:
        mm, dd, yy = int(m.group(1)), int(m.group(2)), m.group(3)
        yyyy = yy_to_yyyy(int(yy), pivot) if len(yy) == 2 else int(yy)
        if 1<=mm<=12 and 1<=int(dd)<=31:
            return yyyy, mm, date(yyyy, mm, int(dd)).isoformat()

    # YYYY-MM or YYYY/MM
    m = re.fullmatch(r"(19\d{2}|20\d{2})[/-](\d{1,2})", t)
    if m:
        return int(m.group(1)), int(m.group(2)), None

    # Month + Year: Jan26 / Jan 2026 / January 2026
    m = re.fullmatch(rf"({_MON_RE})\s*([0-9]{{2}}|[0-9]{{4}})", t)
    if m:
        mm = _MON[m.group(1)[:3]]
        yr = m.group(2)
        yyyy = yy_to_yyyy(int(yr), pivot) if len(yr) == 2 else int(yr)
        return yyyy, mm, None

    # Compact MMMYY: jan26
    m = re.fullmatch(rf"({_MON_RE})(\d{{2}})", t)
    if m:
        return yy_to_yyyy(int(m.group(2)), pivot), _MON[m.group(1)[:3]], None

    # Bare month → choose closest future/this-year month using ref_date
    m = re.fullmatch(rf"({_MON_RE})", t)
    if m:
        mm = _MON[m.group(1)[:3]]
        yyyy = ref_date.year if mm >= ref_date.month else ref_date.year + 1
        return yyyy, mm, None

    return None

def normalize_maturities(maturity_tokens: List[str], legs_count: int, ref_date: date) -> List[Dict[str, Any]]:
    """
    Normalize a list of maturity tokens to length legs_count.
    Each item: {"year":int,"month":int,"iso":str|None,"label":"MonYY"}
    If iso is None, we don't force a day; downstream can choose 3rd Friday.
    """
    parsed: List[Tuple[int,int,Optional[str]]] = []
    for tok in maturity_tokens:
        pm = parse_maturity_token(tok, ref_date=ref_date)
        if pm: parsed.append(pm)

    # Propagate last if too few; if none parsed, leave empty list
    if not parsed:
        return []

    while len(parsed) < legs_count:
        parsed.append(parsed[-1])

    parsed = parsed[:legs_count]

    out = []
    for (y,m,iso) in parsed:
        mon3 = [k for k,v in _MON.items() if v==m][0].title()
        out.append({"year": y, "month": m, "iso": iso, "label": f"{mon3}{str(y)[-2:]}"})
    return out

# ---------------- Ratios & structure → legs ----------------

def expand_ratio(ratio: Optional[str], legs_count: int) -> List[int]:
    """'1x2x3' -> [1,2,3]; None -> [1]*legs; short arrays are repeated on the right."""
    if not ratio:
        base = [1]
    else:
        base = [int(x) for x in ratio.lower().split('x')]
    out = base[:]
    while len(out) < legs_count:
        out.append(out[-1])
    return out[:legs_count]

def infer_cp_list(structure: Any, strikes_count: int) -> List[str]:
    """Map structure → sequence of 'P'/'C' per leg; cp_string supports arbitrary shapes."""
    if isinstance(structure, dict) and "cp_string" in structure:
        return [ch.upper() for ch in structure["cp_string"]]

    s = str(structure).lower()
    if s in ("collar", "rr", "risk_reversal", "straddle", "strangle", "cs", "call_spread", "ps", "put_spread"):
        if s in ("collar", "rr", "risk_reversal"):   # 2 legs
            return ["P","C"]
        if s in ("straddle","strangle"):             # 2 legs
            return ["P","C"]
        if s in ("cs","call_spread"):                # 2 legs
            return ["C","C"]
        if s in ("ps","put_spread"):                 # 2 legs
            return ["P","P"]
    # Fallback: if 2 strikes, default to P,C; if 1 strike, default C
    if strikes_count >= 2:
        return ["P","C"] + ["C"]*(strikes_count-2)  # naive fill
    return ["C"]

def default_sides(structure: Any, cps: List[str], strikes: List[float], quantity: int) -> List[str]:
    """
    Return BUY/SELL per leg based on structure convention.
    - collar:  BUY P, SELL C
    - rr:      SELL P, BUY C
    - cs:      BUY lowerK, SELL higherK
    - ps:      BUY higherK, SELL lowerK
    - straddle/strangle: BUY both if quantity>0 else SELL both
    - cp_string/custom: 'AUTO'
    """
    s = str(structure).lower() if not (isinstance(structure, dict) and "cp_string" in structure) else "custom"
    n = len(cps)
    if s == "collar":
        return ["BUY" if cp=="P" else "SELL" for cp in cps]
    if s in ("rr","risk_reversal"):
        return ["SELL" if cp=="P" else "BUY" for cp in cps]
    if s in ("straddle","strangle"):
        return (["BUY"]*n) if quantity >= 0 else (["SELL"]*n)
    if s in ("cs","call_spread") and len(strikes)>=2:
        lo_idx = strikes.index(min(strikes[:2]))
        sides = ["SELL"]*n
        sides[lo_idx] = "BUY"
        return sides
    if s in ("ps","put_spread") and len(strikes)>=2:
        hi_idx = strikes.index(max(strikes[:2]))
        sides = ["SELL"]*n
        sides[hi_idx] = "BUY"
        return sides
    # custom / cp_string: leave AUTO
    return ["AUTO"]*n



def set_monthly_expiries_to_third_friday(
    legs: List[Dict[str, Any]],
    *,
    preserve_exact_dates: bool = True,
    holiday_adjust: bool = False,
    holiday_is_friday: Optional[callable] = None,
) -> None:
    """
    Mutates each leg in-place:
      - if leg['expiry']['iso'] is missing (or preserve_exact_dates=False),
        fill it with the 3rd Friday ISO for (year, month).
      - optional 'holiday_adjust': if the 3rd Friday is a holiday/closed, move to Thursday.
        Provide 'holiday_is_friday(d: date) -> bool' to flag those dates.

    legs[i]['expiry'] must have {'year': int, 'month': int, 'iso': str|None}.
    """
    for leg in legs:
        exp = leg.get("expiry") or {}
        y, m, iso = exp.get("year"), exp.get("month"), exp.get("iso")
        if not y or not m:
            continue  # nothing to do

        if iso and preserve_exact_dates:
            continue  # user already gave an exact date; leave it

        d = third_friday(y, m)

        # Optional: holiday adjustment (simple rule)
        if holiday_adjust and callable(holiday_is_friday) and holiday_is_friday(d):
            d = d - timedelta(days=1)  # move to Thursday

        exp["iso"] = d.isoformat()
        leg["expiry"] = exp


def build_legs(
    *,
    quantity: int,
    symbol: str,
    maturities: List[str],
    strikes: List[float],
    ratio: Optional[str],
    structure: Any,
    ref_date: date
) -> Dict[str, Any]:
    """
    Expand MVP inputs into normalized legs ready for series resolution.
    Returns: { legs: [...], issues: [...], notes: str }
    Each leg: {cp, strike, side, ratio, qty, expiry{year,month,iso,label}}
    """
    issues: List[Dict[str,str]] = []

    # Determine target leg count
    cp_seq = infer_cp_list(structure, strikes_count=len(strikes))
    legs_n = max(len(cp_seq), len(strikes))

    # Propagate strikes if short
    if len(strikes) == 0:
        issues.append({"code":"MISSING_STRIKES","msg":"No strikes provided"})
        strikes_prop = [None]*legs_n
    else:
        strikes_prop = strikes[:]
        while len(strikes_prop) < legs_n:
            strikes_prop.append(strikes_prop[-1])
        strikes_prop = strikes_prop[:legs_n]

    # Normalize maturities to the leg count
    exp_list = normalize_maturities(maturities, legs_n, ref_date=ref_date)
    if not exp_list:
        issues.append({"code":"MISSING_MATURITIES","msg":"No valid maturities parsed"})
        # keep placeholders so downstream can ask
        exp_list = [{"year":None,"month":None,"iso":None,"label":None} for _ in range(legs_n)]

    # Ratio expansion
    ratios = expand_ratio(ratio, legs_n)

    # BUY/SELL defaults
    sides = default_sides(structure, cp_seq, strikes_prop, quantity)

    # Build legs
    legs = []
    for i in range(legs_n):
        cp = cp_seq[i] if i < len(cp_seq) else cp_seq[-1]
        r  = ratios[i]
        side = sides[i] if i < len(sides) else "AUTO"
        leg_qty_signed = (quantity * r) if side == "BUY" else (-(quantity * r)) if side == "SELL" else (quantity * r)
        legs.append({
            "cp": cp,
            "strike": None if strikes_prop[i] is None else float(strikes_prop[i]),
            "side": side,
            "ratio": int(r),
            "qty": int(leg_qty_signed),
            "expiry": exp_list[i]
        })
        #print(f"leg: {legs[-1]}")
    set_monthly_expiries_to_third_friday(legs, holiday_adjust=True, holiday_is_friday=None)

    #print(f"legs: {legs}")
    notes = ""
    return {"legs": legs, "issues": issues, "notes": notes}
