import re
from datetime import date, timedelta
from typing import List, Dict, Set,Any, Optional,Tuple

_MON = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
        "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
# short+long month names
_MON_RE = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:uary|ch|il|e|y|ust|tember|ober|ember)?"

def yy_to_yyyy(yy: int, pivot: int = 69) -> int:
    return 2000 + yy if yy <= pivot else 1900 + yy

def _add(out: List[Tuple[str,int,int]], seen: Set[Tuple[int,int]], y: int, m: int):
    if not (1 <= m <= 12 and 1900 <= y <= 2100): return
    if (y, m) in seen: return
    lab = [k for k,v in _MON.items() if v==m][0].title() + str(y)[-2:]
    seen.add((y,m)); out.append((lab, y, m))

def extract_maturities(text: str, *, ref_date: date, pivot: int = 69) -> List[Tuple[str,int,int]]:
    t = text.lower()
    out: List[Tuple[str,int,int]] = []
    seen: Set[Tuple[int,int]] = set()

    # 1) MM/DD/YYYY or MM/DD/YY
    for m,d,y in re.findall(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", t):
        mm, dd, yy = int(m), int(d), int(y)
        yyyy = yy_to_yyyy(yy, pivot) if len(y)==2 else yy
        if 1<=mm<=12 and 1<=dd<=31: _add(out, seen, yyyy, mm)

    # 2) YYYY-MM or YYYY/MM
    for y,m in re.findall(r"\b(19\d{2}|20\d{2})[/-](\d{1,2})\b", t):
        _add(out, seen, int(y), int(m))

    # 3) Month + Year words: Jan26 / Jan 2026 / January 2026
    for mon,yr in re.findall(rf"\b({_MON_RE})\s*([0-9]{{2}}|[0-9]{{4}})\b", t):
        mm = _MON.get(mon[:3]);  y = int(yr)
        _add(out, seen, yy_to_yyyy(y, pivot) if len(yr)==2 else y, mm)

    # 4) Compact MMMYY: jan26
    for mon,yr in re.findall(rf"\b({_MON_RE})(\d{{2}})\b", t):
        _add(out, seen, yy_to_yyyy(int(yr), pivot), _MON[mon[:3]])

    # 5) Bare month (NOT followed by a 2- or 4-digit year). Strikes after are fine.
    for mon in re.findall(rf"\b({_MON_RE})\b(?!\s*(?:\d{{2}}(?!\d)|\d{{4}}))", t):
        mm = _MON[mon[:3]]
        yyyy = ref_date.year if mm >= ref_date.month else ref_date.year + 1
        _add(out, seen, yyyy, mm)

    return out


def parse_quantity(text: str) -> int:
    """
    Find quantity as an integer. Supports comma-grouped values (e.g., 1,000).
    Prefers 'buy|sell <N>' if present; otherwise first standalone integer.
    Run this on the ORIGINAL text (do NOT strip commas first).
    """
    # Prefer explicit verb + number
    m = re.search(r'\b(?:buy|sell)\s+(-?\d{1,3}(?:,\d{3})*|\d+)\b', text, flags=re.I)
    if not m:
        # First standalone integer, comma-aware
        m = re.search(r'\b-?\d{1,3}(?:,\d{3})*\b', text)
    if not m:
        return 1
    return int(m.group(1).replace(',', ''))

def parse_quantity(text: str) -> int:
    m = re.search(r"\b-?\d{1,3}(?:,\d{3})*\b", text)
    return int(m.group(0).replace(",", "")) if m else 1

def extract_strikes(text: str, qty: int | None = None) -> list[float]:
    # Prefer slash pairs (handles decimals)
    m = re.search(r'(?<![A-Za-z])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?!\S*[A-Za-z])', text)
    if m:
        return [float(m.group(1)), float(m.group(2))]
    # Otherwise standalone numbers (no commas), not glued to letters
    t = text.replace(",", "")
    nums = [float(x) for x in re.findall(r'(?<![A-Za-z])\d{2,5}(?:\.\d+)?\b', t)]
    # Drop the quantity if it was the first big number
    if qty and nums and int(nums[0]) == qty:
        nums = nums[1:]
    return nums



def parse_strategy_mvp(text: str,ref_date: date | None = None) -> dict:
    if ref_date is None:
        ref_date = date.today()   # evaluated each call 
    t = text.strip()
    qty = int(re.search(r'(-?\d[\d,]*)', t).group(1).replace(',', '')) if re.search(r'(-?\d[\d,]*)', t) else 1
    #qty=parse_quantity(t)
    sym = re.search(r'\b[A-Z]{1,10}\b', t).group(0) if re.search(r'\b[A-Z]{1,10}\b', t) else ""
    mats = re.findall(r'([A-Za-z]{3,9}\s*\d{2,4}|\d{1,2}/\d{1,2}/\d{2,4})', t)
    labels_year_month = extract_maturities(text,ref_date=ref_date)  # [(label, year, month), ...]
    maturities = [lab for (lab, _, _) in labels_year_month]  # keep simple labels for MVP

    strikes = [float(x) for x in  re.findall(r'(?<![A-Za-z])\d{2,5}(?:\.\d+)?\b',t)]
    strikes=extract_strikes(t,qty)
    ratio = (re.search(r'\b\d+(x\d+)+\b', t) or re.search(r'\b\d+x\d+\b', t))
    ratio = ratio.group(0) if ratio else "1x1"
    struct = None
    for s in ["collar","rr","cs","ps","straddle","strangle","calendar","diagonal"]:
        if re.search(rf'\b{s}\b', t, re.I): struct = s.lower(); break
    cp_match = re.search(r'\b[CPcp]{2,}\b', t)
    structure = {"cp_string": cp_match.group(0).upper()} if cp_match and not struct else (struct or "custom")
    ok = bool(qty and sym and mats and strikes and structure)
    return {"ok": ok, "quantity": qty, "symbol": sym, "maturities": maturities,
            "strikes": strikes, "ratio": ratio, "structure": structure}



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