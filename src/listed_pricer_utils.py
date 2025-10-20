from datetime import date, datetime
import pandas as pd

try:
    import QuantLib as ql
    _HAS_QL = True
except Exception:
    _HAS_QL = False

def _to_iso_date_key(x) -> str:
    """Return 'YYYY-MM-DD' for ql.Date | datetime/date | str.
       Supports 'YYYY-MM-DD', 'M/D/YY', 'M/D/YYYY'."""
    # QuantLib Date
    if _HAS_QL and isinstance(x, ql.Date):
        return date(x.year(), x.month(), x.dayOfMonth()).isoformat()
    # pandas/py datetime or date
    if isinstance(x, (datetime, date)):
        return date(x.year, x.month, x.day).isoformat() if isinstance(x, datetime) else x.isoformat()
    # strings
    if isinstance(x, str):
        s = x.strip()
        # fast paths: ISO → done
        if "-" in s:
            return pd.to_datetime(s, errors="raise").date().isoformat()
        # slash format: try m/d/yy then m/d/yyyy
        for fmt in ("%m/%d/%y", "%m/%d/%Y"):
            try:
                return datetime.strptime(s, fmt).date().isoformat()
            except ValueError:
                pass
        # last resort: let pandas guess
        return pd.to_datetime(s, errors="raise").date().isoformat()
    # unknown → error
    raise TypeError(f"Unsupported date type: {type(x)}")