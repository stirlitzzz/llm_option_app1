# dividends.py
import pandas as pd
from pandas.tseries.offsets import DateOffset

ALLOWED_FREQ_STEPS = {1:12, 2:6, 3:4, 4:3, 6:2, 12:1}

def _roll_to_business_day(ts):
    ts = pd.Timestamp(ts)
    while ts.weekday() >= 5:
        ts = ts + pd.Timedelta(days=1)
    return ts

def _parse_mmdd_list(dates_like, base_year):
    out = []
    for x in dates_like:
        ts = x if isinstance(x, pd.Timestamp) else pd.to_datetime(str(x), errors="raise")
        if ts.year == 1970:  # unlikely; but keep consistent with seeds
            ts = pd.to_datetime(f"{base_year}-{ts.strftime('%m-%d')}")
        out.append(ts.strftime("%m-%d"))
    return sorted(out)

def _generate_template_from_first(mmdd_first, freq, base_year):
    step = ALLOWED_FREQ_STEPS.get(freq)
    if step is None:
        raise ValueError(f"Unsupported frequency {freq}. Use one of {sorted(ALLOWED_FREQ_STEPS)}.")
    start = pd.Timestamp(f"{base_year}-{mmdd_first}")
    return sorted((start + DateOffset(months=step*k)).strftime("%m-%d") for k in range(freq))

def _complete_template(mmdd_list, freq, base_year):
    step = ALLOWED_FREQ_STEPS.get(freq)
    if step is None:
        raise ValueError(f"Unsupported frequency {freq}. Use one of {sorted(ALLOWED_FREQ_STEPS)}.")
    seeds = sorted(mmdd_list)
    if len(seeds) >= freq:
        return seeds[:freq]
    if len(seeds) == 1:
        return _generate_template_from_first(seeds[0], freq, base_year)
    cur = pd.Timestamp(f"{base_year}-{seeds[-1]}")
    out = seeds[:]
    while len(out) < freq:
        cur = cur + DateOffset(months=step)
        out.append(cur.strftime("%m-%d"))
    return sorted(out)

def forecast_dividends_from_history(
    df_history: pd.DataFrame,
    years_ahead=3,
    increase_dollars=0.05,
    ref_year=None,
    date_field="ex_dividend_date",
    use_business_day_roll=True,
    frequency_override=None,         # 1,2,3,4,6,12
    first_dates_override=None        # ["MM-DD", ...] or full dates
) -> pd.DataFrame:
    if df_history is None or df_history.empty:
        raise ValueError("df_history is empty")
    df = df_history.copy()
    for col in ["ex_dividend_date","pay_date","declaration_date","record_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.sort_values(date_field).reset_index(drop=True)
    if date_field not in df.columns:
        raise ValueError(f"`date_field='{date_field}'` not in history columns {df.columns.tolist()}")

    today = pd.Timestamp.today().tz_localize(None)
    if ref_year is None:
        ref_year = today.year - 1

    #last_known_year = int(df[date_field].dropna().max().year)
    #first_forecast_year = last_known_year + 1
    last_known_date = df[date_field].max()
    freq = len(template_mmdd)
    step_months = 12 // freq
    first_forecast_date = last_known_date + pd.DateOffset(months=step_months)
    # Build yearly template
    if first_dates_override or frequency_override:
        seeds = _parse_mmdd_list(first_dates_override or
                                 df[df[date_field].dt.year == ref_year][date_field].dt.strftime("%m-%d").tolist(),
                                 base_year=first_forecast_year)
        freq = int(frequency_override) if frequency_override else len(seeds)
        template_mmdd = _complete_template(seeds, freq, base_year=first_forecast_year)
    else:
        df_ref = df[df[date_field].dt.year == ref_year].dropna(subset=[date_field])
        if df_ref.empty:
            raise ValueError(f"No {date_field} rows in reference year {ref_year}.")
        template_mmdd = sorted(df_ref[date_field].dt.strftime("%m-%d").tolist())

    # Increase anchor: first in-year increase
    diffs = df["cash_amount"].astype(float).diff()
    in_ref = df[date_field].dt.year == ref_year
    inc_rows = df[in_ref & (diffs > 0)]
    mapped_anchor = None
    if not inc_rows.empty:
        anchor_mmdd = inc_rows.iloc[0][date_field].strftime("%m-%d")
        mapped_anchor = anchor_mmdd if anchor_mmdd in template_mmdd else \
            next((d for d in template_mmdd if d >= anchor_mmdd), template_mmdd[0])

    current_amount = float(df.iloc[-1]["cash_amount"])
    out = []
    for i in range(1, years_ahead + 1):
        year = last_known_year + i
        for mmdd in template_mmdd:
            month, day = map(int, mmdd.split("-"))
            dt = pd.Timestamp(year=year, month=month, day=day)
            if use_business_day_roll:
                dt = _roll_to_business_day(dt)
            inc_applied = False
            if mapped_anchor and mmdd == mapped_anchor:
                current_amount += increase_dollars
                inc_applied = True
            elif not mapped_anchor and i >= 2 and mmdd == template_mmdd[0]:
                current_amount += increase_dollars
                inc_applied = True
            out.append({date_field: dt, "cash_amount": round(float(current_amount), 4), "increase_applied": inc_applied, "source": "forecast"})
    return pd.DataFrame(out).sort_values(date_field).reset_index(drop=True)

import calendar as _cal
from typing import Iterable

def _same_weekday_anniversary(ts: pd.Timestamp, target_year: int, mode: str = "nearest") -> pd.Timestamp:
    """Project `ts` to `target_year`, adjusting so the weekday matches `ts.weekday()`.
    mode: 'nearest' | 'following' | 'preceding' (tie -> following)
    """
    ts = pd.Timestamp(ts)
    m, d = ts.month, ts.day
    last_dom = _cal.monthrange(target_year, m)[1]
    d = min(d, last_dom)
    base = pd.Timestamp(year=target_year, month=m, day=d)
    want = ts.weekday()  # 0=Mon..6=Sun
    have = base.weekday()
    fwd_shift = (want - have) % 7
    back_shift = -((have - want) % 7)
    cand_fwd = base + pd.Timedelta(days=fwd_shift)
    cand_back = base + pd.Timedelta(days=back_shift)
    if mode == "following":
        return cand_fwd
    if mode == "preceding":
        return cand_back
    # nearest (tie -> following)
    return cand_fwd if abs(fwd_shift) <= abs(back_shift) else cand_back


def _project_year_dates_same_weekday(prev_year_dates: Iterable[pd.Timestamp], target_year: int, mode: str = "nearest") -> list[pd.Timestamp]:
    dates = sorted(pd.to_datetime(list(prev_year_dates)).tolist())
    return [_same_weekday_anniversary(pd.Timestamp(dt), target_year, mode=mode) for dt in dates]


def forecast_dividends_same_weekday(
    df_history: pd.DataFrame,
    *,
    years_ahead: int = 3,
    increase_dollars: float = 0.05,
    date_field: str = "ex_dividend_date",
    keep_same_weekday: bool = True,   # kept for clarity; this implementation always preserves weekday
    weekday_mode: str = "nearest",    # "nearest" | "following" | "preceding"
    roll_weekends: bool = False,      # if True, weekend-roll to Monday (may change weekday)
) -> pd.DataFrame:
    """
    Forecast dividends by projecting each year's *actual* ex-div dates forward one year,
    preserving the weekday of each event.

    Strategy:
      - Fill the remainder of the current year by projecting the most recent *completed* year.
      - For each future year Y, project the dates from Y-1 to Y keeping the same weekday.
      - Apply annual increases on the mapped anchor date.

    Expects df_history columns: [date_field, cash_amount, (optional) ticker].
    """

    if df_history is None or df_history.empty:
        raise ValueError("df_history is empty")

    df = df_history.copy()
    df[date_field] = pd.to_datetime(df[date_field], errors="coerce")
    if "cash_amount" not in df.columns:
        raise ValueError("df_history must contain 'cash_amount'")
    df["cash_amount"] = pd.to_numeric(df["cash_amount"], errors="coerce").astype(float)
    df = df.dropna(subset=[date_field, "cash_amount"]).sort_values(date_field).reset_index(drop=True)

    last_known_date = pd.Timestamp(df[date_field].iloc[-1])
    last_year = int(last_known_date.year)
    have_ticker = "ticker" in df.columns
    ticker_val = df["ticker"].iloc[-1] if have_ticker else None

    # Optional weekend roll helper
    def _weekend_roll(ts: pd.Timestamp) -> pd.Timestamp:
        if not roll_weekends:
            return ts
        while ts.weekday() >= 5:
            ts = ts + pd.Timedelta(days=1)
        return ts

    # Determine the most recent fully-completed reference year (< last_year)
    counts = df[date_field].dt.year.value_counts().sort_index()
    completed_years = [y for y in counts.index if y < last_year]
    if completed_years:
        max_count = counts.loc[completed_years].max()
        ref_full_year = int(max([y for y in completed_years if counts.loc[y] == max_count]))
    else:
        ref_full_year = int(counts.index.min())  # fallback

    def _year_dates(year: int) -> list[pd.Timestamp]:
        return sorted(pd.to_datetime(df.loc[df[date_field].dt.year == year, date_field]).dropna().tolist())

    # Detect latest historical increase anchor
    df_sorted = df.sort_values(date_field).reset_index(drop=True)
    diffs_amt = df_sorted["cash_amount"].diff()
    inc_idx = diffs_amt[diffs_amt > 0].index
    inc_dt = None
    if len(inc_idx) > 0:
        inc_dt = pd.Timestamp(df_sorted.loc[inc_idx[-1], date_field])

    current_amount = float(df["cash_amount"].iloc[-1])
    rows: list[dict] = []

    # --- Remainder of current year: project ref_full_year -> last_year ---
    seeds_prev = _year_dates(ref_full_year)
    if not seeds_prev:
        raise ValueError(f"No historical dates in reference year {ref_full_year}")

    curr_year_full = _project_year_dates_same_weekday(seeds_prev, last_year, mode=weekday_mode)
    for dt in sorted(curr_year_full):
        dt = _weekend_roll(dt)
        if dt > last_known_date:
            row = {date_field: dt, "cash_amount": round(current_amount, 4), "increase_applied": False, "source": "forecast"}
            if have_ticker:
                row["ticker"] = ticker_val
            rows.append(row)

    # --- Future full years: project Y-1 -> Y each time ---
    prev_year_dates = curr_year_full
    for Y in range(last_year + 1, last_year + 1 + years_ahead):
        year_dates = _project_year_dates_same_weekday(prev_year_dates, Y, mode=weekday_mode)
        year_dates = [_weekend_roll(pd.Timestamp(d)) for d in sorted(year_dates)]

        # Map increase anchor to this year's schedule
        anchor_target = None
        if inc_dt is not None:
            nominal_anchor = _same_weekday_anniversary(inc_dt, Y, mode=weekday_mode)
            anchor_target = min(year_dates, key=lambda d: abs((d - nominal_anchor).days))

        for dt in year_dates:
            inc_applied = False
            if anchor_target is not None and dt == anchor_target:
                current_amount += float(increase_dollars)
                inc_applied = True
            row = {date_field: dt, "cash_amount": round(current_amount, 4), "increase_applied": inc_applied, "source": "forecast"}
            if have_ticker:
                row["ticker"] = ticker_val
            rows.append(row)

        prev_year_dates = year_dates

    out = pd.DataFrame(rows).sort_values(date_field).reset_index(drop=True)
    return out

def to_quantlib_dividend_schedule(df: pd.DataFrame, date_col="ex_dividend_date",
                                  amount_col="cash_amount", use_pay_date=False):
    """Return (schedule, [(ql.Date, float)]) for convenience."""
    import pandas as pd
    try:
        import QuantLib as ql
    except ImportError as e:
        raise ImportError("QuantLib not installed. `pip install QuantLib`") from e

    col = "pay_date" if use_pay_date and "pay_date" in df.columns else date_col
    sched, sched_tuples = [], []

    for _, r in df.dropna(subset=[col, amount_col]).iterrows():
        ts = pd.Timestamp(r[col]).to_pydatetime().date()
        qd = ql.Date(ts.day, ts.month, ts.year)
        amt = float(r[amount_col])
        sched_tuples.append((qd, amt))
        sched.append(ql.FixedDividend(amt, qd))

    return sched, sched_tuples