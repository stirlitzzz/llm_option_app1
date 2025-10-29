import QuantLib as ql
import math
import pandas as pd, numpy as np
from datetime import datetime

def zero_curve_from_csv(valuation_date: ql.Date, rows, day_count=ql.Actual365Fixed(), cal=ql.UnitedStates(ql.UnitedStates.NYSE)):
    """
    rows: iterable of (T_years, r_cont_zero). Builds a ZeroCurve dated from valuation_date.
    """
    dates = [valuation_date] + [valuation_date + ql.Period(int(round(T*365)), ql.Days) for T, _ in rows]
    rates = [rows[0][1]] + [float(r) for _, r in rows]  # pad front; QL expects >= 2 nodes
    return ql.YieldTermStructureHandle(ql.ZeroCurve(dates, rates, day_count, cal))

class QLAmericanPricer:
    def __init__(self, valuation_date: ql.Date, r_curve: ql.YieldTermStructureHandle,
                 init_vol: float = 0.20, cal=ql.UnitedStates(ql.UnitedStates.NYSE), dc=ql.Actual365Fixed(),
                 t_grid: int = 400, x_grid: int = 200):
        ql.Settings.instance().evaluationDate = valuation_date
        self.val_date = valuation_date
        self.cal, self.dc = cal, dc
        self.t_grid, self.x_grid = t_grid, x_grid

        # live quotes you can bump
        self.spot_q = ql.SimpleQuote(0.0)
        self.vol_q  = ql.SimpleQuote(max(1e-8, init_vol))

        # handles
        self.u   = ql.QuoteHandle(self.spot_q)
        self.vts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(valuation_date, cal, ql.QuoteHandle(self.vol_q), dc))
        self.rts = r_curve
        self.dts = ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, 0.0, dc, ql.Continuous))  # cont. q=0; use cash divs explicitly

        # stochastic process reused across prices
        self.process = ql.BlackScholesMertonProcess(self.u, self.dts, self.rts, self.vts)

    def _div_schedule(self, dividends):
        sched = []
        if not dividends: return sched
        for d, amt in dividends:
            dq = d if isinstance(d, ql.Date) else ql.Date(d.day, d.month, d.year)
            if dq > self.val_date:
                sched.append(ql.FixedDividend(float(amt), dq))
        return sched

    def price(self, S, K, maturity: ql.Date, cp: str,
            vol: float | None = None, dividends: list[tuple] | None = None,
            want_greeks: bool = True, vega_eps: float = 1e-4,
            borrow: float | None = None):          # <= NEW

        # live quotes
        self.spot_q.setValue(float(S))
        if vol is not None:
            self.vol_q.setValue(max(1e-8, float(vol)))

        payoff   = ql.PlainVanillaPayoff(ql.Option.Call if cp.upper().startswith('C') else ql.Option.Put, float(K))
        exercise = ql.AmericanExercise(self.val_date, maturity)
        option   = ql.VanillaOption(payoff, exercise)

        # --- per-option borrow: build a LOCAL dividend-yield term structure ---
        # Convention: borrow>0 lowers forward like a dividend yield q (i.e., reduces C-P).
        if borrow is not None:
            dts_local = ql.YieldTermStructureHandle(
                ql.FlatForward(self.val_date, float(borrow), self.dc, ql.Continuous)
            )
            process_local = ql.BlackScholesMertonProcess(self.u, dts_local, self.rts, self.vts)
        else:
            dts_local = self.dts  # whatever is set globally (often 0)
            process_local = self.process

        divs = self._div_schedule(dividends or [])

        # --- Engine with explicit cash dividends (preferred path) ---
        engine = None
        try:
            engine = ql.FdBlackScholesVanillaEngine(process_local, divs, self.t_grid, self.x_grid)
        except TypeError:
            # Fallback: escrow PV of discrete divs out of spot if engine signature without schedule is hit
            pv = sum(d.amount() * self.rts.discount(d.date()) for d in divs)
            self.spot_q.setValue(float(S) - pv)
            engine = ql.FdBlackScholesVanillaEngine(process_local, self.t_grid, self.x_grid)

        option.setPricingEngine(engine)

        out = {"theo": option.NPV()}
        if not want_greeks:
            # restore spot if escrow used
            self.spot_q.setValue(float(S))
            return out

        # Greeks (engine-provided if available)
        for g in ("delta", "gamma", "theta"):
            try:
                out[g] = getattr(option, g)()
            except RuntimeError:
                out[g] = None

        # Vega (fallback numeric if engine lacks it)
        try:
            out["vega"] = option.vega()
        except RuntimeError:
            v0 = self.vol_q.value()
            self.vol_q.setValue(v0 + vega_eps)
            p_up = ql.VanillaOption(payoff, exercise); p_up.setPricingEngine(engine); upv = p_up.NPV()
            self.vol_q.setValue(v0 - vega_eps)
            p_dn = ql.VanillaOption(payoff, exercise); p_dn.setPricingEngine(engine); dnv = p_dn.NPV()
            self.vol_q.setValue(v0)
            out["vega"] = ((upv - dnv) / (2*vega_eps)) * 0.01  # per 1 vol point

        # restore spot if escrow path
        self.spot_q.setValue(float(S))
        return out



# ---------- helpers ----------
def _to_qldate(s: str) -> ql.Date:
    # supports 'm/d/yy' or 'm/d/yyyy'
    fmt = '%m/%d/%y' if len(s.rsplit('/',1)[-1])<=2 else '%m/%d/%Y'
    d = datetime.strptime(s, fmt)
    return ql.Date(d.day, d.month, d.year)

def _flat_curve(valuation_date: str | ql.Date, r_cont: float,
                dc=ql.Actual365Fixed()) -> ql.YieldTermStructureHandle:
    vd = _to_qldate(valuation_date) if isinstance(valuation_date, str) else valuation_date
    return ql.YieldTermStructureHandle(ql.FlatForward(vd, r_cont, dc, ql.Continuous))

def _div_list(dividends_df: pd.DataFrame) -> list[tuple]:
    # returns [(ql.Date, amount), ...] sorted
    out = []
    for d, a in zip(dividends_df['date'], dividends_df['amount']):
        dq = _to_qldate(str(d)) if not isinstance(d, ql.Date) else d
        out.append((dq, float(a)))
    out.sort(key=lambda x: (int(x[0].serialNumber()),))
    return out


def _parse_pct_or_float(x):
    if isinstance(x, str) and x.strip().endswith('%'):
        return float(x.strip().strip('%'))/100.0
    return float(x)

# ---------- wrapper ----------
def price_table_with_divs_ql(options_df: pd.DataFrame,
                             dividends_df: pd.DataFrame,
                             valuation_date='10/17/2025',
                             default_r=0.0308,
                             init_vol=0.20,
                             t_grid=400, x_grid=200,
                             want_greeks=True,
                             compute_combo=True):
    """
    Uses QLAmericanPricer.price(...) for each row in options_df.
    Expects columns: ['expiration','strike','spot','cp','vol'] (Vol may be '27.5%' or 0.275)
                     Optional: 'mult'
    Discrete cash dividends provided in dividends_df: columns ['Date','Amount'].
    """
    # Build curve & pricer
    val_qldate = _to_qldate(valuation_date)
    r_curve = _flat_curve(val_qldate, default_r)
    pricer = QLAmericanPricer(val_qldate, r_curve, init_vol=init_vol, t_grid=t_grid, x_grid=x_grid)


    rate_col = None
    for cand in ['Rate', 'r', 'rate', 'risk_free']:
        if cand in options_df.columns:
            rate_col = cand
            break
    # cache pricers keyed Rby rate to avoid rebuilding for every row
    pricer_cache: dict[float, "QLAmericanPricer"] = {}
    pricer_cache[default_r] = pricer
    # Normalize dividends to list of (ql.Date, amt)
    divs = _div_list(dividends_df)

    rows = []
    for i, row in options_df.iterrows():
        # Parse row fields
        S  = float(row['spot'])
        K  = float(row['strike'])
        cp = str(row['cp']).upper()[0]
        exp = row['expiration']
        mat = _to_qldate(str(exp)) if not isinstance(exp, ql.Date) else exp
        borrow = float(row['borrow'])

        vol_raw = row['vol']
        if isinstance(vol_raw, str) and vol_raw.strip().endswith('%'):
            vol = float(vol_raw.strip('%'))/100.0
        else:
            vol = float(vol_raw)

        r_this = _parse_pct_or_float(row[rate_col]) if rate_col else float(default_r)
        pr = pricer_cache.get(r_this)
        if pr is None:
            r_curve = _flat_curve(val_qldate, r_this)  # still continuous comp.
            pr = QLAmericanPricer(val_qldate, r_curve, init_vol=init_vol, t_grid=t_grid, x_grid=x_grid)
            pricer_cache[r_this] = pr

        # Price with your engine
        res = pr.price(S=S, K=K, maturity=mat, cp=cp, vol=vol, dividends=divs,
                           want_greeks=want_greeks,borrow=borrow)

        rows.append({
            **row.to_dict(),
            'theo': res['theo'],
            'delta': res.get('delta'),
            'gamma': res.get('gamma'),
            'theta': res.get('theta'),
            'vega':  res.get('vega')
        })

    priced = pd.DataFrame(rows)

    if compute_combo:
        # C - P per (Expiration, Strike); assumes both legs exist in table
        grp_cols = ['expiration', 'strike']
        # pivot to wide then compute combo
        wide = priced.pivot_table(index=grp_cols, columns='cp', values='theo', aggfunc='first')
        wide['Combo_C_minus_P'] = wide.get('C', np.nan) - wide.get('P', np.nan)
        priced = priced.merge(wide[['Combo_C_minus_P']].reset_index(), on=grp_cols, how='left')

    return priced