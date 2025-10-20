# pip install scipy
import numpy as np
from scipy.linalg import solve_banded

import pandas as pd


def american_pde_price_banded(
    S0, K, T, sigma,
    r_func, q_func=lambda t: 0.0,
    dividends=None,                 # [(t_years, cash), ...]
    cp='C',
    NS=400, NT=800,                 # you can often use NT ~ NS with Rannacher
    theta=0.5,                      # 0.5 = Crank–Nicolson
    rannacher=2,                    # first 2 steps implicit Euler to smooth payoff kink
    Smax=None,
    return_grid=False
):
    """FDM PDE with discrete cash dividends, early exercise, banded solver."""
    dividends = sorted(dividends or [], key=lambda x: x[0])

    # spot grid
    if Smax is None:
        D_sum = sum(d for _, d in dividends)
        base = max(S0, K)
        Smax = max(4*base, 2*base + 4*D_sum)
    S = np.linspace(0.0, Smax, NS+1).astype(np.float64)
    dS = S[1] - S[0]

    # payoff at maturity
    V = (S - K).clip(min=0.0) if cp.upper().startswith('C') else (K - S).clip(min=0.0)

    # time grid (include dividend times exactly)
    times = list(np.linspace(0.0, T, NT+1))
    for t, _ in dividends:
        if 0 < t < T: times.append(t)
    times = np.array(sorted(set(times)), dtype=np.float64)  # ascending
    div_amt = {t: 0.0 for t, _ in dividends if 0 < t <= T}
    for t, a in dividends:
        if 0 < t <= T: div_amt[t] = div_amt.get(t, 0.0) + float(a)
    div_times = set(div_amt.keys())

    # optional full grid storage
    V_grid = None
    if return_grid:
        V_grid = np.empty((len(times), NS+1), dtype=np.float64)
        V_grid[-1] = V

    # prealloc work arrays
    N = NS - 1                         # number of interior nodes
    Si = S[1:-1]
    lower = np.empty(N, dtype=np.float64)
    diag  = np.empty(N, dtype=np.float64)
    upper = np.empty(N, dtype=np.float64)
    rhs   = np.empty(N, dtype=np.float64)
    # banded matrix container (3, N): [upper; diag; lower]
    ab = np.zeros((3, N), dtype=np.float64)

    # backward time stepping
    steps_done = 0
    for j in range(len(times)-1, 0, -1):
        t_cur, t_prev = times[j], times[j-1]
        dt = t_cur - t_prev
        th = 1.0 if steps_done < rannacher else theta
        t_mid = t_prev + th*dt

        r = float(r_func(t_mid)); q = float(q_func(t_mid))
        sig2 = float(sigma)*float(sigma)

        A = 0.5 * sig2 * (Si**2) / (dS**2)
        B = (r - q) * Si / (2.0*dS)

        # LHS (theta part)
        lower[:] = -th * dt * (A - B)   # length N
        diag[:]  =  1.0 + th * dt * (2.0*A + r)
        upper[:] = -th * dt * (A + B)

        # RHS ((1-theta) part)
        lower_r = (1.0 - th) * dt * (A - B)
        diag_r  = 1.0 - (1.0 - th) * dt * (2.0*A + r)
        upper_r = (1.0 - th) * dt * (A + B)

        # boundaries
        if cp.upper().startswith('C'):
            V0, VN = 0.0, S[-1]-K
        else:
            V0, VN = K, 0.0

        rhs[:] = lower_r * V[:-2] + diag_r * V[1:-1] + upper_r * V[2:]
        rhs[0]  -= lower[0] * V0
        rhs[-1] -= upper[-1] * VN

        # build banded tri-diagonal:
        # ab[0,1:] = upper[:-1], ab[1,:] = diag, ab[2,:-1] = lower[1:]
        ab[0, 1:] = upper[:-1]
        ab[1, :]  = diag
        ab[2, :-1]= lower[1:]
        # solve
        V_inner = solve_banded((1, 1), ab, rhs, overwrite_ab=True, overwrite_b=True, check_finite=False)

        V[0], V[-1] = V0, VN
        V[1:-1] = V_inner

        # early exercise
        if cp.upper().startswith('C'):
            np.maximum(V, S - K, out=V)
        else:
            np.maximum(V, K - S, out=V)

        # discrete dividend jump at t_prev
        if t_prev in div_times:
            D = div_amt[t_prev]
            S_shift = np.clip(S - D, 0.0, S[-1])
            V[:] = np.interp(S_shift, S, V)
            # re-enforce exercise right before ex-div
            if cp.upper().startswith('C'):
                np.maximum(V, S - K, out=V)
            else:
                np.maximum(V, K - S, out=V)

        if return_grid:
            V_grid[j-1] = V

        steps_done += 1

    price = float(np.interp(S0, S, V))
    debug = {"S_grid": S, "times": times}
    if return_grid:
        debug["V_grid"] = V_grid
    else:
        debug["V0"] = V
    return price, debug



def american_pde_price_and_greeks(
    S0, K, T, sigma,
    r_func, q_func=lambda t: 0.0,
    dividends=None,
    cp='C',
    NS=400, NT=800,
    theta=0.5,            # CN scheme weight (not the Greek)
    rannacher=2,
    Smax=None,
    return_grid=False,
    vol_bump=0.01,        # 1 vol point (absolute, i.e., 0.01 = 1%)
    rate_bump=1e-4        # 1 basis point
):
    # --- base run with grid so we can read theta from the first time step ---
    p0, dbg0 = american_pde_price_banded(
        S0, K, T, sigma, r_func, q_func, dividends, cp,
        NS, NT, theta, rannacher, Smax, return_grid=True
    )
    Sg      = dbg0["S_grid"]
    times   = dbg0["times"]
    V_grid  = dbg0["V_grid"]         # time x spot
    Vt0     = V_grid[0]              # t=0 slice

    # --- delta/gamma from t=0 slice (your existing approach) ---
    def _delta_gamma_from_grid(S, V, S0, h=None):
        S = np.asarray(S); V = np.asarray(V)
        dS = S[1]-S[0]; h = max(dS, 1e-4*max(S0,1.0)) if h is None else h
        Sm, Sp = max(S[0], S0-h), min(S[-1], S0+h)
        Vm, V0, Vp = np.interp([Sm, S0, Sp], S, V)
        delta = (Vp - Vm) / (Sp - Sm)
        h_eff = max(1e-12, 0.5*(Sp-Sm))
        gamma = (Vp - 2*V0 + Vm) / (h_eff**2)
        return float(delta), float(gamma)

    delta, gamma = _delta_gamma_from_grid(Sg, Vt0, S0)

    # helper: price at S0 from a time-slice
    def _p_from_slice(Vslice):
        return float(np.interp(S0, Sg, Vslice))

    # --- theta (calendar dV/dt at t=0) from first time step on the grid ---
    if len(times) > 1:
        dt = times[1] - times[0]
        p_t0 = _p_from_slice(V_grid[0])
        p_t1 = _p_from_slice(V_grid[1])
        theta_g = (p_t1 - p_t0) / dt         # per year; per-day = theta_g/365
    else:
        theta_g = float("nan")               # degenerate grid

    # --- vega via central bump in sigma (be nice if sigma is tiny) ---
    dsig = min(max(1e-4, 0.5*sigma), abs(vol_bump))
    if sigma > dsig:
        p_up, _ = american_pde_price_banded(S0,K,T,sigma+dsig, r_func,q_func,dividends,cp,NS,NT,theta,rannacher,Smax,False)
        p_dn, _ = american_pde_price_banded(S0,K,T,sigma-dsig, r_func,q_func,dividends,cp,NS,NT,theta,rannacher,Smax,False)
        vega = (p_up - p_dn) / (2.0*dsig)
    else:  # forward diff fallback if sigma is too small
        p_up, _ = american_pde_price_banded(S0,K,T,sigma+dsig, r_func,q_func,dividends,cp,NS,NT,theta,rannacher,Smax,False)
        vega = (p_up - p0) / dsig

    # --- rho via parallel rate shift r(t) ± dr ---
    dr = abs(rate_bump)
    r_up = (lambda t, rf=r_func, bump=dr: rf(t) + bump)
    r_dn = (lambda t, rf=r_func, bump=dr: rf(t) - bump)
    p_r_up, _ = american_pde_price_banded(S0,K,T,sigma, r_up,q_func,dividends,cp,NS,NT,theta,rannacher,Smax,False)
    p_r_dn, _ = american_pde_price_banded(S0,K,T,sigma, r_dn,q_func,dividends,cp,NS,NT,theta,rannacher,Smax,False)
    rho = (p_r_up - p_r_dn) / (2.0*dr)

    out = {
        "price": p0,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,          # per 1.00 (absolute) vol; per 1% vol = vega*0.01
        "theta": theta_g,      # per year; per day = theta_g/365
        "rho": rho             # per 1.00 change in rate; per bp = rho*1e-4
    }

    # pass grid back only if the caller asked for it
    if return_grid:
        out["grid_S"] = Sg
        out["grid_times"] = times
        out["grid_V"] = V_grid

    return out




# --- helpers -------------------------------------------------------
def _to_dt(x): return pd.to_datetime(x, infer_datetime_format=True)
def _yearfrac(d0, d1): return (pd.Timestamp(d1) - pd.Timestamp(d0)).days / 365.0

def _vol_to_float(v):
    if isinstance(v, str):
        v = v.strip()
        if v.endswith("%"): return float(v[:-1]) / 100.0
        return float(v)
    v = float(v)
    return v/100.0 if v > 1.5 else v  # treat 36 => 0.36

def _const_fn(x): return (lambda t, _x=float(x): _x)

# --- main: take option + dividend DataFrames and return priced table ----------
def price_table_with_divs(
    options_df: pd.DataFrame,
    dividends_df: pd.DataFrame,
    valuation_date=None,
    r=0.05, q=0.0,                               # constants or callables
    NS=400, NT=800, scheme_theta=.5, rannacher=4, Smax=None
):
    """
    options_df columns (case-insensitive):
      mult, Expiration, Strike, Spot, CP, EA, Texp, Vol
      - CP: 'C' or 'P'
      - EA: 'A' (american) or 'E' (european)  [american assumed unless you add the small flag in PDE]
      - Texp (years) optional; if missing we use Expiration vs valuation_date.
      - Vol accepts 0.36, '36%', or 36
    dividends_df columns:
      Date, Amount   (ex-div cash)
    """
    val_date = pd.Timestamp.today().normalize() if valuation_date is None else _to_dt(valuation_date)

    # normalize column names
    opt = options_df.copy()
    opt.columns = [c.strip().lower() for c in opt.columns]
    divs = dividends_df.copy()
    divs.columns = [c.strip().lower() for c in divs.columns]
    if 'date' not in divs or 'amount' not in divs:
        raise ValueError("dividends_df must have columns: Date, Amount")

    # precompute dividend times (years from valuation date)
    divs['t_years'] = (_to_dt(divs['date']) - val_date).dt.days / 365.0

    # constant or callable r/q
    r_func = r if callable(r) else _const_fn(r)
    q_func = q if callable(q) else _const_fn(q)

    results = []

    for _, row in opt.iterrows():
        S0 = float(row['spot'])
        K  = float(row['strike'])
        borrow=float(row['borrow'])
        cp = 'C' if str(row.get('cp','C')).upper().startswith('C') else 'P'
        mult = float(row.get('mult', 100))

        # expiry in years
        if 'texp' in row and pd.notna(row['texp']):
            T = float(row['texp'])
        else:
            if 'expiration' not in row or pd.isna(row['expiration']):
                raise ValueError("Need either Texp or Expiration")
            T = _yearfrac(val_date, _to_dt(row['expiration']))

        sigma = _vol_to_float(row['vol'])
        q_func = q if callable(q) else _const_fn(q+borrow)
        #q_func = q_func + borrow
        #q_func = lambda t: q(t)+borrow if callable(q) else _const_fn(q+borrow)
        # dividends for THIS option (0 < t <= T)
        div_list = [(float(t), float(a)) for t, a in zip(divs['t_years'], divs['amount']) if 0.0 < t <= T]

        # --- call your PDE pricer (assumes you already defined it above) ----
        # american_pde_price_and_greeks: from your earlier code
        out = american_pde_price_and_greeks(
            S0, K, T, sigma,
            r_func=r_func, q_func=q_func,
            dividends=div_list, cp=cp,
            NS=NS, NT=NT, theta=scheme_theta, rannacher=rannacher, Smax=Smax,
            return_grid=False
        )

        # scale by contract multiplier; also give convenient 01s
        res = {
            'Price':     out['price'],
            'delta':     out['delta'],
            'gamma':     out['gamma'],
            'vega':      out['vega'],
            'theta':     out['theta'],
            'rho':       out['rho'],
            'Theo':      out['price'] * mult,
            'Delta':     out['delta'] * mult,
            'Gamma':     out['gamma'] * mult,                         # per $1
            'Vega':      out['vega']  * mult,                         # per 1.00 vol
            'Vega01':    out['vega']  * 0.01 * mult,                  # per 1 vol point
            'Theta_yr':  out['theta'] * mult,                          # per year
            'Theta_day': out['theta'] / 365.0 * mult,                 # per calendar day
            'Rho':       out['rho']   * mult,                         # per 1.00 rate
            'Rho01':     out['rho']   * 1e-4 * mult                   # per 1bp
        }

        results.append(res)

    res_df = pd.concat([opt.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    return res_df