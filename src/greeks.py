import pandas as pd


class GreekUnits:
    """
    Describe how source greeks are reported.
    We convert to a standard:
      - per share
      - vega per 1 vol point (0.01)
      - theta per day
      - gamma per $ (Δ per $1)
    """
    def __init__(self,
                 per_contract: bool = False,
                 contract_multiplier: float = 100.0,
                 vega_per_vol_point: bool = True,
                 theta_per_day: bool = True,
                 gamma_per_dollar: bool = True):
        self.per_contract = per_contract
        self.mult = contract_multiplier
        self.vega_per_vol_point = vega_per_vol_point
        self.theta_per_day = theta_per_day
        self.gamma_per_dollar = gamma_per_dollar


def standardize_greeks(df: pd.DataFrame,
                       units: GreekUnits,
                       prefix: str = "mdl_") -> pd.DataFrame:
    """
    Returns a copy with standardized greeks:
      mdl_delta_std, mdl_gamma_std, mdl_vega_std, mdl_theta_std, mdl_theo_std
    All on a per-share basis, vega per 0.01, theta per day, gamma per $.
    """
    d = df.copy()

    per_contract_factor = (1.0 / units.mult) if units.per_contract else 1.0

    if f"{prefix}delta" in d:
        d[f"{prefix}delta_std"] = d[f"{prefix}delta"] * per_contract_factor

    if f"{prefix}gamma" in d:
        d[f"{prefix}gamma_std"] = d[f"{prefix}gamma"] * per_contract_factor

    if f"{prefix}vega" in d:
        v = d[f"{prefix}vega"] * per_contract_factor
        if not units.vega_per_vol_point:
            v = v * 100.0
        d[f"{prefix}vega_std"] = v

    if f"{prefix}theta" in d:
        t = d[f"{prefix}theta"] * per_contract_factor
        if not units.theta_per_day:
            t = t / 365.0
        d[f"{prefix}theta_std"] = t

    if f"{prefix}theo" in d:
        d[f"{prefix}theo_std"] = d[f"{prefix}theo"] * per_contract_factor

    return d


