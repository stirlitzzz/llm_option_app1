import pandas as pd

from listed_pricer_data_loader import get_ivol_yc, apply_ivol_yc, fetch_and_pick
from listed_pricer_qlib import price_table_with_divs_ql
from listed_pricer_utils import _to_iso_date_key


def tag_source(df: pd.DataFrame, prefix: str, keys=("expiration","strike","cp")) -> pd.DataFrame:
    ren = {c: f"{prefix}{c}" for c in df.columns if c not in keys}
    return df.rename(columns=ren)


def run_pipeline(pd_val_date: pd.Timestamp,
                 options_df: pd.DataFrame,
                 dividends_df: pd.DataFrame,
                 underlying: str):
    """
    End-to-end pipeline used by the notebook UI.

    Returns:
      df_show, combined_full, yc, mkt
    """

    # 0) Helpers ---------------------------------------------------------------
    def _as_mmddyyyy(d: pd.Timestamp) -> str:
        return d.strftime('%m/%d/%Y')

    def _as_iso(d: pd.Timestamp) -> str:
        return d.strftime('%Y-%m-%d')

    # 1) Build vol/yield context + apply to options ---------------------------
    yc = get_ivol_yc(pd_val_date)
    opts2 = apply_ivol_yc(yc, options_df)
    legs_df = opts2[["expiration","strike","cp"]]

    # 2) Price with QL (readable, per-row flat r) -----------------------------
    priced_ql_ir = price_table_with_divs_ql(
        opts2, dividends_df,
        valuation_date=_as_mmddyyyy(pd_val_date),
        default_r=0.0308,
        init_vol=0.2755,
        t_grid=400, x_grid=200,
        want_greeks=True,
        compute_combo=True
    )
    priced_ql_ir["ValuationDate"] = pd_val_date

    # 3) Fetch market rows for requested legs ---------------------------------
    out = fetch_and_pick(
        underlying,
        trade_date=_as_iso(pd_val_date),
        legs_df=legs_df,
        snap_to_nearest=False
    )
    issues = out["issues"]
    mkt_rows = out["matched_df"]
    print(f"issues: {issues}")

    # 4) Normalize keys/cases on both sides -----------------------------------
    # priced/pricer side
    priced = priced_ql_ir.rename(columns={"CP": "cp"}).copy()

    # market side (keep only relevant columns)
    mkt = mkt_rows[[
        'expiration','strike','cp','bid','ask','iv','openInterest',
        'volume','delta','gamma','vega','underlying_price','optionId'
    ]].copy()

    priced = tag_source(priced, "mdl_")
    mkt = tag_source(mkt, "mkt_")

    priced["Exp_key"] = priced["expiration"].map(_to_iso_date_key)
    priced["strike"] = priced["strike"].astype(float).round(2)
    priced["cp"] = priced["cp"].str.upper().str[0]

    mkt["Exp_key"] = mkt["expiration"].map(_to_iso_date_key)
    mkt["strike"] = mkt["strike"].astype(float).round(2)
    mkt["cp"] = mkt["cp"].str.upper().str[0]

    # 5) Safe merge (one-to-one expected) -------------------------------------
    combined = priced.merge(
        mkt,
        on=['Exp_key','strike','cp'],
        how='left',
        validate='one_to_one'
    ).drop(columns=['Exp_key'])

    # 6) Display subset --------------------------------------------------------
    cols_show = [
        "expiration","strike","mdl_spot","mdl_cp","mdl_ea","mdl_texp",
        "mdl_vol","mdl_borrow","mkt_bid","mdl_theo","mkt_ask"
    ]
    cols_show = [c for c in cols_show if c in combined.columns]
    df_show = combined[cols_show].copy()

    return df_show, combined, yc, mkt


