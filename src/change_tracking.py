import hashlib
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


# Public constants
KEYS = ("expiration", "strike", "cp")
NUM_COLS = ("qty", "vol", "borrow", "spot", "texp")
EPS = dict(abs=1e-9, rel=1e-6)


STATE = {"legs_prev": None, "params_prev": None, "step": 0}


def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Be explicit about formats to avoid per-row parser warnings
    try:
        d["expiration"] = pd.to_datetime(d["expiration"], errors="coerce")
        m = d["expiration"].isna()
        if m.any():
            d.loc[m, "expiration"] = pd.to_datetime(
                df.loc[m, "expiration"], format="%m/%d/%Y", errors="coerce"
            )
        d["expiration"] = pd.to_datetime(d["expiration"]).dt.date.astype(str)
    except Exception:
        d["expiration"] = pd.to_datetime(d["expiration"]).dt.date.astype(str)
    d["strike"] = d["strike"].astype(float).round(8)
    d["cp"] = d["cp"].astype(str).str.upper().str[0]
    return d


def _is_close(a, b, abs_eps=EPS["abs"], rel_eps=EPS["rel"]):
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        a, b = float(a), float(b)
        return abs(a - b) <= max(abs_eps, rel_eps * max(abs(a), abs(b), 1.0))
    except Exception:
        return a == b


def diff_legs(prev: Optional[pd.DataFrame], cur: pd.DataFrame):
    curN = _norm_keys(cur)
    if prev is None or len(prev) == 0:
        return {
            "added": curN.copy(),
            "removed": curN.iloc[0:0].copy(),
            "modified": curN.iloc[0:0].copy(),
        }

    prevN = _norm_keys(prev)

    pk_prev = prevN[list(KEYS)].astype(str).agg("|".join, axis=1)
    pk_cur = curN[list(KEYS)].astype(str).agg("|".join, axis=1)

    add_keys = set(pk_cur) - set(pk_prev)
    rem_keys = set(pk_prev) - set(pk_cur)
    com_keys = list(set(pk_cur) & set(pk_prev))

    added = curN[pk_cur.isin(add_keys)].reset_index(drop=True)
    removed = prevN[pk_prev.isin(rem_keys)].reset_index(drop=True)

    prevC = prevN[pk_prev.isin(com_keys)].set_index(list(KEYS))
    curC = curN[pk_cur.isin(com_keys)].set_index(list(KEYS))

    mods = []
    for k in curC.index:
        row_prev = prevC.loc[k]
        row_cur = curC.loc[k]
        changed = {}
        for c in NUM_COLS:
            if c in row_cur and c in row_prev and not _is_close(row_prev[c], row_cur[c]):
                changed[c] = (row_prev[c], row_cur[c])
        if changed:
            mods.append(
                {
                    **{kk: k[i] for i, kk in enumerate(KEYS)},
                    **{f"{c}_old": v[0] for c, v in changed.items()},
                    **{f"{c}_new": v[1] for c, v in changed.items()},
                }
            )
    modified = pd.DataFrame(mods)
    return {"added": added, "removed": removed, "modified": modified}


def diff_params(prev: Optional[dict], cur: dict):
    if not prev:
        return {k: (None, cur[k]) for k in cur.keys()}
    out = {}
    for k, v in cur.items():
        pv = prev.get(k, None)
        if isinstance(v, (int, float)) and isinstance(pv, (int, float)):
            if not _is_close(pv, v):
                out[k] = (pv, v)
        else:
            if pv != v:
                out[k] = (pv, v)
    return out


def hash_legs(df: pd.DataFrame):
    d = _norm_keys(df)[list(KEYS + ("qty", "vol", "borrow"))].astype(str)
    key = sorted(d.agg("|".join, axis=1).tolist())
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:10]


def _default_sink(path: str) -> Callable[[dict], None]:
    def _write(payload: dict):
        with open(path, "a") as f:
            f.write(json.dumps(payload) + "\n")
    return _write


def track_and_update(
    legs_cur: pd.DataFrame,
    params_cur: dict,
    log_path: Optional[str] = None,
    run_id: Optional[str] = None,
    sink: Optional[Callable[[dict], None]] = None,
):
    """
    Compare against last snapshot, return (legs_diff, params_diff, meta), then update STATE.

    Logging is optional and flexible:
      - If sink is provided, it will be called with the JSON record.
      - Else if log_path is provided, JSONL will be appended to that path.
      - Schema includes version and allows adding fields without breaking readers.
    """
    legs_diff = diff_legs(STATE["legs_prev"], legs_cur)
    params_diff = diff_params(STATE["params_prev"], params_cur)

    meta = {
        "step": STATE["step"] + 1,
        "legs_hash": hash_legs(legs_cur),
        "run_id": run_id,
    }

    if sink or log_path:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "schema_version": 1,
            "run_id": run_id,
            "step": meta["step"],
            "legs_hash": meta["legs_hash"],
            "legs_diff": {
                k: (len(v) if isinstance(v, pd.DataFrame) else v) for k, v in legs_diff.items()
            },
            "params_diff": params_diff,
        }
        writer = sink or _default_sink(log_path)  # type: ignore[arg-type]
        try:
            writer(record)
        except Exception:
            pass

    STATE["legs_prev"] = deepcopy(legs_cur)
    STATE["params_prev"] = deepcopy(params_cur)
    STATE["step"] += 1
    return legs_diff, params_diff, meta


# --------- Session helpers (run_id and journal rotation) ---------
def new_run_id(prefix: str = "sess") -> str:
    """Return a new run/session id like 'sess-YYYYMMDD-HHMMSS'."""
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def get_default_journal_dir() -> str:
    """Prefer notebooks/cache if present, else cache/ at repo root."""
    candidates = ["notebooks/cache", "cache"]
    for c in candidates:
        if Path(c).exists():
            return c
    # ensure cache exists to avoid write errors
    Path("cache").mkdir(parents=True, exist_ok=True)
    return "cache"


def get_journal_path(run_id: str, base_dir: Optional[str] = None) -> str:
    """Build a per-session JSONL path: <base_dir>/journal_<run_id>.jsonl"""
    base = base_dir or get_default_journal_dir()
    Path(base).mkdir(parents=True, exist_ok=True)
    return str(Path(base) / f"journal_{run_id}.jsonl")


