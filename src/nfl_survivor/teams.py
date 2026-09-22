"""Canonical team codes.

nflverse sources disagree on abbreviations: box scores use present-day codes
(LV, LAC, LA) while the schedule file keeps the code the franchise used at the
time (OAK, SD, STL), and older roster files carry a third set (ARZ, HST, BLT).
Everything is normalised to the present-day code so joins line up across eras.
"""
from __future__ import annotations

import pandas as pd

ALIASES = {
    "OAK": "LV", "SD": "LAC", "STL": "LA", "SL": "LA", "LAR": "LA",
    "ARZ": "ARI", "HST": "HOU", "BLT": "BAL", "CLV": "CLE",
    "WSH": "WAS", "JAC": "JAX", "SDG": "LAC", "RAI": "LV",
}


def normalize(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().replace(ALIASES)


def normalize_columns(df: pd.DataFrame, columns) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = normalize(df[col])
    return df
