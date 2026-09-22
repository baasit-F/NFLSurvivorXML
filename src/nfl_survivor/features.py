"""Build the game-level design matrix.

One row per game, from the home team's perspective; `home_win` is the target.
Modelling from one side only is deliberate — mirroring every game into two rows
doubles the data but halves the independence, and it lets a model cheat by
learning the mirror instead of the football.

Leakage rules, enforced by tests/test_no_leakage.py:
  * every form feature is shifted one game back within the team's history
  * `result`, `home_score`, `away_score`, `total` never enter the design
  * the closing line is a *pre-kickoff* number, so it is allowed
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config as C
from . import ingest, ratings

TARGET = "home_win"
ID_COLS = ["game_id", "season", "week", "gameday", "home_team", "away_team"]
# Anything derived from the final score. Never in X.
OUTCOME_COLS = ["result", "home_score", "away_score", "total", "overtime", TARGET]

ROLL_WINDOWS = (4, 8)


def spread_to_prob(spread: float | np.ndarray, sd: float = C.MARGIN_SD) -> np.ndarray:
    """P(home wins) given a home-perspective point spread.

    Final margins scatter around the spread very close to normally, so the
    normal CDF is the right link. `sd` is fitted in `calibrate_margin_sd`.
    """
    return norm.cdf(np.asarray(spread, dtype=float) / sd)


def calibrate_margin_sd(games: pd.DataFrame) -> float:
    """The residual SD of (final margin - closing spread). ~13.4 points."""
    m = games[games["result"].notna() & games["spread_line"].notna()]
    return float((m["result"] - m["spread_line"]).std())


def _devig(home_ml: pd.Series, away_ml: pd.Series) -> pd.Series:
    """Moneyline -> home win probability with the book's margin divided out."""
    def implied(ml: pd.Series) -> pd.Series:
        ml = ml.astype(float)
        return np.where(ml < 0, -ml / (-ml + 100.0), 100.0 / (ml + 100.0))
    h, a = implied(home_ml), implied(away_ml)
    total = h + a
    return pd.Series(np.where(total > 0, h / total, np.nan), index=home_ml.index)


def _team_form(team_weeks: pd.DataFrame) -> pd.DataFrame:
    """Trailing offensive and defensive EPA per team-week, strictly lagged.

    Defence is the mirror of offence: a team's defensive EPA in a game is the
    EPA its opponent generated, so it comes from a self-join rather than from
    any defensive column.
    """
    tw = team_weeks.copy()
    for col in ("passing_epa", "rushing_epa"):
        if col not in tw.columns:
            tw[col] = np.nan
    tw["off_epa"] = tw["passing_epa"].fillna(0) + tw["rushing_epa"].fillna(0)

    opp = tw[["season", "week", "team", "off_epa"]].rename(
        columns={"team": "opponent_team", "off_epa": "def_epa"})
    tw = tw.merge(opp, on=["season", "week", "opponent_team"], how="left")

    tw = tw.sort_values(["team", "season", "week"])
    out = [tw[["season", "week", "team"]]]
    for w in ROLL_WINDOWS:
        for col in ("off_epa", "def_epa"):
            # shift(1) first, then roll: the window can only ever see games
            # that finished before the one being predicted.
            rolled = (tw.groupby("team")[col]
                        .transform(lambda s: s.shift(1).rolling(w, min_periods=2).mean()))
            out.append(rolled.rename(f"{col}_r{w}"))
    # How much of the current season is in the books — tells the model when the
    # rolling means are still mostly last year's team.
    out.append(tw.groupby(["team", "season"]).cumcount().rename("games_played_season"))
    return pd.concat(out, axis=1)


def build(first: int = C.FIRST_SEASON, last: int = C.CURRENT_SEASON) -> pd.DataFrame:
    games = ingest.load_games(first, last)
    elo = ratings.run(games)
    df = games.merge(elo, on="game_id", how="left", validate="one_to_one")

    df[TARGET] = np.where(df["result"].notna(), (df["result"] > 0).astype(float), np.nan)

    # --- market ---------------------------------------------------------
    df["market_spread"] = df["spread_line"].astype(float)
    df["market_prob"] = spread_to_prob(df["market_spread"])
    df["market_ml_prob"] = _devig(df["home_moneyline"], df["away_moneyline"])
    df["has_market"] = df["market_spread"].notna().astype(int)
    df["total_line"] = df["total_line"].astype(float)

    # --- ratings --------------------------------------------------------
    df["elo_spread"] = ratings.elo_to_spread(df["elo_diff"])
    # The disagreement between the model and the book. When the market exists
    # this is the only place a tree can find an edge over it; when it doesn't,
    # it's NaN and the tree falls back to elo_spread alone.
    df["elo_minus_market"] = df["elo_spread"] - df["market_spread"]

    # --- situation ------------------------------------------------------
    df["rest_diff"] = df["home_rest"].astype(float) - df["away_rest"].astype(float)
    df["short_week_home"] = (df["home_rest"].astype(float) <= 4).astype(int)
    df["short_week_away"] = (df["away_rest"].astype(float) <= 4).astype(int)
    df["off_bye_home"] = (df["home_rest"].astype(float) >= 13).astype(int)
    df["off_bye_away"] = (df["away_rest"].astype(float) >= 13).astype(int)
    df["div_game"] = df["div_game"].fillna(0).astype(int)
    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    df["is_turf"] = (~df["surface"].fillna("grass").str.contains("grass", case=False)).astype(int)
    df["temp"] = df["temp"].astype(float)
    df["wind"] = df["wind"].astype(float)
    df["neutral_site"] = (df["location"].fillna("Home") != "Home").astype(int)

    # --- form -----------------------------------------------------------
    try:
        form = _team_form(ingest.load_team_weeks(first, last))
    except FileNotFoundError:
        form = None
    if form is not None:
        fcols = [c for c in form.columns if c not in ("season", "week", "team")]
        for side in ("home", "away"):
            f = form.rename(columns={"team": f"{side}_team",
                                     **{c: f"{side}_{c}" for c in fcols}})
            df = df.merge(f, on=["season", "week", f"{side}_team"], how="left")
        for c in fcols:
            df[f"d_{c}"] = df[f"home_{c}"] - df[f"away_{c}"]

    df["week_num"] = df["week"].astype(int)
    return df.sort_values(["season", "week", "gameday", "game_id"], ignore_index=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns that are neither identifiers nor outcomes."""
    drop = set(ID_COLS) | set(OUTCOME_COLS) | {
        # raw schedule bookkeeping and anything the outcome leaks through
        "old_game_id", "gsis", "nfl_detail_id", "pfr", "pff", "espn", "ftn",
        "away_score", "home_score", "spread_line", "home_rest", "away_rest",
        "away_moneyline", "home_moneyline", "away_spread_odds", "home_spread_odds",
        "under_odds", "over_odds", "game_type",
    }
    return sorted(c for c in df.columns
                  if c not in drop and pd.api.types.is_numeric_dtype(df[c]))


def main() -> None:
    df = build()
    path = C.DATA_PROC / "games_features.parquet"
    df.to_parquet(path, index=False)
    feats = feature_columns(df)
    played = df[TARGET].notna().sum()
    print(f"{len(df):,} games ({played:,} played), {len(feats)} features -> {path}")
    print("margin SD vs closing line:", round(calibrate_margin_sd(df), 2))


if __name__ == "__main__":
    main()
