"""Elo power ratings, walked forward one game at a time.

Why this module exists: the closing spread is the single best predictor of an
NFL game, but sportsbooks only post lines two or three weeks ahead. A survivor
pool needs a win probability for every one of the 18 weeks on day one, so the
back half of the season has to be projected from team strength alone. Elo is
that projection — cheap, transparent, and good enough that the blend with the
market (see `features.py`) barely loses anything in the weeks where both exist.

Every rating this module emits is a *pre-game* rating: it is computed from
games that had already finished when the game in question kicked off. That
property is what `tests/test_no_leakage.py` checks, and it is the difference
between a backtest and a fantasy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _expected(elo_diff: float) -> float:
    """Elo's logistic: probability the team with `elo_diff` in its favour wins."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def _mov_multiplier(margin: float, elo_diff_winner: float) -> float:
    """FiveThirtyEight's margin-of-victory scaler.

    Two jobs. The log dampens blowouts so a 45-point win does not move a rating
    three times as far as a 15-point one. The denominator is autocorrelation
    control: good teams run up the score on bad teams, so without it the strong
    keep inflating and ratings diverge.
    """
    return np.log(abs(margin) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))


def run(games: pd.DataFrame, k: float = C.ELO_K, hfa: float = C.ELO_HFA,
        regress: float = C.ELO_REGRESS) -> pd.DataFrame:
    """Walk every game in order, returning pre-game ratings for each.

    Unplayed games (`result` is NaN) are rated but never update anything, so
    calling this on a schedule that runs past today is safe: the ratings simply
    stop moving once the played games run out.
    """
    games = games.sort_values(["season", "week", "gameday", "game_id"])
    elo: dict[str, float] = {}
    last_season: int | None = None
    rows = []

    for g in games.itertuples(index=False):
        if last_season is not None and g.season != last_season:
            # Off-season: pull everyone part-way back to average. Roster churn,
            # the draft, and schedule regression all make last year's rating a
            # biased forecast of this year's team.
            for t in elo:
                elo[t] = C.ELO_START + (elo[t] - C.ELO_START) * (1.0 - regress)
        last_season = g.season

        home = elo.setdefault(g.home_team, C.ELO_START)
        away = elo.setdefault(g.away_team, C.ELO_START)
        diff = (home + hfa) - away

        rows.append({
            "game_id": g.game_id,
            "elo_home_pre": home,
            "elo_away_pre": away,
            "elo_diff": diff,
            "elo_prob_home": _expected(diff),
        })

        if pd.isna(g.result) or g.result is None:
            continue  # future game: rate it, but learn nothing from it

        margin = float(g.result)
        if margin > 0:
            actual, diff_winner = 1.0, diff
        elif margin < 0:
            actual, diff_winner = 0.0, -diff
        else:
            actual, diff_winner = 0.5, 0.0  # ties: ~0.2% of games, split them

        mult = _mov_multiplier(margin, diff_winner) if margin != 0 else 1.0
        shift = k * mult * (actual - _expected(diff))
        elo[g.home_team] = home + shift
        elo[g.away_team] = away - shift

    return pd.DataFrame(rows)


def final_ratings(games: pd.DataFrame, k: float = C.ELO_K, hfa: float = C.ELO_HFA,
                  regress: float = C.ELO_REGRESS) -> dict[str, float]:
    """Ratings after the last *played* game — the starting point for forecasts."""
    played = games[games["result"].notna()]
    if played.empty:
        return {}
    elo: dict[str, float] = {}
    last_season: int | None = None
    for g in played.sort_values(["season", "week", "gameday", "game_id"]).itertuples(index=False):
        if last_season is not None and g.season != last_season:
            for t in elo:
                elo[t] = C.ELO_START + (elo[t] - C.ELO_START) * (1.0 - regress)
        last_season = g.season
        home = elo.setdefault(g.home_team, C.ELO_START)
        away = elo.setdefault(g.away_team, C.ELO_START)
        diff = (home + hfa) - away
        margin = float(g.result)
        if margin > 0:
            actual, dw = 1.0, diff
        elif margin < 0:
            actual, dw = 0.0, -diff
        else:
            actual, dw = 0.5, 0.0
        mult = _mov_multiplier(margin, dw) if margin != 0 else 1.0
        shift = k * mult * (actual - _expected(diff))
        elo[g.home_team] = home + shift
        elo[g.away_team] = away - shift
    return elo


def elo_to_spread(elo_diff: float | np.ndarray) -> float | np.ndarray:
    """Elo points -> points of point-spread, home perspective."""
    return np.asarray(elo_diff, dtype=float) / C.ELO_PER_POINT


def tune_k(games: pd.DataFrame, grid=(10, 14, 18, 20, 22, 26, 30),
           first_test: int = C.FIRST_TEST_SEASON) -> pd.DataFrame:
    """Pick K by out-of-sample log-loss rather than by folklore.

    Scored only on seasons >= first_test so the early ratings have burnt in.
    """
    played = games[games["result"].notna()].copy()
    out = []
    for k in grid:
        r = run(played, k=k)
        m = played.merge(r, on="game_id")
        m = m[m["season"] >= first_test]
        y = (m["result"] > 0).astype(float)
        p = m["elo_prob_home"].clip(1e-6, 1 - 1e-6)
        out.append({
            "k": k,
            "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
            "brier": float(((p - y) ** 2).mean()),
            "accuracy": float(((p > 0.5) == (y > 0.5)).mean()),
        })
    return pd.DataFrame(out).sort_values("log_loss", ignore_index=True)
