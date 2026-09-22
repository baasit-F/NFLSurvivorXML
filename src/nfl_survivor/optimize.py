"""Turn win probabilities into a season-long pick plan.

The naive strategy — take the highest win probability available each week — is
provably suboptimal, and not by a little. Teams are a consumable resource. Burn
Kansas City on a 78% week 3 game and you have nothing left for the week 14
slate where nobody is favoured by more than a touchdown.

The right framing is an assignment problem. Survival to the end of the season
is the product of the weekly win probabilities along your path, so

    maximise  prod_w p(team_w, w)      ==      maximise  sum_w log p(team_w, w)

subject to each team being used at most once. Taking logs turns a product into
a sum, which turns the whole thing into linear assignment on a weeks x teams
cost matrix — solvable exactly, in milliseconds, by the Hungarian algorithm.
No search, no heuristics, no regret.

What this module deliberately does *not* do is play the pool. Maximising your
own survival is the right objective only if prize money is split among
survivors. `pool.py` handles the game-theoretic layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import config as C

# Cost assigned to a (team, week) pair the team cannot be picked in — a bye, a
# team already spent, or a week already played. Large enough never to be chosen
# when any legal alternative exists, finite so the solver cannot go singular.
BLOCKED = 1e6


def probability_matrix(preds: pd.DataFrame) -> pd.DataFrame:
    """Long per-game predictions -> a weeks x teams grid of win probabilities.

    `preds` needs columns: week, home_team, away_team, p_home. Each game
    contributes two cells, one per side. Cells with no game (bye) stay NaN.
    """
    rows = []
    for r in preds.itertuples(index=False):
        rows.append({"week": r.week, "team": r.home_team,
                     "opponent": r.away_team, "p": r.p_home})
        rows.append({"week": r.week, "team": r.away_team,
                     "opponent": r.home_team, "p": 1.0 - r.p_home})
    long = pd.DataFrame(rows)
    return long.pivot(index="week", columns="team", values="p").sort_index()


def opponent_matrix(preds: pd.DataFrame) -> pd.DataFrame:
    """weeks x teams grid of the opponent each team faces, for display."""
    rows = []
    for r in preds.itertuples(index=False):
        rows.append({"week": r.week, "team": r.home_team, "v": f"vs {r.away_team}"})
        rows.append({"week": r.week, "team": r.away_team, "v": f"at {r.home_team}"})
    return pd.DataFrame(rows).pivot(index="week", columns="team", values="v").sort_index()


def cell_matrix(preds: pd.DataFrame, column: str) -> pd.DataFrame:
    """weeks x teams grid carrying any per-game column onto both sides."""
    rows = []
    for r in preds.itertuples(index=False):
        v = getattr(r, column)
        rows.append({"week": r.week, "team": r.home_team, "v": v})
        rows.append({"week": r.week, "team": r.away_team, "v": v})
    return pd.DataFrame(rows).pivot(index="week", columns="team", values="v").sort_index()


def shrink_future(P: pd.DataFrame, current_week: int, per_week: float = 0.03,
                  floor: float = 0.55) -> pd.DataFrame:
    """Pull distant weeks toward a coin flip.

    The model is calibrated against games about to kick off, where the injury
    report is known and the starting quarterback is not a guess. A week-15
    probability computed in week 2 is genuinely less certain than the model
    thinks, and nothing in the training data can teach it that, because every
    historical game was predicted from the doorstep.

    So it is applied here instead: each week of lead time shrinks the edge over
    0.5 by `per_week`, bottoming out at `floor` of the original edge. Without
    this the optimiser hoards its best teams for phantom late-season certainty.
    """
    weeks = P.index.to_numpy()
    lead = np.clip(weeks - current_week, 0, None)
    factor = np.maximum(1.0 - per_week * lead, floor)[:, None]
    return pd.DataFrame(0.5 + (P.to_numpy() - 0.5) * factor,
                        index=P.index, columns=P.columns)


def plan(P: pd.DataFrame, used: list[str] | None = None,
         current_week: int | None = None, horizon: int | None = None) -> pd.DataFrame:
    """Optimal assignment of one team per remaining week.

    `used` are teams already spent and permanently unavailable. `horizon` caps
    how many weeks ahead to plan; the assignment still respects the constraint
    inside that window, which is what matters, since only this week's pick is
    binding and the rest is re-solved next week.
    """
    used = set(used or [])
    if current_week is not None:
        P = P.loc[P.index >= current_week]
    if horizon is not None:
        P = P.iloc[:horizon]

    teams = [t for t in P.columns if t not in used]
    P = P[teams]
    if P.empty or not teams:
        return pd.DataFrame(columns=["week", "team", "p_win"])

    # -log p, with byes and impossible cells blocked.
    with np.errstate(divide="ignore"):
        cost = -np.log(np.clip(P.to_numpy(dtype=float), 1e-9, 1.0))
    cost[~np.isfinite(cost)] = BLOCKED
    cost[np.isnan(P.to_numpy(dtype=float))] = BLOCKED

    if cost.shape[1] < cost.shape[0]:
        raise ValueError(f"{cost.shape[0]} weeks to fill but only "
                         f"{cost.shape[1]} teams left — pool is already dead")

    week_idx, team_idx = linear_sum_assignment(cost)
    out = pd.DataFrame({
        "week": P.index.to_numpy()[week_idx],
        "team": np.asarray(teams)[team_idx],
        "p_win": P.to_numpy(dtype=float)[week_idx, team_idx],
    }).sort_values("week", ignore_index=True)
    # A blocked cell means the week had no legal pick left.
    out.loc[out["p_win"].isna(), "team"] = None
    return out


def greedy(P: pd.DataFrame, used: list[str] | None = None,
           current_week: int | None = None) -> pd.DataFrame:
    """The strategy everyone actually plays: best available team, every week.

    Here to be beaten. Reported alongside `plan` so the gap is visible.
    """
    used = set(used or [])
    if current_week is not None:
        P = P.loc[P.index >= current_week]
    picks = []
    for week, row in P.iterrows():
        avail = row.drop(labels=[t for t in used if t in row.index], errors="ignore").dropna()
        if avail.empty:
            picks.append({"week": week, "team": None, "p_win": np.nan})
            continue
        team = avail.idxmax()
        used.add(team)
        picks.append({"week": week, "team": team, "p_win": float(avail.max())})
    return pd.DataFrame(picks)


def survival(path: pd.DataFrame) -> float:
    """P(surviving every week in the plan) — the product along the path."""
    p = path["p_win"].dropna().to_numpy(dtype=float)
    return float(np.prod(p)) if len(p) else 0.0


def survival_curve(path: pd.DataFrame) -> pd.DataFrame:
    """Cumulative survival week by week — where a plan is likely to die."""
    out = path.copy()
    out["cum_survival"] = out["p_win"].fillna(0.0).cumprod()
    return out
