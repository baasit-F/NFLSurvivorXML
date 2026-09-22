"""Week-by-week recommendations, with the cost of disagreeing with them.

`optimize.plan` returns one optimal path and says nothing about how close the
runners-up were. That is the wrong shape for an actual decision. A week where
the best pick is worth 4% more season survival than the second-best is a very
different week from one where three teams are within half a percent, and you
cannot tell them apart from the plan alone.

So for every candidate team in a given week this module forces that pick, then
re-solves the assignment over every remaining week with that team spent. The
difference in whole-season survival is the true cost of the deviation — it
prices not just the extra risk this week but the knock-on effect of having
burned that team early. Hungarian runs in milliseconds, so pricing all ~30
candidates is free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import optimize as O


def forced_survival(P: pd.DataFrame, week: int, team: str,
                    used: list[str] | None = None,
                    horizon: int | None = None) -> float:
    """Season survival if you take `team` in `week` and play optimally after.

    Returns 0.0 when the forced pick leaves the remaining weeks unfillable,
    which is the correct answer: that path is dead.
    """
    used = list(used or [])
    p_now = P.at[week, team]
    if pd.isna(p_now):
        return 0.0

    later = P.loc[P.index > week]
    if later.empty:
        return float(p_now)
    try:
        rest = O.plan(later, used=used + [team], horizon=horizon)
    except ValueError:
        return 0.0
    return float(p_now) * O.survival(rest)


def week_alternatives(P: pd.DataFrame, week: int, used: list[str] | None = None,
                      horizon: int | None = None, top_n: int = 5) -> pd.DataFrame:
    """Every pickable team in `week`, priced by what taking it costs the season.

    `cost_vs_best` is the fraction of season survival given up relative to the
    optimal pick — the number to look at when you want to take a different team
    for reasons the model does not know about.
    """
    used = set(used or [])
    row = P.loc[week].dropna()
    rows = []
    for team, p in row.items():
        if team in used:
            continue
        s = forced_survival(P, week, team, list(used), horizon)
        rows.append({"team": team, "p_win": float(p), "season_survival": s})

    out = pd.DataFrame(rows).sort_values("season_survival", ascending=False,
                                         ignore_index=True)
    if out.empty:
        return out
    best = out["season_survival"].iloc[0]
    out["cost_vs_best"] = np.where(best > 0, out["season_survival"] / best - 1.0, np.nan)
    return out.head(top_n)


def season_report(P: pd.DataFrame, used: list[str] | None = None,
                  current_week: int | None = None, horizon: int | None = None,
                  top_n: int = 3) -> pd.DataFrame:
    """The full plan, each week annotated with its runners-up.

    Alternatives for week W are priced assuming the optimal picks for weeks
    before W were taken — which is what you actually face when you arrive at
    week W having followed the plan.
    """
    plan = O.plan(P, used=used, current_week=current_week, horizon=horizon)
    spent = list(used or [])
    rows = []

    for r in plan.itertuples(index=False):
        alts = week_alternatives(P, r.week, spent, horizon, top_n=top_n + 1)
        alts = alts[alts["team"] != r.team].head(top_n)
        rows.append({
            "week": r.week,
            "pick": r.team,
            "p_win": r.p_win,
            "alternatives": ", ".join(
                f"{a.team} {a.p_win:.0%} ({a.cost_vs_best:+.1%})"
                for a in alts.itertuples(index=False)),
            "margin_over_2nd": (-alts["cost_vs_best"].iloc[0]
                                if len(alts) else np.nan),
        })
        spent.append(r.team)

    out = pd.DataFrame(rows)
    out["cum_survival"] = out["p_win"].fillna(0.0).cumprod()
    return out


def flexibility(report: pd.DataFrame) -> pd.DataFrame:
    """Weeks sorted by how little the choice matters.

    A small `margin_over_2nd` means the model is close to indifferent, and that
    is where your own information — an injury the ratings have not absorbed, a
    pool-popularity read — should override the plan. A large one means the plan
    is worth following even if you dislike the pick.
    """
    return report.sort_values("margin_over_2nd")[
        ["week", "pick", "p_win", "margin_over_2nd", "alternatives"]]
