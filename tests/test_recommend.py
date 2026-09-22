"""Recommendations must price deviations correctly, and never lie about ties."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.nfl_survivor import optimize as O
from src.nfl_survivor import predict as PR
from src.nfl_survivor import recommend as R
from src.nfl_survivor import features as F


def toy() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.90, 0.85, 0.80, 0.50],
         [0.88, 0.50, 0.50, 0.50],
         [0.86, 0.50, 0.50, 0.50]],
        index=[1, 2, 3], columns=["A", "B", "C", "D"])


def test_best_alternative_equals_the_optimal_plan():
    """Forcing the optimal week-1 pick must reproduce the optimal survival."""
    P = toy()
    best = O.plan(P)
    forced = R.forced_survival(P, 1, best.loc[best["week"] == 1, "team"].iloc[0])
    assert forced == pytest.approx(O.survival(best), rel=1e-9)


def test_alternatives_are_ranked_and_costed():
    alts = R.week_alternatives(toy(), 1, top_n=4)
    assert list(alts["season_survival"]) == sorted(alts["season_survival"], reverse=True)
    assert alts["cost_vs_best"].iloc[0] == pytest.approx(0.0)
    assert (alts["cost_vs_best"].iloc[1:] < 0).all()


def test_forcing_a_bye_team_is_dead():
    P = toy()
    P.loc[1, "A"] = np.nan
    assert R.forced_survival(P, 1, "A") == 0.0


def test_forcing_a_pick_that_strands_later_weeks_is_dead():
    """Spending the only team that can cover a later week kills the path."""
    P = pd.DataFrame([[0.9, 0.6], [0.8, np.nan]], index=[1, 2], columns=["A", "B"])
    assert R.forced_survival(P, 1, "A") == 0.0   # B alone cannot fill week 2
    assert R.forced_survival(P, 1, "B") > 0.0


def test_season_report_never_repeats_a_team():
    rep = R.season_report(toy())
    assert rep["pick"].nunique() == len(rep)


def test_current_week_skips_a_week_already_under_way():
    """A week with a Monday-nighter left is already decided, not open."""
    df = pd.DataFrame({
        "season": [2026] * 6,
        "week": [1, 1, 2, 2, 3, 3],
        F.TARGET: [1.0, 0.0, 1.0, np.nan, np.nan, np.nan],
    })
    assert PR.current_week(df, 2026) == 3


def test_current_week_is_one_before_kickoff():
    df = pd.DataFrame({"season": [2026] * 3, "week": [1, 2, 3],
                       F.TARGET: [np.nan] * 3})
    assert PR.current_week(df, 2026) == 1
