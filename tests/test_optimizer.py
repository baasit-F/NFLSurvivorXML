"""The assignment solver has to be provably right, not plausibly right."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.nfl_survivor import optimize as O


def toy() -> pd.DataFrame:
    """Three weeks, four teams. Constructed so greedy is knowably wrong.

    Team A is the best pick in all three weeks, but B and C are only good in
    week 1. Greedy burns A in week 1 and is left with a 0.50 in week 3;
    the optimiser saves A for week 3 where nothing else is available.
    """
    return pd.DataFrame(
        [[0.90, 0.85, 0.80, 0.50],
         [0.88, 0.50, 0.50, 0.50],
         [0.86, 0.50, 0.50, 0.50]],
        index=[1, 2, 3], columns=["A", "B", "C", "D"])


def test_one_pick_per_week_and_no_repeats():
    plan = O.plan(toy())
    assert list(plan["week"]) == [1, 2, 3]
    assert plan["team"].nunique() == 3


def test_optimal_beats_greedy_on_the_constructed_case():
    P = toy()
    opt, grd = O.plan(P), O.greedy(P)
    assert O.survival(opt) > O.survival(grd)
    assert opt.loc[opt["week"] == 1, "team"].iloc[0] != "A"


def test_matches_brute_force():
    """On a small grid the solver must match exhaustive search exactly."""
    from itertools import permutations
    rng = np.random.default_rng(0)
    P = pd.DataFrame(rng.uniform(0.35, 0.85, (4, 6)),
                     index=range(1, 5), columns=list("ABCDEF"))
    best = max(np.prod([P.iloc[w, t] for w, t in enumerate(combo)])
               for combo in permutations(range(6), 4))
    assert O.survival(O.plan(P)) == pytest.approx(best, rel=1e-9)


def test_byes_are_never_picked():
    P = toy()
    P.loc[2, "A"] = np.nan  # A is on bye in week 2
    plan = O.plan(P)
    assert plan.loc[plan["week"] == 2, "team"].iloc[0] != "A"


def test_used_teams_are_excluded():
    plan = O.plan(toy(), used=["A"])
    assert "A" not in set(plan["team"].dropna())
    assert set(plan["team"]) == {"B", "C", "D"}  # forced to spend the rest


def test_spending_teams_can_make_the_plan_infeasible():
    """Two of four teams gone with three weeks left is a dead entry."""
    with pytest.raises(ValueError, match="already dead"):
        O.plan(toy(), used=["A", "B"])


def test_raises_when_more_weeks_than_teams():
    P = pd.DataFrame(np.full((5, 3), 0.6), index=range(1, 6), columns=list("ABC"))
    with pytest.raises(ValueError, match="already dead"):
        O.plan(P)


def test_shrink_pulls_distant_weeks_toward_a_coin_flip():
    P = pd.DataFrame([[0.90]] * 5, index=range(1, 6), columns=["A"])
    S = O.shrink_future(P, current_week=1)
    edges = (S["A"] - 0.5).to_numpy()
    assert edges[0] == pytest.approx(0.40)         # no lead time, untouched
    assert np.all(np.diff(edges) < 0)              # monotonically shrinking
    assert np.all(edges > 0)                       # never flips a favourite


def test_survival_is_the_product_along_the_path():
    plan = pd.DataFrame({"week": [1, 2], "team": ["A", "B"], "p_win": [0.8, 0.5]})
    assert O.survival(plan) == pytest.approx(0.40)
