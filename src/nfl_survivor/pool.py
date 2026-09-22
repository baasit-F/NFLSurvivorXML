"""Monte Carlo over the pool — because survival is not the payoff.

`optimize.py` maximises the probability that *you* reach week 18. That is the
right objective only if you get paid for surviving. You do not. You get paid
for outlasting other people, and the pot is split among whoever is left.

The consequence is that pick popularity matters as much as win probability. If
60% of a 100-person pool takes the same 85% favourite, that pick wins you very
little when it hits — you are splitting with sixty people — and costs you
nothing special when it misses, because sixty people go out with you. A 72%
team that 4% of the pool is on is worth more: the 28% of the time it busts you
are out, but the 72% of the time it holds you have gained ground on everybody.

That is the whole contrarian argument, and this module measures it instead of
asserting it.

The one thing it cannot do is know real pick popularity. Public survivor pick
distributions are published weekly by several sites; without them, `popularity`
models the crowd as picking roughly in proportion to win probability raised to
a power. `ALPHA` is the crowd's chalkiness and is the model's weakest joint —
it is a guess, and the sensitivity to it is worth checking before trusting any
number that comes out of here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

ALPHA = 6.0  # crowd chalkiness; higher = more concentrated on the top favourite


def popularity(P: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """Modelled share of the pool taking each team each week.

    A power law on win probability, normalised within the week. Crude but it
    reproduces the shape that matters: the field piles onto the one or two
    biggest favourites and almost nobody takes a coin flip voluntarily.
    """
    W = P.to_numpy(dtype=float).copy()
    W[np.isnan(W)] = 0.0
    W = np.where(W > 0, W ** alpha, 0.0)
    totals = W.sum(axis=1, keepdims=True)
    W = np.divide(W, totals, out=np.zeros_like(W), where=totals > 0)
    return pd.DataFrame(W, index=P.index, columns=P.columns)


def simulate(preds: pd.DataFrame, my_plan: pd.DataFrame, n_entrants: int = 100,
             n_sims: int = 2000, alpha: float = ALPHA,
             seed: int = C.RANDOM_STATE) -> dict[str, float]:
    """Play the rest of the season `n_sims` times and score your entry.

    The field picks greedily from the popularity distribution, without
    repeating teams. You follow `my_plan`. Everyone alive at the last week
    anybody was alive splits the pot evenly, which is the standard rule.

    Returns your expected share of the pot, plus the diagnostics that explain
    it: how often you survive, and how crowded the finish is when you do.
    """
    from . import optimize as O

    rng = np.random.default_rng(seed)
    P = O.probability_matrix(preds)
    weeks = P.index.to_numpy()
    teams = list(P.columns)
    tix = {t: i for i, t in enumerate(teams)}
    n_weeks, n_teams = len(weeks), len(teams)

    pop = popularity(P, alpha).to_numpy()
    playing = ~np.isnan(P.to_numpy(dtype=float))  # team has a game that week

    # Your plan as a (week -> team index) array; -1 means no pick (dead week).
    wpos_plan = {w: i for i, w in enumerate(weeks)}
    mine = np.full(n_weeks, -1, dtype=int)
    for w, t in zip(my_plan["week"], my_plan["team"]):
        if t in tix and w in wpos_plan:
            mine[wpos_plan[w]] = tix[t]

    # Games as flat arrays, so a season resolves in one vectorised draw. Drawing
    # per game rather than per team is what keeps the two sides of a game
    # consistent: exactly one of them can win.
    wpos = {w: i for i, w in enumerate(weeks)}
    g_week = np.array([wpos[r.week] for r in preds.itertuples(index=False)])
    g_home = np.array([tix[r.home_team] for r in preds.itertuples(index=False)])
    g_away = np.array([tix[r.away_team] for r in preds.itertuples(index=False)])
    g_phome = preds["p_home"].to_numpy(dtype=float)

    my_share, my_survives, field_at_end = [], 0, []

    for _ in range(n_sims):
        home_win = rng.random(len(g_phome)) < g_phome
        won = np.zeros((n_weeks, n_teams), dtype=bool)
        won[g_week, g_home] = home_win
        won[g_week, g_away] = ~home_win

        alive = np.ones(n_entrants, dtype=bool)
        used = np.zeros((n_entrants, n_teams), dtype=bool)
        me_alive = True
        last_alive_week, last_alive_mask, me_last = -1, alive.copy(), True

        for wi in range(n_weeks):
            avail = playing[wi][None, :] & ~used
            w_pop = np.where(avail, pop[wi][None, :], 0.0)
            totals = w_pop.sum(axis=1, keepdims=True)
            w_pop = np.divide(w_pop, totals, out=np.zeros_like(w_pop), where=totals > 0)

            picks = np.full(n_entrants, -1, dtype=int)
            live = np.where(alive & (totals[:, 0] > 0))[0]
            for e in live:
                picks[e] = rng.choice(n_teams, p=w_pop[e])
                used[e, picks[e]] = True

            survived = np.zeros(n_entrants, dtype=bool)
            survived[live] = won[wi, picks[live]]
            new_alive = alive & survived

            my_pick = mine[wi]
            new_me = me_alive and my_pick >= 0 and bool(won[wi, my_pick])

            if new_alive.any() or new_me:
                alive, me_alive = new_alive, new_me
                last_alive_week, last_alive_mask, me_last = wi, alive.copy(), me_alive
            else:
                break  # everybody died this week; the previous week decides

        survivors = int(last_alive_mask.sum()) + int(me_last)
        if me_last and survivors > 0:
            my_share.append(1.0 / survivors)
            my_survives += 1
        else:
            my_share.append(0.0)
        field_at_end.append(survivors)

    share = float(np.mean(my_share))
    return {
        "expected_pot_share": share,
        "fair_share": 1.0 / (n_entrants + 1),
        "edge_vs_fair": share * (n_entrants + 1) - 1.0,
        "p_reach_end": my_survives / n_sims,
        "mean_survivors": float(np.mean(field_at_end)),
    }


def compare(preds: pd.DataFrame, plans: dict[str, pd.DataFrame],
            n_entrants: int = 100, n_sims: int = 2000,
            alpha: float = ALPHA) -> pd.DataFrame:
    """Score several candidate plans against the same simulated field."""
    rows = []
    for name, plan in plans.items():
        rows.append({"plan": name, **simulate(preds, plan, n_entrants, n_sims, alpha)})
    return pd.DataFrame(rows).sort_values("expected_pot_share", ascending=False,
                                          ignore_index=True)


def main() -> None:
    import pandas as pd

    from . import optimize as O
    from . import predict as PR

    df = pd.read_parquet(C.DATA_PROC / "games_features.parquet")
    _, week, rest = PR.project(C.CURRENT_SEASON, df)
    P = O.shrink_future(O.probability_matrix(rest), week)
    plans = {"optimal": O.plan(P, current_week=week),
             "greedy": O.greedy(P, current_week=week)}

    rows = []
    for n in (25, 100, 500):
        r = compare(rest, plans, n_entrants=n, n_sims=3000)
        r.insert(0, "n_entrants", n)
        rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(C.OUTPUTS / "pool_simulation.csv", index=False)
    print(out.to_string(index=False))
    print("\nNote: survival rates are 1-12%, so 3,000 sims leaves the optimal/greedy")
    print("gap inside the noise. Treat the ordering here as undetermined.")


if __name__ == "__main__":
    main()
