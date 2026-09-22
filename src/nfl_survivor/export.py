"""Emit the whole projection as a single JS payload for the GitHub Pages board.

Written as `docs/data.js` assigning a global rather than as `data.json` fetched
at runtime, for one practical reason: a `fetch` of a sibling file fails under
`file://`, so a JSON build would only ever work once deployed. A plain script
tag works identically from a local checkout and from Pages, which means the
board can be opened and checked without a server.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import optimize as O
from . import predict as PR
from . import ratings as RT
from . import recommend as R
from . import ingest


def _clean(x):
    """NaN and numpy scalars are not JSON; None and floats are."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else round(float(x), 5)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def build_payload(season: int = C.CURRENT_SEASON, used: list[str] | None = None,
                  top_n: int = 4) -> dict:
    used = list(used or [])
    df = pd.read_parquet(C.DATA_PROC / "games_features.parquet")
    fitted, week, rest = PR.project(season, df)

    P_raw = O.probability_matrix(rest)
    P = O.shrink_future(P_raw, week)
    plan = O.plan(P, used=used, current_week=week)
    opp = O.opponent_matrix(rest)

    # Every remaining game, both sides, with the line where one exists.
    games = []
    for r in rest.itertuples(index=False):
        games.append({
            "week": int(r.week),
            "away": r.away_team, "home": r.home_team,
            "p_home": _clean(r.p_home), "p_away": _clean(1.0 - r.p_home),
            "source": r.source,
            "spread": _clean(getattr(r, "market_spread", None)),
            "total": _clean(getattr(r, "total_line", None)),
        })

    # Results so far this season, so the board can show the weeks already gone.
    done = df[(df["season"] == season) & (df[F.TARGET].notna())]
    results = [{
        "week": int(r.week), "away": r.away_team, "home": r.home_team,
        "home_score": _clean(r.home_score), "away_score": _clean(r.away_score),
        "winner": r.home_team if r.result > 0 else (r.away_team if r.result < 0 else None),
    } for r in done.itertuples(index=False)]

    # The plan, each week with its priced runners-up.
    spent = list(used)
    plan_rows = []
    for r in plan.itertuples(index=False):
        alts = R.week_alternatives(P, r.week, spent, top_n=top_n + 1)
        alts = alts[alts["team"] != r.team].head(top_n)
        plan_rows.append({
            "week": int(r.week), "team": r.team,
            "opponent": opp.at[r.week, r.team] if r.team in opp.columns else None,
            "p_win": _clean(r.p_win),
            "source": str(rest.loc[(rest["week"] == r.week) &
                                   ((rest["home_team"] == r.team) |
                                    (rest["away_team"] == r.team)), "source"].iloc[0]),
            "alternatives": [{"team": a.team, "p_win": _clean(a.p_win),
                              "cost": _clean(a.cost_vs_best)}
                             for a in alts.itertuples(index=False)],
        })
        spent.append(r.team)
    cum = np.cumprod([p["p_win"] or 0.0 for p in plan_rows])
    for row, c in zip(plan_rows, cum):
        row["cum_survival"] = round(float(c), 6)

    elo = RT.final_ratings(ingest.load_games())
    summary = pd.read_csv(C.OUTPUTS / "backtest_summary.csv")

    return {
        "generated": dt.date.today().isoformat(),
        "season": season,
        "current_week": int(week),
        "used": used,
        "market_sd": round(fitted.market.sd, 2),
        "teams": sorted(P_raw.columns.tolist()),
        "weeks": [int(w) for w in P_raw.index.tolist()],
        "elo": {k: round(v) for k, v in sorted(elo.items(), key=lambda kv: -kv[1])},
        "games": games,
        "results": results,
        "plan": plan_rows,
        "greedy": [{"week": int(r.week), "team": r.team, "p_win": _clean(r.p_win)}
                   for r in O.greedy(P, used=used, current_week=week).itertuples(index=False)],
        "survival": {"optimal": round(O.survival(plan), 6),
                     "greedy": round(O.survival(O.greedy(P, used=used,
                                                         current_week=week)), 6)},
        "backtest": [{k: _clean(v) for k, v in row.items()}
                     for row in summary.to_dict("records")],
    }


def main() -> None:
    payload = build_payload()
    docs = C.ROOT / "docs"
    docs.mkdir(exist_ok=True)
    out = docs / "data.js"
    out.write_text("window.SURVIVOR_DATA = "
                   + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"{len(payload['games'])} games, {len(payload['plan'])} planned weeks "
          f"-> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
