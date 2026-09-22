"""Fit on everything played, project every remaining game, emit a pick plan."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import model as M
from . import optimize as O
from . import recommend as R


def current_week(df: pd.DataFrame, season: int) -> int:
    """The next week you can still pick — the first with no game played yet.

    Not the first week containing an unplayed game. A week with a Monday-night
    game left is a week whose survivor pick locked on Sunday morning; treating
    it as open would have the planner spend a team on a decision you already
    made. The first *untouched* week is the one you are actually choosing.
    """
    s = df[df["season"] == season]
    if s.empty:
        return 1
    started = s.groupby("week")[F.TARGET].apply(lambda c: c.notna().any())
    untouched = started[~started].index
    return int(untouched.min()) if len(untouched) else int(s["week"].max()) + 1


def project(season: int = C.CURRENT_SEASON, df: pd.DataFrame | None = None,
            week: int | None = None):
    if df is None:
        df = pd.read_parquet(C.DATA_PROC / "games_features.parquet")

    fitted = M.SurvivorModel().fit(df[df[F.TARGET].notna()])
    if week is None:
        week = current_week(df, season)

    rest = df[(df["season"] == season) & (df["week"] >= week)].copy()
    rest["p_home"] = fitted.predict_proba(rest)
    rest["source"] = fitted.source(rest)
    return fitted, week, rest


def pick_table(plan: pd.DataFrame, opponents: pd.DataFrame,
               source: pd.DataFrame) -> pd.DataFrame:
    out = plan.copy()
    out["opponent"] = [opponents.at[w, t] if (t and w in opponents.index
                                              and t in opponents.columns) else None
                       for w, t in zip(out["week"], out["team"])]
    out["source"] = [source.at[w, t] if (t and w in source.index
                                         and t in source.columns) else None
                     for w, t in zip(out["week"], out["team"])]
    out["cum_survival"] = out["p_win"].fillna(0.0).cumprod()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Project the season and plan picks")
    ap.add_argument("--season", type=int, default=C.CURRENT_SEASON)
    ap.add_argument("--used", default="", help="comma-separated teams already picked")
    ap.add_argument("--horizon", type=int, default=None, help="plan this many weeks ahead")
    ap.add_argument("--no-shrink", action="store_true",
                    help="skip the lead-time shrink on distant weeks")
    ap.add_argument("--week", type=int, default=None,
                    help="plan from this week instead of the detected one")
    args = ap.parse_args()
    used = [t.strip().upper() for t in args.used.split(",") if t.strip()]

    df = pd.read_parquet(C.DATA_PROC / "games_features.parquet")
    fitted, week, rest = project(args.season, df, args.week)

    print(f"season {args.season}, planning from week {week}")
    print(f"  {len(rest)} games remaining; "
          f"{(rest['source'] == 'market').sum()} priced by the market, "
          f"{(rest['source'] == 'ratings').sum()} by the ratings model")
    print(f"  market scale sd={fitted.market.sd:.2f}")
    if used:
        print(f"  teams already used: {', '.join(used)}")

    P_raw = O.probability_matrix(rest)
    P = P_raw if args.no_shrink else O.shrink_future(P_raw, week)
    opp = O.opponent_matrix(rest)
    src_mat = O.cell_matrix(rest, "source")

    optimal = pick_table(O.plan(P, used=used, current_week=week, horizon=args.horizon),
                         opp, src_mat)
    naive = pick_table(O.greedy(P, used=used, current_week=week), opp, src_mat)
    report = R.season_report(P, used=used, current_week=week, horizon=args.horizon)
    report["opponent"] = optimal["opponent"].to_numpy()
    report["source"] = optimal["source"].to_numpy()

    C.OUTPUTS.mkdir(exist_ok=True)
    optimal.to_csv(C.OUTPUTS / "pick_plan.csv", index=False)
    report.to_csv(C.OUTPUTS / "recommendations.csv", index=False)
    naive.to_csv(C.OUTPUTS / "pick_plan_greedy.csv", index=False)
    rest[["game_id", "season", "week", "home_team", "away_team",
          "p_home", "source"]].to_csv(C.OUTPUTS / "game_probabilities.csv", index=False)

    print()
    print("RECOMMENDED PICKS  (alternatives show the season-survival cost of switching)")
    show = report[["week", "pick", "opponent", "p_win", "source",
                   "cum_survival", "alternatives"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print("WEEKS WHERE THE CHOICE BARELY MATTERS (override these with your own read)")
    print(R.flexibility(report).head(5)[["week", "pick", "margin_over_2nd"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(f"  season survival, optimal: {O.survival(optimal):.4f}")
    print(f"  season survival, greedy : {O.survival(naive):.4f}")
    edge = O.survival(optimal) / max(O.survival(naive), 1e-12) - 1
    print(f"  optimiser edge          : {edge:+.1%}")
    print()
    print(f"  THIS WEEK ({week}): {optimal.iloc[0]['team']} "
          f"{optimal.iloc[0]['opponent']} at {optimal.iloc[0]['p_win']:.1%}"
          f"  (greedy would take {naive.iloc[0]['team']})")


if __name__ == "__main__":
    main()
