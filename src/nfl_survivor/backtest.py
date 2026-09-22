"""Walk-forward evaluation, one season at a time.

For each test season the model sees only seasons strictly before it. Nothing is
tuned on the test seasons; the hyperparameters in `model.new_classifier` were
chosen once, on 2006-2011, and never revisited.

Four columns are reported side by side because the honest question is not "is
the model good" but "is it better than the line, and how much do you give up in
the weeks where there is no line".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import model as M


def _elo_only(df: pd.DataFrame) -> np.ndarray:
    return df["elo_prob_home"].to_numpy(dtype=float)


def run(df: pd.DataFrame | None = None,
        first_test: int = C.FIRST_TEST_SEASON,
        last_test: int = C.CURRENT_SEASON) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None:
        df = pd.read_parquet(C.DATA_PROC / "games_features.parquet")
    played = df[df[F.TARGET].notna()].copy()

    rows, preds = [], []
    seasons = [s for s in sorted(played["season"].unique())
               if first_test <= s <= last_test]

    for season in seasons:
        train = played[played["season"] < season]
        test = played[played["season"] == season]
        if len(train) < 500 or test.empty:
            continue

        fitted = M.SurvivorModel().fit(train)
        y = test[F.TARGET].to_numpy(dtype=float)

        p_full = fitted.predict_proba(test)
        p_base = fitted.market.predict_proba(test)
        # The ratings head forced onto every game: what week 14 looks like when
        # you have to call it in week 1.
        p_rate = fitted.ratings.predict_proba(test)
        p_elo = _elo_only(test)
        # The documented negative result — boosted trees over everything.
        gbm_feats = F.feature_columns(train)
        p_gbm = (M.new_classifier()
                 .fit(train[gbm_feats].astype(float), train[F.TARGET].astype(int))
                 .predict_proba(test[gbm_feats].astype(float))[:, 1])

        for name, p in [("routed_model", p_full), ("market_baseline", p_base),
                        ("ratings_only", p_rate), ("elo_only", p_elo),
                        ("boosted_all_features", p_gbm)]:
            ok = ~np.isnan(p)
            rows.append({"season": season, "method": name, **M.score(y[ok], p[ok])})

        preds.append(pd.DataFrame({
            "season": season, "week": test["week"].to_numpy(),
            "game_id": test["game_id"].to_numpy(),
            "home_team": test["home_team"].to_numpy(),
            "away_team": test["away_team"].to_numpy(),
            "home_win": y, "p_model": p_full, "p_ratings": p_rate,
            "p_market": p_base, "p_elo": p_elo, "p_gbm": p_gbm,
        }))
        ok = ~np.isnan(p_base)
        print(f"  {season}: n={len(test):3d}  market {M.score(y[ok], p_base[ok])['brier']:.4f}"
              f"  ratings {M.score(y, p_rate)['brier']:.4f}")

    by_season = pd.DataFrame(rows)
    predictions = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return by_season, predictions


def summarise(by_season: pd.DataFrame) -> pd.DataFrame:
    """Pooled across seasons, weighted by games — not a mean of means."""
    g = by_season.groupby("method")
    out = pd.DataFrame({
        "seasons": g["season"].nunique(),
        "games": g["n"].sum(),
        "log_loss": g.apply(lambda d: np.average(d["log_loss"], weights=d["n"]),
                            include_groups=False),
        "brier": g.apply(lambda d: np.average(d["brier"], weights=d["n"]),
                         include_groups=False),
        "accuracy": g.apply(lambda d: np.average(d["accuracy"], weights=d["n"]),
                            include_groups=False),
    })
    return out.sort_values("brier").reset_index()


def main() -> None:
    print("walk-forward backtest")
    by_season, preds = run()
    summary = summarise(by_season)

    by_season.to_csv(C.OUTPUTS / "backtest_by_season.csv", index=False)
    summary.to_csv(C.OUTPUTS / "backtest_summary.csv", index=False)
    preds.to_parquet(C.DATA_PROC / "backtest_predictions.parquet", index=False)

    print()
    print(summary.to_string(index=False))
    print()
    best = preds[preds["p_model"].notna()]
    print("calibration of the routed model:")
    print(M.calibration_table(best["home_win"], best["p_model"]).to_string(index=False))


if __name__ == "__main__":
    main()
