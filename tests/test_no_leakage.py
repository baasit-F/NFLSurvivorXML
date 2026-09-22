"""The tests that matter: nothing in X may know how the game ended."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.nfl_survivor import features as F
from src.nfl_survivor import model as M
from src.nfl_survivor import ratings, config as C


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    path = C.DATA_PROC / "games_features.parquet"
    if not path.exists():
        pytest.skip("run `make data && make features` first")
    return pd.read_parquet(path)


def test_no_outcome_columns_in_design(df):
    """Score-derived columns must never reach a model."""
    banned = {"result", "home_score", "away_score", "total", "overtime", F.TARGET}
    assert not banned & set(F.feature_columns(df))


def test_no_feature_perfectly_predicts_the_target(df):
    """A |correlation| near 1 with the outcome means something leaked."""
    played = df[df[F.TARGET].notna()]
    y = played[F.TARGET].astype(float)
    for col in F.feature_columns(played):
        x = played[col].astype(float)
        if x.notna().sum() < 100 or x.nunique() < 2:
            continue
        r = abs(np.corrcoef(x[x.notna() & y.notna()], y[x.notna() & y.notna()])[0, 1])
        assert r < 0.9, f"{col} correlates {r:.3f} with the outcome"


def test_elo_ratings_are_pre_game(df):
    """Elo for a game must come only from games played before it.

    Truncating the history at week W and re-rating must reproduce exactly the
    ratings the full run assigned to week-W games. If ratings were leaking
    future results, the two would differ.
    """
    from src.nfl_survivor import ingest
    games = ingest.load_games(2015, 2019)
    full = ratings.run(games)
    cut = games[(games["season"] < 2019) | (games["week"] <= 5)]
    partial = ratings.run(cut)

    merged = full.merge(partial, on="game_id", suffixes=("_full", "_part"))
    assert len(merged) == len(cut)
    np.testing.assert_allclose(merged["elo_diff_full"], merged["elo_diff_part"],
                               atol=1e-9)


def test_form_features_exclude_the_current_game(df):
    """Rolling EPA is shifted, so week 1 of a season cannot have a full window."""
    wk1 = df[(df["week"] == 1) & (df["season"] == df["season"].max())]
    if wk1.empty:
        pytest.skip("no week-1 rows")
    assert wk1["home_games_played_season"].fillna(0).max() == 0


def test_ratings_model_uses_no_market_features(df):
    """The head that covers unposted weeks must not touch a line."""
    assert not set(M.RATINGS_FEATURES) & {
        c for c in df.columns if c.startswith(M.MARKET_PREFIXES)}


def test_market_baseline_beats_chance(df):
    played = df[df[F.TARGET].notna() & df["market_spread"].notna()]
    base = M.MarketBaseline().fit(played)
    s = M.score(played[F.TARGET], base.predict_proba(played))
    assert s["accuracy"] > 0.60
    assert s["brier"] < 0.25  # 0.25 is a constant 0.5 forecast
