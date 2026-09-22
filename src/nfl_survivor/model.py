"""Win-probability models, and the evidence for why they are this simple.

The design here is the result of the backtest in `backtest.py`, not of taste.
Three things were measured and all three came out against the obvious answer:

  1. Gradient boosting over every feature loses to the closing spread alone
     (Brier 0.2173 vs 0.2119, 2012-2026 walk-forward). The line is an efficient
     market and roughly 40 features of public data do not beat it.

  2. Among the market-free models — the ones that have to carry weeks 4-18,
     where no line has been posted — plain logistic regression beats every
     boosted variant tried, and beats raw Elo too. With ~5,000 games and one
     dominant axis of signal, boosting finds noise.

  3. The normal CDF's scale should be fitted by log-loss, not set to the
     residual standard deviation of (margin - spread). Final margins are
     fat-tailed and spike hard on 3 and 7, so the moment estimate (13.2) is
     the wrong scale for a probability; the fitted one (~11.5) is better out
     of sample in every split tested.

So: use the market where it exists, use a small logistic model where it does
not, and keep the boosted model in the backtest table as the documented
negative result rather than deleting the evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import features as F

MARKET_PREFIXES = ("market_", "has_market", "total_line", "elo_minus_market")

# The market-free feature set. Deliberately short: every column added past this
# made out-of-sample log-loss worse, which is what you expect when the sample
# is 5,000 games and the features are mostly correlated restatements of
# "who is better".
RATINGS_FEATURES = [
    "elo_diff",          # team strength, the axis that matters
    "rest_diff",         # short weeks and byes
    "d_off_epa_r8",      # trailing offensive form, home minus away
    "d_def_epa_r8",      # trailing defensive form
    "div_game",          # division games are tighter than strength implies
    "is_dome",
    "neutral_site",
    "week_num",
]


def market_free(cols: list[str]) -> list[str]:
    return [c for c in cols if not c.startswith(MARKET_PREFIXES)]


class MarketBaseline:
    """P(home win) from the closing spread, with the scale fitted by log-loss.

    This is both the benchmark and, in the weeks where a line exists, the
    production model. Nothing beat it.
    """

    def __init__(self, sd: float = C.MARGIN_SD):
        self.sd = sd

    def fit(self, df: pd.DataFrame) -> "MarketBaseline":
        d = df[df[F.TARGET].notna() & df["market_spread"].notna()]
        if d.empty:
            return self
        y = d[F.TARGET].to_numpy(dtype=float)
        s = d["market_spread"].to_numpy(dtype=float)

        def nll(sd: float) -> float:
            p = np.clip(norm.cdf(s / sd), 1e-9, 1 - 1e-9)
            return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

        self.sd = float(minimize_scalar(nll, bounds=(8.0, 20.0), method="bounded").x)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return norm.cdf(df["market_spread"].to_numpy(dtype=float) / self.sd)


class RatingsModel:
    """Logistic regression for games with no posted line.

    Carries weeks 4-18 of the plan, which is most of it. Worse than the market
    — Brier 0.221 against 0.212 — and that gap is simply the cost of having to
    answer in week 1 a question the books will not price until week 15.
    """

    FEATURES = RATINGS_FEATURES

    def __init__(self, C_reg: float = 1.0):
        self.C_reg = C_reg
        self.pipe = None
        self.features: list[str] = []

    def fit(self, df: pd.DataFrame) -> "RatingsModel":
        train = df[df[F.TARGET].notna()]
        self.features = [c for c in self.FEATURES if c in train.columns]
        self.pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=self.C_reg, max_iter=2000,
                               random_state=C.RANDOM_STATE),
        ).fit(train[self.features].astype(float), train[F.TARGET].astype(int))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe.predict_proba(df[self.features].astype(float))[:, 1]

    def coefficients(self) -> pd.Series:
        lr = self.pipe.named_steps["logisticregression"]
        return pd.Series(lr.coef_[0], index=self.features).sort_values(
            key=np.abs, ascending=False)


def new_classifier(**kwargs) -> HistGradientBoostingClassifier:
    """The boosted model that lost. Kept so backtest.py can keep showing it."""
    params = dict(
        loss="log_loss", max_iter=200, learning_rate=0.02, max_leaf_nodes=4,
        min_samples_leaf=40, l2_regularization=2.0, early_stopping=False,
        random_state=C.RANDOM_STATE,
    )
    params.update(kwargs)
    return HistGradientBoostingClassifier(**params)


class SurvivorModel:
    """Production model: market where a line exists, ratings where it does not."""

    def __init__(self):
        self.market = MarketBaseline()
        self.ratings = RatingsModel()

    def fit(self, df: pd.DataFrame) -> "SurvivorModel":
        train = df[df[F.TARGET].notna()]
        self.market.fit(train)
        self.ratings.fit(train)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = self.ratings.predict_proba(df)
        has_line = df["market_spread"].notna().to_numpy()
        if has_line.any():
            p = p.copy()
            p[has_line] = self.market.predict_proba(df.loc[has_line])
        return p

    def source(self, df: pd.DataFrame) -> np.ndarray:
        """Which head produced each row — surfaced in every output table."""
        return np.where(df["market_spread"].notna(), "market", "ratings")


def score(y, p) -> dict[str, float]:
    """Brier and log-loss first; accuracy is reported but never optimised.

    Survivor multiplies probabilities across 18 weeks, so being right about
    *how* sure you are matters more than being on the right side of 0.5.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "accuracy": float(((p > 0.5) == (y > 0.5)).mean()),
    }


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Predicted vs realised win rate by probability decile."""
    d = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    d["bin"] = pd.cut(d["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    g = d.groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    g["gap"] = g["actual"] - g["predicted"]
    return g.reset_index()
