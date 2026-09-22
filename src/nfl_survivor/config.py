"""Project-wide paths and constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

for _d in (DATA_RAW, DATA_PROC, OUTPUTS):
    _d.mkdir(parents=True, exist_ok=True)

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"

# nflverse carries closing lines back to 1999, but the early years are thin and
# the league changed (no 32-team alignment until 2002, 16-game seasons until
# 2021). 2006 is the first season every model feature is populated.
FIRST_SEASON = 2006
CURRENT_SEASON = 2026

# Walk-forward backtest starts here: the ratings model needs a few seasons of
# history before its carry-over prior means anything.
FIRST_TEST_SEASON = 2012

WEEKS = 18  # regular-season weeks, 2021-present (17 before that)

# --- Elo / power ratings -------------------------------------------------
ELO_START = 1500.0
ELO_K = 20.0            # points moved per game, tuned in ratings.tune_k
ELO_REGRESS = 0.25      # fraction pulled back to the mean each off-season
ELO_HFA = 48.0          # home-field advantage in Elo points (~1.7 pts of spread)
ELO_PER_POINT = 25.0    # Elo points per point of point-spread

# Margin-to-probability: NFL final margins are ~normal around the spread with
# this standard deviation. Used to convert a projected spread into P(win).
MARGIN_SD = 13.20

RANDOM_STATE = 1123
