"""Download and cache the nflverse tables this project depends on.

Everything here is public data published by the nflverse project:
https://github.com/nflverse/nflverse-data
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import urllib.request

import pandas as pd

from . import config as C
from . import teams

SEASONAL_ASSETS = {"stats_team": "stats_team_week_{season}.parquet"}
STATIC_ASSETS = {"schedules": "games.parquet"}


def _download(tag: str, filename: str, force: bool = False) -> None:
    dest = C.DATA_RAW / filename
    if dest.exists() and not force and dest.stat().st_size > 0:
        return
    url = f"{C.NFLVERSE}/{tag}/{filename}"
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def download_all(first: int = C.FIRST_SEASON, last: int = C.CURRENT_SEASON,
                 force: bool = False) -> None:
    jobs = list(STATIC_ASSETS.items())
    for season in range(first, last + 1):
        for tag, template in SEASONAL_ASSETS.items():
            jobs.append((tag, template.format(season=season)))

    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_download, t, f, force): (t, f) for t, f in jobs}
        for fut in cf.as_completed(futures):
            tag, fn = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - surface which asset failed
                print(f"  ! {tag}/{fn}: {exc}")
    print(f"raw data ready in {C.DATA_RAW}")


def load_games(first: int = C.FIRST_SEASON, last: int = C.CURRENT_SEASON) -> pd.DataFrame:
    """Regular-season schedule, one row per game, oldest first.

    `result` is home_score - away_score and is NaN for unplayed games.
    `spread_line` is the closing line from the home team's perspective:
    positive means the home team is favoured. NaN once you get more than a
    few weeks past today, which is the whole reason ratings.py exists.
    """
    df = pd.read_parquet(C.DATA_RAW / STATIC_ASSETS["schedules"])
    df = df[(df["game_type"] == "REG") &
            (df["season"].between(first, last))].copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    df = teams.normalize_columns(df, ["home_team", "away_team"])
    df["gameday"] = pd.to_datetime(df["gameday"])
    return df.sort_values(["season", "week", "gameday", "game_id"], ignore_index=True)


def load_team_weeks(first: int = C.FIRST_SEASON, last: int = C.CURRENT_SEASON) -> pd.DataFrame:
    """Weekly team box scores — the source of the EPA form features."""
    frames = []
    for season in range(first, last + 1):
        path = C.DATA_RAW / SEASONAL_ASSETS["stats_team"].format(season=season)
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError("no cached stats_team_week files; run `make data`")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["season_type"] == "REG"].copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    return teams.normalize_columns(df, ["team", "opponent_team"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Download nflverse source data")
    ap.add_argument("--first", type=int, default=C.FIRST_SEASON)
    ap.add_argument("--last", type=int, default=C.CURRENT_SEASON)
    ap.add_argument("--force", action="store_true", help="re-download cached files")
    args = ap.parse_args()
    download_all(args.first, args.last, args.force)


if __name__ == "__main__":
    main()
