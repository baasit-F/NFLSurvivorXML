# Data

Everything comes from [nflverse](https://github.com/nflverse/nflverse-data), a
community project that publishes cleaned NFL data as versioned release assets.
Nothing is scraped, hand-entered, or bought. `make data` pulls the files below
into `data/raw/` (git-ignored, ~15 MB).

| Release tag | File | What it gives us |
|---|---|---|
| `schedules` | `games.parquet` | Every regular-season game 1999–present: final scores, **closing spread and moneyline**, rest days, roof/surface, kickoff weather, starting QB |
| `stats_team` | `stats_team_week_{season}.parquet` | Weekly team box scores; the source of the trailing offensive and defensive EPA features |

## The closing line

`spread_line` is the single most important column in this repo. It is the
closing point spread from the **home team's perspective** — positive means the
home team is favoured. nflverse carries it back to 1999 with no gaps.

Two properties matter:

**It is pre-kickoff information.** Using it is not leakage. It is a number that
existed and was public before the game started, which is exactly the standard
every other feature is held to.

**It runs out.** Sportsbooks post lines about three weeks ahead. As of the 2026
snapshot in this repo, 48 of 272 games have one. A survivor pool needs a
probability for all 18 weeks in week 1, so 82% of the schedule has to be
projected from team strength instead. That asymmetry is the reason
`ratings.py` and the two-headed model in `model.py` exist.

## Team codes

Box scores use present-day abbreviations (`LV`, `LAC`, `LA`) while
`games.parquet` keeps the code the franchise used at the time (`OAK`, `SD`,
`STL`). `teams.normalize` maps everything to the present-day code. Without it
every pre-relocation Raider, Charger and Ram silently loses its join — the kind
of bug that produces a model that looks fine and is quietly wrong for three
franchises.

## Ties

NFL ties are about 0.2% of games. Elo splits them (actual = 0.5). The win
probability models treat them as losses for the home team, which is what the
survivor rules do in most pools — check yours, since some grade a tie as a
loss for both sides and some push.

## Licence

nflverse data is released under
[CC-BY-4.0](https://github.com/nflverse/nflverse-data). This repo is MIT.
