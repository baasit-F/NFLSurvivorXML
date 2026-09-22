# NFLSurvivorXML

An NFL survivor-pool planner: pick one team to win each week, never reuse a
team, and stay alive to week 18.

The headline finding is a negative one, and the repo is built around it rather
than around hiding it.

> **A gradient-boosted model over ~40 features of public NFL data does not beat
> the closing point spread.** It loses on both Brier score and log-loss across
> a 15-season walk-forward backtest. The closing line is an efficient market.
>
> So the machine learning in this repo is not pointed at beating the market.
> It is pointed at the 82% of the schedule the market has not priced yet.

## The two problems

Survivor looks like a prediction problem. It is mostly not.

**1. Win probability.** Needed for every game, every week. Sportsbooks solve
this better than you can — but they only post lines about three weeks out. In
the 2026 snapshot here, 48 of 272 games have a spread. The other 224 have to
come from somewhere.

**2. Pick allocation.** This is where the game actually is. Teams are a
consumable resource: spend Kansas City on a 78% week-3 game and you have
nothing left for week 14. Maximising survival across the whole season is a
constrained assignment problem, not a weekly ranking.

## What it does

```
schedule + closing lines ──┐
                           ├─> win probability, per game, all 18 weeks
Elo ratings + EPA form ────┘            │
                                        ▼
                          Hungarian assignment over weeks x teams
                                        │
                                        ▼
                     pick plan + Monte Carlo against the pool
```

## Results

Walk-forward, 2012–2026. Each season is predicted by a model that has seen only
strictly earlier seasons. 3,679 games.

| method | log-loss | Brier | accuracy |
|---|---|---|---|
| **market baseline** (closing spread) | **0.6097** | **0.2112** | 66.4% |
| routed model (what ships) | 0.6097 | 0.2112 | 66.4% |
| boosted trees, all features | 0.6151 | 0.2132 | *66.9%* |
| **ratings only** (no market) | 0.6324 | 0.2213 | 63.7% |
| raw Elo | 0.6340 | 0.2217 | 64.4% |

Three things worth reading off that table:

**The boosted model has the best accuracy and the worst-but-one calibration.**
That is the trap. Survivor multiplies probabilities across 18 weeks, so being
right about *how* sure you are compounds and being on the right side of 0.5
does not. Accuracy is the wrong metric and it is the one that looks best.

**The routed model ties the market exactly**, because where a line exists the
routed model *is* the market. Adding the model on top, at any blend weight,
made log-loss monotonically worse. Zero was the optimum, so zero is what ships.

**The ratings head gives up ~0.010 of Brier** against the market. That is the
real, measurable price of having to answer a week-14 question in week 1.

### Two things that did help

*Fitting the probability scale instead of assuming it.* Converting a spread to
a probability via the normal CDF needs a scale. The obvious choice is the
residual SD of (final margin − spread), which is 13.2 points. Fitting it by
log-loss instead gives ~11.5, and it wins out of sample in every split tested
(train<2012: 0.6123 → 0.6098; train<2018: 0.6103 → 0.6083; train<2022:
0.6091 → 0.6060). NFL margins are fat-tailed and spike on 3 and 7, so the
moment estimate is the wrong scale for a probability.

*Logistic regression over boosting, for the market-free head.* Every boosted
variant tried lost to a plain logistic on eight features — and lost to raw Elo
too. With ~5,000 games and one dominant axis of signal, boosting finds noise.

## The optimiser

Maximising survival is
`max Π p(team_w, w)` = `max Σ log p(team_w, w)` subject to each team being used
at most once. Logs turn the product into a sum, which makes it exact linear
assignment — the Hungarian algorithm, milliseconds, no search or heuristics.

Against the strategy everyone actually plays (take the best available team each
week), on the 2026 schedule from week 2:

```
season survival, optimal: 0.0017
season survival, greedy : 0.0014
optimiser edge          : +24.0%
```

`tests/test_optimizer.py` checks the solver against exhaustive search on small
grids, so that edge is a property of the method and not of one lucky schedule.

`recommend.py` prices the runners-up: for every candidate team it forces that
pick and re-solves the rest of the season, so `cost_vs_best` captures not just
the extra risk this week but the knock-on cost of burning that team early. The
weeks where that cost is near zero are the ones where your own information —
an injury the ratings have not absorbed, a read on pool popularity — should
override the plan.

Distant weeks are shrunk toward a coin flip before solving. The model is
calibrated on games about to kick off, where the injury report is known and the
starting quarterback is not a guess; a week-15 probability computed in week 2
is genuinely less certain than the model thinks, and no amount of training data
can teach it that, because every historical game was predicted from the
doorstep.

## Playing the pool

Survival is not the payoff. You get paid for outlasting other people, and the
pot splits among whoever is left — so pick *popularity* matters as much as win
probability. `pool.py` simulates a field of entrants drawing from a chalkiness
model and scores candidate plans by expected share of the pot.

**This part does not currently work well enough to act on.** Across 3,000 sims
the optimal and greedy plans trade places depending on pool size (greedy ahead
at 25 and 500 entrants, optimal at 100). Survival rates are 1–12%, so almost
every simulation contributes nothing and the gap stays inside the noise. Two
things would fix it, both listed below.

## Known limitations

- **No real pick-popularity data.** The crowd is modelled as picking in
  proportion to win probability raised to a power. Real survivor pick
  distributions are published weekly; wiring them in is the single highest-value
  change available and would make `pool.py` mean something.
- **The pool simulation is underpowered.** Needs variance reduction (common
  random numbers across plans, importance sampling on the survival tail) or an
  order of magnitude more sims.
- **No injury or starting-QB signal.** The market prices a backup QB instantly;
  the ratings head has no idea. This is most of the remaining gap between the
  two heads.
- **Elo is team-level and slow.** It cannot see a team that just traded for a
  quarterback, and it takes weeks to catch a genuinely changed roster.
- **Ties are graded as home losses.** Check your pool's rule; some push.
- **The ratings head is weakest in weeks 4-6** (Brier 0.2345 against 0.2135 in
  weeks 11-14). Early in the season Elo has absorbed two or three games of the
  current roster and is still mostly carrying last year forward, so it
  over-reacts to small samples. Those picks deserve the least confidence, which
  is the opposite of what their stated probabilities suggest.

## Usage

```bash
pip install -r requirements.txt
make data       # download nflverse tables (~15 MB)
make features   # build the game-level design matrix
make backtest   # walk-forward evaluation against the closing line
make predict    # project the live season, solve the pick plan
make pool       # Monte Carlo the plan against a simulated field
make test
```

Mid-season, tell it which teams you have already spent:

```bash
PYTHONPATH=. python3 -m src.nfl_survivor.predict --used KC,BAL,SF --horizon 8
```

## Layout

| file | what it does |
|---|---|
| `ingest.py` | download and load the nflverse tables |
| `ratings.py` | Elo, walked forward one game at a time |
| `features.py` | the game-level design matrix, with the leakage rules |
| `model.py` | market baseline, ratings head, and the routing between them |
| `backtest.py` | walk-forward evaluation, including the models that lost |
| `optimize.py` | Hungarian assignment over the remaining schedule |
| `pool.py` | Monte Carlo against a simulated field |
| `recommend.py` | runners-up for each week, priced by what switching costs |
| `predict.py` | fit, project, and emit the pick plan |

Data sources and the leakage argument: [`docs/DATA.md`](docs/DATA.md).

## Licence

MIT. Data from [nflverse](https://github.com/nflverse/nflverse-data) under CC-BY-4.0.
