PY := PYTHONPATH="$(CURDIR)" python3

.PHONY: all data features backtest predict pool export test clean

all: data features backtest predict export

data:      ## download nflverse source tables into data/raw
	$(PY) -m src.nfl_survivor.ingest

features:  ## build the game-level design matrix
	$(PY) -m src.nfl_survivor.features

backtest:  ## walk-forward evaluation against the closing line
	$(PY) -m src.nfl_survivor.backtest

predict:   ## project the live season and solve the pick plan
	$(PY) -m src.nfl_survivor.predict

pool:      ## Monte Carlo the pool for the current plan
	$(PY) -m src.nfl_survivor.pool

export:    ## refresh docs/data.js for the GitHub Pages board
	$(PY) -m src.nfl_survivor.export

test:
	$(PY) -m pytest tests -q

clean:
	rm -rf data/processed outputs/*.csv outputs/*.png
