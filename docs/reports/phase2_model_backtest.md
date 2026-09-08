# Phase 2 — Model + backtest

**Status:** complete. **Date:** 2026-09-08.

## Headline

On the five held-out seasons 2021–2025, replayed week by week exactly as production will run
(110 retrains, 1,424 games, each fit using only games that had finished before that week's first
kickoff):

| | accuracy | log loss | Brier | AUC | ECE | spread MAE |
|---|---|---|---|---|---|---|
| **Model** | **64.4%** | **0.632** | 0.221 | 0.695 | 0.039 | **10.08** |
| Vegas closing line | 66.5% | 0.610 | 0.212 | 0.727 | 0.026 | 9.76 |

**The model does not beat Vegas.** It trails by 2.1 points of accuracy, 0.022 of log loss and
0.31 points of spread MAE, and its against-the-spread record is 687–704–33, 49.4% (breakeven
at −110 is 52.4%). It is, however, a real model: bare Elo is at 62–63%, the untuned first attempt was 63.1%,
and the gap to the market roughly halved over this phase. Every number above comes from a
window that nothing was tuned on.

## What was built

- **`model/metrics.py`** — win-probability (accuracy, Brier, log loss, AUC, ECE, reliability
  bins) and spread (MAE, RMSE, ATS record vs the line) scoring. Fixed now so the backtest, the
  grader and the site report identical things.
- **`model/baseline.py`** — Vegas spread → win probability, a logistic fit *per fold on training
  rows only*. The baseline gets no look at the future either.
- **`model/backtest.py`** — walk-forward folds (season-level for tuning, weekly for reporting),
  `ModelSpec`, the overlap guard, and the disjoint `TUNING_SEASONS` (2012–2019) /
  `HOLDOUT_SEASONS` (2021–2025) split. 2020 sits between them: training data for the holdout,
  never a scored season.
- **`model/train.py`** — fits on everything completed, saves `data/models/latest.joblib` with
  provenance (trained-through game, timestamp, code version), and writes the weekly holdout
  replay to `data/processed/backtest.json` including calibration bins for the site.

## What ships, and why it is not a GBDT

**Logistic regression for win probability, ridge for margin, 24 features.** The plan named a
gradient-boosted model; CLAUDE.md says to optimise purely for predictive quality. Those turned
out to conflict, and the invariant wins:

| tuning seasons 2012–19, walk-forward | log loss | accuracy |
|---|---|---|
| LightGBM, sensible defaults | 0.6471 | 63.5% |
| LightGBM, best of 60 random configs | 0.6249 | 64.9% |
| Logistic regression (any C in 0.03–10) | **0.6190** | **65.8%** |
| 0.7·LR + 0.3·LightGBM blend | 0.6197 | 65.9% |
| Vegas | 0.6117 | 66.2% |

The default GBDT was *worse than Elo alone* — overfitting 5k rows. Under search it collapsed to
`num_leaves=2` (stumps: one split per tree), i.e. it was trying to become a linear model and
still losing to one. Blends did not beat the linear model alone. On these features the signal
is linear, and the boosted contender is kept in the code as `LGBM_SPEC` so the comparison stays
reproducible.

## What earned its place

**Retuned Elo.** K=20 / ⅔ carryover were conventional defaults; walk-forward search over the
tuning seasons preferred K≈50 with 50% carryover — Elo was under-reacting. This alone moved
Elo-only log loss 0.6301 → 0.6246 and the full model 0.6238 → 0.6224. K 40–60 are within 0.0001
of each other, so the middle of the plateau was taken rather than the grid edge.

**Quarterback features** (the largest single gain). Listed starters are in the schedule for
every game since 2002 and are published in advance (2026 Week 1 already carries all 32). Three
features from a per-QB shrunk rolling EPA-per-attempt, looked up by a strict as-of on the QB's
last completed game:

- `qb_rating_diff` — the starters' ratings, home minus away
- `qb_change_delta` — starter's rating minus that of whoever started the team's previous game
  (zero if unchanged). A backup stepping in is invisible at team level; this is where it shows.
  11.4% of team-games have a starter change.
- `qb_exp_diff` — log prior starts

Tuning seasons: log loss 0.6224 → 0.6190. **Holdout, confirmatory:** 0.6391 → 0.6321 and 62.6%
→ 64.6% — a larger gain out of sample than in, so this was not selection noise.

## What was tried and rejected, with numbers

- **Box-score form features** (turnovers, sacks, penalties, yards/attempt, first-down rate,
  pass rate; 22 candidates, each screened as an addition to the model on the tuning seasons):
  best delta −0.0003 log loss. None kept.
- **Time-decay sample weights** (half-lives 2–12 seasons): every setting slightly *worse* than
  uniform on the tuning seasons (best 0.6194 vs 0.6190). Not adopted — see limitations for why
  this one was tempting.
- **Lagged league home-field-advantage feature** (mean home win / margin over the previous 256–
  768 completed games, completion-time strict): neutral on tuning (±0.0001) and neutral on the
  holdout (0.6319–0.6324 vs 0.6321). A linear model cannot learn a coefficient for something
  that barely varied in its training data.
- **Margin-derived win probability** (ridge margin → normal CDF): 0.6197 vs the classifier's
  0.6190. Same information, marginally worse calibration.

## Second pass: everything else that was tried

After the first version shipped, a deliberate attempt to squeeze out whatever else public data
supports. Same rules: screened on the tuning seasons, holdout touched once at the end.

**Adopted: a longer QB rating window.** Walk-forward log loss improved monotonically with window
length — 8 games (+0.0008, worse than none), 16 (the original), 32 (−0.0012), then a plateau
from 48 games on (−0.0022 to −0.0025); shrinkage priors of 50–200 attempts were within 0.0005 of
one another. `QB_WINDOW` is now 64 with a 100-attempt prior, both mid-plateau. Tuning seasons:
0.6190 → 0.6167. **The holdout did not move** (0.6321 → 0.6322, 64.6% → 64.4%). It stays
because the selection rule is the tuning window, and reverting on holdout evidence would be the
same sin as adopting on it; the report simply records that this one did not transfer.

**Rejected, each with numbers, on the tuning seasons:**

| candidate | best Δ log loss | note |
|---|---|---|
| SRS-style joint margin rating (6 configs, completion-time strict) | −0.0002 | as an Elo *replacement*: +0.0010 |
| Bye / short-week indicators, playoff flag, primetime | −0.0002 to 0 | |
| Time-zone difference, west-coast body clock at 1pm ET | +0.0006 / +0.0014 | |
| Wind, temperature (dome-aware), freezing/windy flags, grass | +0.0003 / −0.0003 | grass at −0.0003 was the single best situational feature: noise |
| Coaching change vs previous game | 0 | |
| EPA form at 4 / 16 games, season-to-date, opponent-adjusted | −0.0003 to +0.0013 | |
| Team form window 4–32 (properly re-run; a first attempt silently tested nothing) | ±0.0005, no trend | stays at 8 |
| Interactions: Elo×week, QB×week, Elo², QB×experience | 0 to +0.0009 | |
| CPOE-based QB rating alongside EPA | +0.0003 | |
| Injury reports 2009+: Out/Doubtful counts, offensive-only, plus Questionable | −0.0005 (kickoff as-of), 0 (Tuesday as-of) | only the Sunday pass could use the kickoff version; not worth a per-pass feature set |

Nineteen situational groups, six rating configurations and six injury variants, and nothing
cleared −0.0005. That is a result: on schedule + EPA + QB data, a linear model is close to
saturated, and the remaining gap to the market is information we do not have (roster detail
beyond the QB, and the market's own aggregation of it) rather than modelling left on the table.

## Per-season, holdout

| season | model acc | model logloss | Vegas acc | Vegas logloss |
|---|---|---|---|---|
| 2021 | 60.0% | 0.652 | 62.1% | 0.628 |
| 2022 | 63.4% | 0.629 | 66.5% | 0.603 |
| 2023 | 64.6% | 0.633 | 67.4% | 0.623 |
| 2024 | 68.4% | 0.614 | 70.5% | 0.589 |
| 2025 | 65.6% | 0.633 | 66.0% | 0.606 |

## Known limitations — the honest part

**The model over-calls home wins in every holdout season.** Predicted home rate 56.7% vs actual
51.6% in 2021; 55.2% vs 53.3% in 2025. League home-field advantage fell from ~57% (2002–2018)
to ~51–55% after 2019, and a model that weights 2004 like 2024 carries the old level. This is
the largest identifiable gap to the market, and it is where most of the calibration deficit
(ECE 0.037 vs 0.026) lives. The reliability bins locate it precisely: in the coin-flip zone the
model's 0.4–0.5 bin predicts ~45% home wins and observes ~38%; its 0.5–0.6 bin predicts
~55% and observes ~51%. Above 0.6 it is calibrated to within two points, as is Vegas.

It is *not* fixed here, deliberately. Both principled remedies (decay weights, an HFA feature)
were neutral-to-negative on the tuning seasons, where HFA happened to be flat. Adopting either
because it helps 2021–2025 would be tuning on the holdout, after which the holdout number means
nothing. The live 2026 season is the only uncontaminated test left; this is the first thing
Phase 8 should evaluate prospectively.

**QB as-of is marginally optimistic.** For a completed game the "listed starter" is the actual
starter — what the market priced at kickoff — but a Tuesday pass sees a projection that can
change by Sunday. The rating lookup itself is strictly pre-kickoff; the identity is the soft
spot. Expect a small live-vs-backtest gap from this.

**Season folds vs weekly folds agree.** Season-level walk-forward on the holdout gives 0.6325 vs
0.6321 weekly, so the cheap scheme used for tuning ranks configurations the way the faithful
one does.

## Numbers

- `pytest` → 49 passed (incl. 15 leakage-gate tests); `ruff` → clean
- `python -m nfl_predict.model.train` → model saved, 110 weekly folds in ~15s
- Retrain cost is trivially within GitHub Actions free minutes.

## Deferred

- Home-field drift (above): evaluate live, not by retuning on the holdout.
- No injury/roster features beyond the QB. Roster-lock timing needs its own as-of design.
- Weather: `temp`/`wind` are null for ~45% of games (domes and gaps); not screened.
- `GAME_DURATION = 4h` remains an assumption shared with Phase 1.
