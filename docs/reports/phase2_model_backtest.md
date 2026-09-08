# Phase 2 — Model + backtest

**Status:** complete. **Date:** 2026-09-08.

## Headline

On the five held-out seasons 2021–2025, replayed week by week exactly as production will run
(110 retrains, 1,424 games, each fit using only games that had finished before that week's first
kickoff):

| | accuracy | log loss | Brier | AUC | ECE | spread MAE |
|---|---|---|---|---|---|---|
| **Model** | **65.0%** | **0.631** | 0.221 | 0.696 | 0.043 | **10.07** |
| Vegas closing line | 66.5% | 0.610 | 0.212 | 0.727 | 0.026 | 9.76 |

**The model does not beat Vegas.** It trails by 1.5 points of accuracy, 0.021 of log loss and
0.30 points of spread MAE, and its against-the-spread record is 682–709–33, 49.0% (breakeven
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

**Logistic regression for win probability, ridge for margin, 30 features.** The plan named a
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

## Third pass: diagnose first, then the unconventional ideas

This round started from a diagnosis instead of a feature list, and validated on a wider tuning
window (2006–2019, 14 folds) because decisions at the ±0.0005 level are noisy on eight.

**Where the gap to Vegas actually lives** (tuning-season out-of-fold, gap = model log loss − Vegas):
weeks 5–9 are dead even (−0.0002); weeks 10–17 carry the bulk (+0.015); games with a QB change
are three times worse than games without (+0.022 vs +0.007). The 113 games where the two disagree
by 20+ points of win probability are dominated by **Week 17 games in which a clinched team rested
its starters** — Manning and Brady both listed as starters in 2009 Week 17 and pulled after a
series. The listed starter is the star; the market knows he will not play. That is information we
do not have before kickoff, not a modelling failure. A stakes feature built from as-of season
records (eliminated / 12+ wins) did not capture it (+0.0007), because "clinched a seed and will
rest" is not "has 12 wins".

**Adopted: a draft-position prior for quarterbacks** (`home_qb_draft`, `away_qb_draft`, 1 = first
overall, 0 = undrafted). A rookie has no rating history, so the model treated a #1 pick and a UDFA
identically until a few hundred attempts in; the market does not. −0.0007 on the wide window,
−0.0002 on the narrow one, consistent sign, sound mechanism, two features. **Holdout, touched
once: 0.6322 → 0.6303, 64.4% → 64.5%, MAE 10.08 → 10.06.** It transferred.

**The interesting rejection: market distillation.** Train the margin model on the *closing line
of past games* as a lower-noise target (α·line + (1−α)·margin), with no line used at prediction
time. On 2006–2019 it looked like the best idea of the round (−0.0012 log loss, −0.03 MAE). On
2012–2019 it *hurt* (+0.0010). The entire gain sits in 2006–2011, when the model was data-starved;
in the modern era the line adds nothing our features do not already carry. Rejected on the
evidence, which also spares the project the argument about whether a model trained on the market
is still independent of it.

**Also rejected, wide window:** SRS-style joint margin rating (best −0.0002; as an Elo
replacement +0.0010), team-specific home advantage (+0.0002), ensembles over training windows —
all / last 8 / last 4 seasons (+0.0005), intercept recalibrated on recent seasons (+0.0002 to
+0.0012), previous-season EPA as an early-season prior (+0.0002), early/late/postseason flags or
interactions in place of `week` (+0.0004 / +0.0012), a diffs-only 12-feature set (+0.0001), and
stronger regularisation (C=0.01 −0.0006 wide but flat narrow: an early-era effect again). The
Elo parameters re-validated on the wide window: K 35–70 and carryover 0.35–0.65 all within
0.0006, the chosen values at the optimum.

**The QB window re-checked on the wide window** and held: 16 games +0.0015, 32 +0.0006, 64 and 96
at 0. The earlier concern that it did not transfer to the holdout is now moot; with the draft
prior the holdout moved.

The calibration diagnostic is worth recording: on the tuning seasons the model's logits have
slope 0.964 and intercept −0.02 (perfect is 1, 0). It is not systematically over- or
under-confident; the residual error is game-specific information, not a global miscalibration
that a Platt layer would fix.

## Fourth pass: new data sources, and the noise floor

The previous three passes screened features built from the same sources. This one went after
*untapped* public data, on both tuning windows (2006–2019 and 2012–2019) simultaneously.

**Adopted: play-by-play form** — four features from `load_pbp` (complete back to 2002), each a
16-game rolling mean, offence and defence: **EPA per play excluding turnover plays** (turnover
EPA swings are huge and the recovery is close to a coin flip, so they are noise in a quality
measure) and **explosive-play rate** (share of plays gaining 20+, a stable trait that EPA
averages wash out). Together: −0.0011 on the wide window, −0.0023 on the narrow. The raw 8-game
EPA form stays; dropping it in favour of these was worse. Holdout, touched once: log loss
0.6303 → 0.6306, accuracy 64.5% → 65.0%, MAE flat. Kept by the selection rule.

**Rejected from pbp**, same windows: success rate (8 and 16), competitive-only (10–90% win
probability) EPA and success rate, turnover rates, early-down pass/rush EPA, CPOE, early-down
pass rate over expected — all between +0.0010 and −0.0004 — and replacing the raw EPA form with
any of the pbp variants.

**Rejected, other sources:**

| candidate | WIDE | NARROW | note |
|---|---|---|---|
| Snap-weighted injuries (sum of prior-game snap share of players Out, 2012+) | +0.0013 | — | worse even with kickoff-time reports; Tuesday-time: 0 |
| Offseason roster continuity (share of last season's snaps still on the week-1 roster) | — | −0.0002 (2015–19) | weeks 1–4 only: 0.6503 → 0.6497 |
| Linear season trend (lets home advantage drift) | +0.0004 | +0.0004 | |
| Referee home-win rate as-of | −0.0005 | −0.0003 | assignments are not known in advance for upcoming games; unusable regardless |
| Coach career win% / tenure | +0.0010 / +0.0005 | +0.0002 | |
| Turnover-margin form | 0 | +0.0010 | |
| Elo: higher K in weeks 1–4; EPA-driven updates; split home/away ratings | +0.0004 to +0.0048 | +0.0014 to +0.0052 | every variant worse |
| Variance scaling by predicted total | 0 | 0 | |

**The noise floor.** Four passes and roughly eighty candidates in, the pattern is now clear
enough to state as a finding. The paired standard error of a log-loss difference is about
0.002–0.003 on the tuning windows and about 0.004 on the 1,424-game holdout. Every adopted
change since the QB features has been of that size, which is why two of them (QB window, pbp
form) did not visibly move the holdout and one (draft prior) did: at this scale, transfer is a
coin flip. Vegas is 0.020 ahead. An improvement large enough to matter — 0.01 or more — would
have been unmistakable in any of these screens, and nothing came close. On public pre-kickoff
data this model is at the resolution limit of the dataset, and further screening on 2002–2019
cannot distinguish a real 0.001 gain from luck. The right next experiment is a live season.

## Per-season, holdout

| season | model acc | model logloss | Vegas acc | Vegas logloss |
|---|---|---|---|---|
| 2021 | 63.2% | 0.651 | 62.1% | 0.628 |
| 2022 | 63.7% | 0.626 | 66.5% | 0.603 |
| 2023 | 65.3% | 0.633 | 67.4% | 0.623 |
| 2024 | 68.1% | 0.615 | 70.5% | 0.589 |
| 2025 | 64.6% | 0.628 | 66.0% | 0.606 |

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
