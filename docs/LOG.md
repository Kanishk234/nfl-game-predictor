# Log

Running dated log. Short entries — what was done, what broke, what's still open.

## 2026-09-08 — Phase 0 scaffold

- Created the layout from CLAUDE.md: `src/nfl_predict/{data,model,odds}`, `data/{predictions,odds,results,processed}`, `site/`, `tests/`, `.github/workflows/`, `docs/reports/`, `notebooks/`.
- `pyproject.toml` (setuptools, src layout, py>=3.12), dev extras, pytest config with the `leakage` marker, ruff at line-length 100.
- Entrypoint modules are honest stubs — they print "not implemented yet (Phase N)" and exit 1 rather than pretending to work.
- Workflows: `ci.yml` (ruff + pytest + `pytest -m leakage`), `deploy-site.yml` (Pages), and schedule-only placeholders `predict-thu.yml` / `predict-sun.yml` / `grade.yml` with the ET→UTC cron conversion documented in each file.
- Ran locally: `pytest` → 3 passed, 1 skipped; `pytest -m leakage` → exit 0; `ruff check .` → clean.

Open:
- The leakage gate currently **collects one skipped test and asserts nothing**. It exits 0 so CI isn't red, but it is not yet a real gate — Phase 1 must replace the skip with assertions over the processed dataset. Until then, a green gate means nothing.
- Local Python is 3.14; CI pins 3.12. Deps (nflreadpy/xgboost/lightgbm) were not installed locally — only pytest/ruff — so the dependency list in `pyproject.toml` is unverified against a real resolve. First Phase 1 run will confirm it.
- ~~Phase 0 exit criteria not met yet.~~ Withdrawn — Pages enabled and `deploy-site` ran; placeholder verified live at https://kanishk234.github.io/nfl-game-predictor/. Phase 0 complete, report in `docs/reports/phase0_scaffold.md`.
- Noted for Phase 6: Pages serves under `/nfl-game-predictor/`, other static hosts serve at `/`. `site_build.py` must emit relative paths so the site is host-portable.

## 2026-09-08 — Phase 1 data pipeline

- `data/games.py`: schedules 2002+ normalized to `kickoff_utc` (ET strings -> UTC via zoneinfo, DST-correct). 6,771 rows / 6,499 completed.
- `data/schedule.py`: week targeting + `assert_before_kickoff` — the gate enforced in code, not by trusting cron.
- `data/features.py`: Elo (completion-time queue so simultaneous kickoffs can't see each other), rolling EPA/margin form, franchise-code aliases.
- `data/pipeline.py`: builds the frame, runs the gate, writes `data/processed/games.parquet`. 19 features.
- Leakage gate is now real: 11 tests, synthetic ones that prove the guard *fires*, not just that real data happens to pass.
- `pytest` 33 passed, `ruff` clean, pipeline runs end to end.

Broke / fixed:
- **Cron was unsound.** Thursday pass assumed TNF opens the week; 2026 Week 1 opens **Wednesday Sept 9**. Old cron would have published Week 1 ~21h after kickoff. Renamed to `predict-early` (Tue 16:00 UTC) / `predict-late` (Sun 14:00 UTC), plus the in-code kickoff guard.
- **Relocated franchises had null EPA.** Schedules use SD/STL/OAK, team stats use LAC/LA/LV. Silent 11.5% null offensive form -> 0.31% after mapping. Regression test added.
- Gate strictness: `<=` would let 57 real games through (4:15pm ET game "completes" exactly at the 8:15pm kickoff). Using `<`.

Decided:
- **2020: keep with a `no_crowd` indicator, don't exclude.** Measured 49.6% home win / +0.06 margin vs 56.2% / +2.29 elsewhere. Excluding would break Elo's sequential chain into 2021+ — worse than the anomaly.
- **Spread is the Vegas baseline, not moneyline** — `spread_line` has zero gaps back to 2002; moneyline is absent pre-2006.

Open:
- CLAUDE.md and PLAN.md still say Thursday/Sunday and `thu`/`sun` pass tags; wording needs updating to early/late.
- Elo params untuned (Phase 2). No injury/QB features yet. JAX 2002 home games missing upstream.

## 2026-09-08 — Phase 2 model + backtest

- `model/metrics.py`, `model/baseline.py`, `model/backtest.py`, `model/train.py`. Vegas baseline fitted per fold on training rows only.
- Validation: tuning on walk-forward 2012–2019, reporting on weekly-replay 2021–2025 (110 retrains). Disjoint by construction; 2020 in neither.
- **Ships logistic regression + ridge, not a GBDT.** LR 0.6190 vs best-of-60 LightGBM 0.6249 on tuning; default GBDT was worse than Elo alone. Blends didn't help. `LGBM_SPEC` kept for reproducibility.
- **Holdout: model 64.6% / logloss 0.632 / MAE 10.07 vs Vegas 66.5% / 0.610 / 9.76.** Does not beat the market; ATS 49.0%.
- Earned its place: Elo retune (K 20→50, carry ⅔→½, 0.6301→0.6246 elo-only); QB features (+0.007 logloss, +2.0 pts acc on holdout — bigger out of sample than in).
- Rejected with numbers: 22 box-score features (best −0.0003), time-decay weights (worse), lagged-HFA feature (neutral both windows), margin→prob (worse).

Broke / fixed:
- polars `join_asof` with `by` returned a wrong-shaped column when no group matched; replaced with explicit join + strict filter in `qb_features.as_of`. Two leakage tests caught it.
- 2026 team/player stats files don't exist yet upstream (season not started); loaders skip missing seasons rather than fail.

Open:
- **Model over-calls home wins in every holdout season** (56.7% predicted vs 51.6% actual in 2021). League HFA dropped post-2019. Both principled fixes were neutral on tuning seasons; adopting them for the holdout would contaminate it. Left in, documented, to be judged live in 2026 (Phase 8).
- QB "listed starter" is the actual starter for past games — slightly optimistic vs a Tuesday projection.
- `data/models/` and `backtest.json` are gitignored and regenerable; site (Phase 6) rebuilds them.

## 2026-09-08 — Phase 2 second pass (squeeze)

- Screened on tuning seasons: SRS joint rating (6 configs), 19 situational/multi-window/interaction groups, CPOE QB rating, team form window 4-32, injury reports 2009+ (6 variants), QB window/prior/replacement grid.
- **Adopted only `QB_WINDOW` 16 -> 64, prior 200 -> 100** (tuning 0.6190 -> 0.6167, monotone plateau from 48). Holdout unchanged (0.6322). Kept per the selection rule; recorded as not transferring.
- Everything else rejected; nothing cleared -0.0005. Feature set is near saturation for public data + linear model.
- Fixed a screen that silently tested nothing (form window bound as a default arg). Injuries: 2025 rows lack `date_modified`; some polars concat schema drift across seasons.
- Holdout now: 64.4% / 0.632 / MAE 10.08 / ATS 49.4% vs Vegas 66.5% / 0.610 / 9.76. `pytest` 49 passed, ruff clean.

## 2026-09-08 — Phase 2 third pass (diagnose, then unconventional)

- Diagnosed OOF gap to Vegas: weeks 10-17 (+0.015) and QB-change games (+0.022) carry it; wk 5-9 dead even. Top disagreements = Week 17 rested starters (Manning/Brady 2009). Info gap, not model gap.
- Widened tuning window to 2006-2019 (14 folds) for this round's decisions.
- **Adopted QB draft-position prior** (`home_qb_draft`/`away_qb_draft`): -0.0007 wide, -0.0002 narrow. Holdout (once): 0.6322 -> 0.6303, MAE 10.08 -> 10.06. Transferred.
- **Rejected market distillation** (train margin on past closing lines): -0.0012 on 2006-19 but +0.0010 on 2012-19 — all gain in the data-starved early era.
- Rejected: SRS rating, team HFA, training-window ensembles, recent-intercept recal, prev-season EPA prior, week flags/interactions, diffs-only set, stakes/eliminated features, C=0.01. Elo params and QB window re-validated on wide window.
- Calibration slope 0.964 / intercept -0.02 on tuning: no global miscalibration to fix.
- Holdout now: 64.5% / 0.630 / MAE 10.06 / ATS 49.2% vs Vegas 66.5% / 0.610 / 9.76. 26 features. `pytest` 50 passed.

## 2026-09-08 — Phase 2 fourth pass (new data sources)

- Pulled `load_pbp` 2002-2025 (complete, ~70s). Screened 13 pbp-derived stats on both tuning windows.
- **Adopted pbp form (4 features, 16-game window):** no-turnover EPA/play and explosive-play rate, off+def. WIDE -0.0011, NARROW -0.0023. Holdout once: logloss 0.6303 -> 0.6306 (flat), acc 64.5 -> 65.0%.
- Rejected: success rate, competitive-only EPA/SR, TO rates, early-down EPA, CPOE, pass-rate-over-expected; snap-weighted injuries (+0.0013 even at kickoff as-of); roster continuity; season trend; referee (unusable in advance anyway); coach; TO-margin form; three Elo variants (all worse); total-variance scaling.
- **Noise floor stated in report:** SE of a logloss diff ~0.002-0.003 tuning / ~0.004 holdout. Every post-QB gain is that size; Vegas leads by 0.020. Model is at the dataset's resolution limit on public data.
- Holdout now: 65.0% / 0.631 / MAE 10.07 vs Vegas 66.5% / 0.610 / 9.76. 30 features. `pytest` 51 passed, ruff clean.
