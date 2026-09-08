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
