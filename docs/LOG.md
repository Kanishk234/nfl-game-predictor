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
