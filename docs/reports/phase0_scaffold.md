# Phase 0 — Scaffold

**Status:** complete. **Date:** 2026-09-08.

## What was built

- **Layout** per CLAUDE.md: `src/nfl_predict/{data,model,odds}`, `data/{predictions,odds,results,processed}`, `site/`, `tests/`, `.github/workflows/`, `docs/reports/`, `notebooks/`.
- **`pyproject.toml`** — setuptools src-layout, `requires-python >=3.12`, runtime deps (nflreadpy, polars, pandas, numpy, scikit-learn, xgboost, lightgbm, requests), `[dev]` extras, pytest config declaring the `leakage` marker, ruff at line-length 100.
- **Entrypoints** — every command in CLAUDE.md's Commands section exists as a module and runs. Each prints `not implemented yet (Phase N) - see docs/PLAN.md` and exits 1. Stubs that fail loudly, not stubs that look plausible.
- **Workflows** — `ci.yml` (ruff + `pytest` + `pytest -m leakage`), `deploy-site.yml` (Pages, push + `workflow_dispatch`), and schedule-only placeholders `predict-thu.yml`, `predict-sun.yml`, `grade.yml`.

## Cron conversion (documented in each workflow file)

GitHub Actions cron is always UTC with no DST awareness, so one fixed time must be safe across
both EDT (UTC-4, Sep–early Nov) and EST (UTC-5, Nov–Jan).

| Job | cron (UTC) | EDT | EST | must precede |
|---|---|---|---|---|
| predict-thu | `0 21 * * 4` | 5:00 PM | 4:00 PM | TNF 8:15 PM ET |
| predict-sun | `0 14 * * 0` | 10:00 AM | 9:00 AM | early slate 1:00 PM ET |
| grade | `0 12 * * 2` | 8:00 AM | 7:00 AM | (runs after MNF) |

The 3h+ slack on the predict jobs is deliberate: scheduled runs on free runners get delayed
under load, and a delayed run breaks the gate, not merely the schedule.

## Numbers

- `pytest` → 3 passed, 1 skipped
- `pytest -m leakage` → exit 0 (1 skipped, 3 deselected)
- `ruff check .` → clean
- Pages deploy → live and serving the placeholder at https://kanishk234.github.io/nfl-game-predictor/ (verified by fetching the page, not by assuming the workflow's green check meant it rendered)

## Deferred / open

- **The leakage gate asserts nothing yet.** It collects a single *skipped* test. This keeps the
  plumbing wired (an empty marker selection exits 5 and reds out CI) but a green gate currently
  verifies nothing. **Phase 1 must replace the skip with real assertions over the processed
  dataset.** Until then, treat the green check as meaningless.
- **Dependency list is unverified.** Local Python is 3.14 and only pytest/ruff were installed;
  CI pins 3.12 but has not yet run a full resolve. Phase 1's first CI run confirms it.
- **Relative paths in the site (for Phase 6).** Pages serves this under
  `/nfl-game-predictor/`; a future move to another static host would serve it at `/`.
  `site_build.py` must emit relative asset/data paths (`./data/...`) so both work unchanged.
- Phase 1 decisions still open per PLAN.md: 2020 COVID handling (exclude / downweight / flag),
  and the odds-API and retrain-cadence questions listed at the bottom of the plan.
