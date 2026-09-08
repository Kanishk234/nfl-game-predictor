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
- Phase 0 exit criteria not met yet: needs a push, GitHub Pages enabled on the repo (Settings → Pages → Source: GitHub Actions), and a manual `deploy-site` run producing a live URL.
