# nfl-predict — Implementation Plan

The running schedule once it is live is in [WEEKLY_FLOW.md](WEEKLY_FLOW.md).

High-level phased plan. Each phase ends with a summary report in `docs/reports/` per CLAUDE.md.
Phases are ordered so each one produces something checkable before the next starts — no phase
should be "trust me, it'll come together in phase 6."

## Phase 0 — Scaffold
- Repo structure per CLAUDE.md layout.
- `pyproject.toml`, dev dependencies, pytest config.
- Empty GitHub Actions workflow files (no logic yet, just confirm they trigger on schedule).
- Confirm GitHub Pages is enabled on the repo and a trivial "hello" site deploys end to end.
- **Exit criteria:** a placeholder site is live at the GitHub Pages URL, deployed by a workflow
  triggered manually.

## Phase 1 — Data pipeline
- Pull historical games (nflreadpy), **2002–present** — bounded to the current 32-team/
  current-division-structure era rather than reaching further back, since older seasons reflect
  a materially different game (rule changes, roster sizes, pass-rate evolution) and add noise
  more than signal for a tabular GBDT. "Present" means refreshed on every retrain, not a fixed
  end year — always pull everything available up to the most recently completed week.
- Flag the 2020 season (COVID: empty stadiums, opt-outs, schedule disruption) as a known anomaly
  — decide during this phase whether to exclude it, downweight it, or add an indicator feature
  for it, and document the choice and why.
- Feature engineering: Elo, EPA-based team form, rest days, home/away, and whatever else earns
  its place — with an explicit "as-of" timestamp per feature.
- Leakage guard tests (`pytest -m leakage`): assert no feature's as-of time is >= the game's
  kickoff time.
- **Exit criteria:** a processed dataset + a passing leakage test suite, committed report on
  coverage (how many seasons/games, any gaps).

## Phase 2 — Baseline model + backtest
- Train the best-performing model achievable on the feature set — a GBDT (XGBoost/LightGBM) as a
  strong starting point, but not capped there if a richer feature set, ensembling multiple model
  families, or a heavier hyperparameter search meaningfully improves the backtest. Unlike the
  FPGA project, there is no resource ceiling here beyond GitHub Actions' free compute minutes for
  the weekly retrain — optimize for accuracy/calibration, not for simplicity or deployability.
- Backtest on held-out seasons; report accuracy, AUC, spread MAE, and — critically — how it
  compares to the closing Vegas line on the same held-out games (using historical odds data if
  available, or acknowledging the gap if not yet wired up).
- **Exit criteria:** a report with real backtest numbers, honest about where the model sits
  relative to a Vegas-line baseline.

## Phase 3 — Odds baseline integration
- Wire up a free odds API (The Odds API free tier) to fetch current lines for the upcoming week.
- Store frozen odds snapshots per publish pass under `data/odds/`.
- **Exit criteria:** a script that, run on demand, fetches and commits this week's lines in the
  same format predictions will use, so the two are directly comparable.

## Phase 4 — Prediction generation
- `predict.py`: loads latest trained model, generates win prob + spread for every game in the
  target week, writes immutable JSON under `data/predictions/`, tagged with pass (`thu`/`sun`),
  timestamp, and model version.
- Wire in the Phase 3 odds fetch so each prediction run also freezes the baseline at the same
  moment.
- **Exit criteria:** running this manually for a real upcoming week produces a correct,
  timestamped prediction file plus a matching odds snapshot.

## Phase 5 — Grading / results tracking
- `grade.py`: after games finish, pull final scores, compare to both our predictions and the
  frozen baseline, write results + a running accuracy/calibration history.
- Decide and document the scoring metrics up front (win accuracy, Brier score for win prob,
  spread MAE/ATS record) so the site has something consistent to display.
- **Exit criteria:** running this against a past completed week produces correct grading output
  and an accuracy-history file that a second run doesn't duplicate or corrupt.

## Phase 6 — Static site
- `site_build.py` reads `data/` and renders: this week's predictions vs. baseline, past weeks'
  results, running accuracy/calibration charts over the season.
- Plain static HTML/JS (no framework needed unless it earns its place) reading the committed
  JSON directly.
- **Exit criteria:** the site correctly renders a full round-trip of fake or past data — a
  predicted week, its baseline, and its graded result.

## Phase 7 — Automation
- GitHub Actions workflows:
  - Thursday cron (pre-TNF, correct UTC offset) → retrain (or reuse latest model, TBD) → predict
    → freeze odds → commit → rebuild + deploy site.
  - Sunday cron (pre-early-slate) → same, fresh pass.
  - Post-weekend cron (e.g. Tuesday) → grade completed games → commit → rebuild + deploy site.
- **Exit criteria:** one full live week runs unattended — Thursday and Sunday predictions
  publish on schedule, grading runs after, site reflects it all correctly — with me only
  reviewing/pushing commits Claude proposes, per the workflow rules.

## Phase 8 — Live season monitoring + polish
- Watch the first few real weeks closely: confirm no leakage slipped through, confirm timestamps
  land where expected relative to actual kickoff, confirm the site doesn't silently break on an
  edge case (bye weeks, Thursday-only weeks, postseason once regular season ends).
- Only after the regular-season loop is solid: extend to postseason (single-elimination changes
  some assumptions — no "next week" in the same sense, home field via seeding not schedule).
- **Exit criteria:** a full regular season running unattended with a clean accuracy history and
  no manual interventions needed week to week.

---

**Open questions to resolve before/during Phase 1** (not blocking scaffold, but worth deciding
early): which odds API tier actually covers a full season within free limits; how far back
historical odds data is available for backtesting purposes; how to handle bye weeks and
Thursday-only weeks in the retrain/publish schedule; whether "model version" in Phase 4 means a
literal retrain each pass or reusing Thursday's model for Sunday's predictions (only refreshing
injury/lineup-adjacent features) — cheaper and likely just as accurate, but worth deciding
deliberately rather than defaulting into it.