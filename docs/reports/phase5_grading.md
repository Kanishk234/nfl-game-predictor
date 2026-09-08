# Phase 5 — Grading / results tracking

**Status:** complete (code); first real grading lands after Week 1 finishes. **Date:** 2026-09-08.

## What was built

**`grade.py`** — `python -m nfl_predict.grade` reads every published prediction file, pulls
final scores from nflreadpy, and writes:

- `data/results/<season>_<week>.json` — per-game grades for the model and the frozen Vegas
  baseline, per-pass summaries, and a week summary. Rewritten as more of the week's games
  finish; `complete` flags when all have.
- `data/results/history.json` — season-to-date record with reliability bins for the site's
  calibration chart. **Rebuilt from the per-week files on every run, never appended to**, so a
  second run is idempotent by construction (tested: two runs produce identical output, one week
  entry, no duplicated games).

Predictions are read-only to this module. Nothing here can alter a published prediction.

## Which prediction counts

A game can appear in both passes. Its **official** prediction is the latest pass generated
before that game's kickoff — the early pass for Wednesday/Thursday games, the late pass for the
Sunday/Monday slate. A pass generated at or after a game's kickoff is ignored for that game
(tested). Every pass is also graded on its own under `by_pass`, so the early-vs-late question
("does the fresher pass actually help?") is answered by the data rather than assumed.

## Metrics, fixed up front (per PLAN.md)

Per game: correct pick, Brier, absolute margin error; against the frozen line, an ATS result of
win / loss / push (push when the actual margin lands exactly on the line). Aggregated: accuracy,
Brier, log loss, AUC, ECE, spread MAE for both model and Vegas; ATS record with the −110
breakeven (52.4%) recorded beside it so the number is read honestly. Vegas win probability is
the de-vigged moneyline, falling back to the spread-derived probability if a moneyline is
missing.

**Tie rule.** A prediction of exactly 0.500 counts as a home pick, everywhere: `pick` in the
prediction file, the per-game grade, and `win_prob_metrics`. The metrics function previously
used a strict `>`; the grader's tests caught the mismatch and it is now `>=` throughout. The
backtest numbers are unchanged to three decimals (exact 0.500 predictions are vanishingly rare).

## Verified

- 8 tests: the official rule (latest pre-kickoff pass wins; a post-kickoff pass is ignored),
  the per-game math including an ATS push, aggregation, unplayed games skipped not scored, a
  partial week marked incomplete then completing, a week with nothing played writing no file,
  and the idempotency check above.
- Real run against 2026 Week 1: "no completed games yet", no results file written. The first
  real results file appears Tuesday Sept 15 when the grade job (Phase 7) or a manual run
  follows Monday night.
- `pytest` → 72 passed (offline), `ruff` clean.

## Deferred

- No per-team or per-situation breakdowns yet (home/away, favourite/underdog). The per-game
  grades carry everything needed; the site can compute views over them in Phase 6.
- Results for a week whose prediction file is later superseded by a corrected one: the naming
  convention for corrections is still undefined (see Phase 4), so the grader currently takes
  every `<season>_<week>_*.json` as a pass. Define both together when the first correction is
  ever needed.
