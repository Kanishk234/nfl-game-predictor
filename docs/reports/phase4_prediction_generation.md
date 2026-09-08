# Phase 4 — Prediction generation

**Status:** complete. **Date:** 2026-09-08.

## What was built

**`predict.py`** — the prediction pass. `python -m nfl_predict.predict --pass early|late`:

1. Rebuilds the processed frame from nflreadpy and runs the leakage gate.
2. Computes the target week (the week of the next kickoff) and **refuses to proceed if that
   week's first kickoff has passed** — the gate, enforced in code before anything is written.
3. Retrains on every completed game (the expanding-window retrain CLAUDE.md describes; the late
   pass retrains too, so Thursday's result is in Sunday's model).
4. Selects the target week's games that have not kicked off. Early pass: the whole week. Late
   pass: the Sunday/Monday slate; games already played keep the early pass's prediction.
5. Takes one timestamp, fetches the odds snapshot with it, predicts, and writes both files —
   `data/predictions/<season>_<week>_<pass>.json` and `data/odds/<season>_<week>_<pass>.json`
   — stamped with that single instant. Both refuse to overwrite an existing file.

## What a prediction row carries

`p_home`, `pred_margin`, `pick`, and beside them the frozen Vegas baseline for the same game:
`spread_line` (their margin prediction), `p_home_moneyline` (their win probability, de-vigged),
and `p_home_from_spread` (win probability derived from the spread the way the backtest did it,
so live weeks stay comparable to the 2021–2025 holdout history). A game with no line gets
`vegas: null`, never an invented number.

The record header carries the gate evidence (`seconds_before_first_kickoff`) and full model
provenance: learner, params, feature count and hash, training-set size, the last game trained
through, training timestamp, and the git commit.

## The bug this phase found

**Upcoming games had no features.** `elo_ratings`, `rolling_form` and `pbp_form` only emitted
rows for played games, so every 2026 Week 1 row had null Elo and form, and the model would have
predicted from imputed column means — plausible-looking numbers, all garbage. Nothing in Phases
1–2 could have caught it, because the backtest only ever scores played games.

Fixed with one shared as-of mechanism (`latest_state_before_kickoff`): every team-level feature
is now a per-team *post-game state* stamped with when it became knowable, looked up as the
latest state strictly before each kickoff. For played games this reproduces the old
shift-by-one rolling window exactly — the holdout backtest is bit-identical (0.6306 / 65.0%)
— and for unplayed games it yields the team's current state. The QB features already worked
this way; now everything does.

## Verified — 2026 Week 1, early pass

- Ran for real: **16/16 games**, written **31.0 hours before the first kickoff** (Wed Sept 9,
  8:20 PM ET), all 16 with a three-book Vegas line, prediction and odds stamped with the same
  instant.
- Model trained through `2025_22_SEA_NE` (6,499 games), commit `5c78b84`, 30 features.
- Picks agree with the Vegas favourite in 13 of 16 games. The three disagreements (BAL @ IND,
  BUF @ HOU, NYJ @ TEN) are all near-coin-flips where the model leans home and the market leans
  away — consistent with the home-lean the backtest documented, and exactly what the live
  season is for.
- The API key is verifiably absent from both written files.
- `pytest` → 73 passed, including 9 for the pass: early/late selection, exact-kickoff exclusion,
  immutability, the gate inside `run`, and an end-to-end run with stubbed odds and model
  confirming the two files share one timestamp.

## Retrain-vs-reuse, decided

The plan left open whether the Sunday pass retrains or reuses Tuesday's model. It retrains: the
fit is seconds, and reusing would mean Sunday's predictions ignore Thursday's result while the
features already reflect it — an inconsistency for no saving. `--no-retrain` exists for
debugging only.

## Deferred

- The processed frame rebuild downloads ~25 seasons of play-by-play each run (~70 s). Fine for
  two runs a week; a cache would be a Phase 7 nicety, not a need.
- No automatic corrected-run naming yet. If a published prediction is ever found wrong, the
  rule is a new file with its own timestamp; the naming convention for that can wait until it
  is needed.
