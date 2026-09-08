# Phase 3 — Odds baseline integration

**Status:** complete. **Date:** 2026-09-08.

## What was built

**`odds/fetch.py`** — fetches current NFL lines from The Odds API (free tier), parses them into
per-bookmaker lines plus a consensus, matches every event to an nflverse `game_id`, and freezes
the result as `data/odds/<season>_<week>_<pass>.json`. Exposes `fetch_snapshot` for the Phase 4
prediction pass to call, so the baseline is frozen at the same instant as the prediction (a
design invariant), and a CLI (`python -m nfl_predict.odds.fetch --pass early`) for on-demand use.

## Snapshot format

Per game: `game_id`, `kickoff_utc`, per-bookmaker `spread_home` / prices / `moneyline_home` /
`moneyline_away` / `total` / `last_update`, and a `consensus` block:

- `spread_line` — median across books, **sign-flipped to match nflreadpy's convention**
  (positive = home favoured), so it is directly comparable to the historical `spread_line` the
  backtest scored against.
- `total_line` — median total.
- `p_home_moneyline` — median de-vigged home win probability (proportional overround removal).
- `n_books`.

Plus snapshot metadata: `fetched_at_utc`, markets/bookmakers/region, quota headers, and an
explicit `games_without_lines` list so a partial week is visible rather than silent.

## Verified

- Real fetch for 2026 Week 1: **16/16 games matched**, three books each. Written to a scratch
  path, not `data/odds/` — the first committed snapshot belongs to the Phase 4 prediction run
  that freezes both together.
- Week 1 consensus sample: SEA −3 vs NE (p_home 0.618), total 44.5. Sign convention confirmed
  against the schedule's `spread_line` for the same game.
- The key is verifiably absent from the written snapshot (grepped for the literal value).
- `pytest` → 14 odds tests: spread sign, de-vig math, unknown-team skip, schedule matching,
  wrong-week rejection, immutability, and four tests that the key cannot escape via exceptions,
  non-200 responses, or the snapshot itself.

## Free-tier budget

The API returns the **entire remaining season** in one call (272 events), billed one credit
per market per region: 3 credits per snapshot. Two passes a week is ~25 of the 500 monthly
credits. Quota headers are recorded in every snapshot so drift is visible.

## Secret handling

Per the project's standing rule: `ODDS_API_KEY` is read from the environment only (locally from
a gitignored `.env` if unset; in CI from an Actions secret). It has to travel as a query
parameter, so every string that could carry it out — exception text, response body, URL — is
scrubbed before leaving the module, and a write is refused if the snapshot contains request
parameters. The frontend never touches the API; it reads the committed JSON.

## Bug found and fixed

`load_teams()` lists the Rams twice (`LA` and `LAR`). A plain name→abbreviation dict kept `LAR`,
which the schedule never uses, so `2026_01_SF_LA` silently failed to match on the first real run
(15/16). The map is now restricted to abbreviations present in the target week's schedule, with
a regression test.

## Deferred

- Only US books (DraftKings, FanDuel, BetMGM). Enough for a consensus; more books cost nothing
  extra in credits but were not needed.
- No retry/backoff on transient API errors; the pass fails loudly instead. Revisit in Phase 7
  if a scheduled run ever hits one.
