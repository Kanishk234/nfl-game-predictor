# Phase 1 — Data pipeline

**Status:** complete. **Date:** 2026-09-08.

## What was built

- **`data/games.py`** — schedules 2002–present, normalized to `kickoff_utc`. nflreadpy gives
  `gameday`/`gametime` as *local Eastern strings with no timezone*; we attach
  `America/New_York` and convert, so the mid-season EDT→EST switch is handled instead of being
  an hour wrong for half the season. Raises rather than warns if any game lacks a date/time — a
  null kickoff makes the gate unenforceable.
- **`data/schedule.py`** — week targeting and the mechanical kickoff guard (see below).
- **`data/features.py`** — Elo, rolling EPA/margin form, franchise-code normalization.
- **`data/pipeline.py`** — assembles the modelling frame, runs the gate, writes
  `data/processed/games.parquet`.

## Coverage

| | |
|---|---|
| Rows, 2002–2026 | **6,771** |
| Completed games | **6,499** |
| Upcoming (2026) | 272 |
| Features | 19 |
| `spread_line` nulls on completed games | **0** |
| Feature null rate (worst column, completed) | 0.31% |
| as-of lead before kickoff | min 5 min, median 2d 12h |

**Historical odds are a solved problem, contrary to the plan's assumption.**
`load_schedules()` carries `spread_line` and `total_line` with zero gaps across all 6,499
completed games since 2002. Phase 2 can score against the real closing line on every held-out
game — no "acknowledging the gap." Moneylines are absent pre-2006 and patchy to 2010, so
**spread is the baseline, not moneyline**. The Odds API is therefore only needed for *upcoming*
games (Phase 3), which materially shrinks the free-tier risk.

## The leakage gate is now real

The Phase 0 placeholder asserted nothing. It now has 11 tests, and the ones that matter are
synthetic: they construct the exact situations that leak and assert the guard fires. A green run
over real data only proves no leak happened to occur; these prove the guard works.

The subtle hazard is Elo. It is league-wide, so a naive walk in kickoff order lets a Sunday
1:00pm game learn from *another* Sunday 1:00pm game that had not finished. Elo here defers every
rating update until that game's completion time (`kickoff + 4h`), so simultaneous games start
from identical ratings.

**Strictness is not cosmetic.** With `<=` instead of `<`, 57 real games violate the gate: under
the 4-hour assumption a 4:15pm ET game completes at exactly 8:15pm ET, the instant the Sunday
night game kicks off. A result knowable only *at* kickoff is not knowable *before* it. The gate
is `<` throughout, matching the schedule guard.

Rows with no prior information carry a null as-of rather than a fabricated one — null means "drew
on nothing", which is trivially safe and honest.

## Two bugs found and fixed

**1. The cron schedule was unsound.** `predict-thu.yml` assumed the week always opens with
Thursday Night Football. It does not — since 2002 there have been 136 Saturday, 13 Friday, 6
Wednesday and 5 Tuesday regular-season games, and **2026 Week 1 opens Wednesday Sept 9 at
00:20 UTC**. A Thursday 21:00 UTC run would have published Week 1's prediction ~21 hours after
kickoff: an unrecoverable gate violation in the first live week.

Fixed two ways. The passes are now `predict-early` (Tuesday 16:00 UTC, ~32h before the earliest
possible Wednesday opener, and after Monday night has been graded) and `predict-late` (Sunday
14:00 UTC, unchanged). More importantly the cron is no longer trusted: `assert_before_kickoff`
computes the target week's *earliest actual kickoff* and refuses to publish if it has passed, so
a delayed runner fails loudly instead of publishing a late prediction.

**2. Relocated franchises had no EPA.** Schedules keep the abbreviation a franchise used at the
time (`SD`, `STL`, `OAK`); team stats use the current one (`LAC`, `LA`, `LV`). The join silently
produced null EPA for every San Diego, St. Louis and Oakland game — **11.5% of completed games
had null offensive form**. After mapping, **0.31%**. A regression test asserts no schedule team
code lacks a stats counterpart, so a future nflverse rename fails loudly.

## The 2020 decision: keep it, flagged

Measured rather than assumed. 2020 is the only season since 2002 with a sub-50% home win rate:

| | home win % | mean home margin |
|---|---|---|
| 2020 | **49.6%** | **+0.06** |
| All other seasons | 56.2% | +2.29 |

Home-field advantage effectively vanished — consistent with empty stadiums. **Decision: keep the
season with a `no_crowd` indicator feature (269 games flagged), rather than excluding it.**

Three reasons. The anomaly is concentrated in a single learnable effect (HFA), which an indicator
lets the model neutralize for those rows while still using their Elo and EPA signal. Excluding
would discard ~4% of training data. And decisively — Elo is a *sequential chain*, so deleting a
season mid-chain would corrupt every rating from 2021 onward, a worse distortion than the one
being corrected.

Worth noting 2019 was also weak (51.6%, −0.14 margin) with no pandemic, so season-level HFA is
noisy; the indicator is the modest intervention, not a claim that 2020 is fully explained.

## Numbers

- `pytest` → **33 passed**; `pytest -m leakage` → **11 passed**; `ruff` → clean
- `python -m nfl_predict.data.pipeline` → 6,771 rows, gate passes
- Elo `elo_diff` sign alone predicts the winner at **63.4%** (a feature sanity check, not a model)

## Deferred

- **Elo is untuned.** K=20, home advantage 65, ⅔ season carryover are conventional defaults, not
  fitted. Phase 2 owns tuning.
- **`GAME_DURATION = 4h` is an assumption**, deliberately conservative (real games run ~3h10m).
  Later completion means a stricter gate, so erring high is the safe direction.
- **Upstream gap:** JAX's 2002 home games are missing from team stats entirely (8 team-game
  rows). Left as nulls; GBDTs handle them natively.
- **No injury/QB features.** Roster-lock timing needs its own as-of handling; deferred rather
  than rushed into the gate.
- **`predict-early`/`predict-late` renaming has not been reflected in CLAUDE.md or PLAN.md**,
  which still say "Thursday"/"Sunday" and tag passes `thu`/`sun`. Those files are yours to edit.
