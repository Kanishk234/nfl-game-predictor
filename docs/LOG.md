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

## 2026-09-08 — Phase 3 odds baseline

- `odds/fetch.py`: The Odds API -> per-book lines + consensus (median spread sign-flipped to nflreadpy convention, de-vigged moneyline prob) -> matched to nflverse game_id -> immutable snapshot. `fetch_snapshot` for Phase 4; CLI for on-demand.
- Real fetch 2026 wk1: 16/16 games, written to scratch (first committed snapshot comes with the Phase 4 prediction run). Key verified absent from output. 14 tests.
- Budget: whole season in one call, 3 credits/snapshot, ~25/month of 500.

Broke / fixed:
- `load_teams()` has the Rams as both LA and LAR; dict kept LAR -> `2026_01_SF_LA` silently unmatched (15/16). Name map now restricted to the week's schedule codes. Regression test added.

Open:
- **Week 1 opens Wed Sept 9 (00:20 UTC Sept 10).** Phase 4 predict.py must run before then for Week 1 to be in the track record.

## 2026-09-08 — Phase 4 prediction generation

- `predict.py`: rebuild frame -> gate (refuse after first kickoff) -> retrain -> predict unplayed games in target week -> fetch odds with the SAME timestamp -> write both immutable files.
- Rows carry `p_home`, `pred_margin`, `pick` + frozen Vegas `spread_line`, `p_home_moneyline`, `p_home_from_spread` (backtest-comparable). Header: gate seconds, full model provenance.
- **Ran for real: `data/predictions/2026_01_early.json` + `data/odds/2026_01_early.json`, 16/16 games, 31.0h before first kickoff.** First entry in the track record. 13/16 picks agree with Vegas.

Broke / fixed:
- **Upcoming games had null Elo/form features** — the feature builders only emitted played games. Would have silently predicted from imputed means. Fixed with a shared `latest_state_before_kickoff` as-of helper; holdout backtest bit-identical after.
- Decided: late pass retrains (seconds; keeps model consistent with features).

Open:
- **Push before Wed Sept 9, 8:20 PM ET** or Week 1's prediction is not provably pre-kickoff.
- Late pass for Week 1 should run Sunday morning (Phase 7 cron; manual until then).

## 2026-09-08 — Phase 5 grading

- `grade.py`: per-week results (model + frozen Vegas, per-pass, ATS w/l/push) + season history with calibration bins. History rebuilt from per-week files each run -> idempotent by construction (tested).
- Official prediction per game = latest pass generated before that kickoff; post-kickoff passes ignored (tested).
- Tie rule unified: p_home >= 0.5 is a home pick in predict, grade, and `win_prob_metrics` (was strict `>` in metrics; grader tests caught it).
- Real run vs Week 1: no completed games yet, no file written. First real grading Tue Sept 15.
- 8 grading tests; suite 72 passed offline.

Open:
- Correction-file naming convention still undefined (needed by both predict and grade); define when first needed.

## 2026-09-08 — Phase 7 automation

- Crons wired: grade Tue 12:00 UTC, predict-early Tue 16:00 UTC, predict-late Sun 14:00 UTC. Each: run -> bot commits only its data dir -> push (rebase, full-history checkout) -> deploy.
- `GITHUB_TOKEN` pushes don't trigger workflows -> `deploy-site` made `workflow_call` and chained as a downstream job; checkout pinned to `ref: main`.
- Shared concurrency group `data-writes`; gate failure = red job, nothing written.
- All YAML parses; commit shell simulated locally. **Real scheduled run not yet verified** — that's the exit criterion.

Open:
- Dispatch `grade` by hand once to exercise the chain safely. Never dispatch predict-* early (immutable slots).
- Phase 6 site_build must be added to the workflows when it exists.

## 2026-09-08 — Phase 6 static site

- `site_build.py`: Python renders `site/index.html` from data/ — ledger with per-game spread gauge (ours vs line vs final), provenance per pass, season-to-date strip + weekly accuracy + calibration (when enough games), collapsed past weeks, 2021-25 backtest section, verification section. Zero JS, inline SVG.
- `data/backtest.json` now committed (moved out of gitignored processed/).
- Chart colours validated with the dataviz validator, light and dark; dark pair re-stepped to pass.
- Wired `site_build` + `git add site` into all three cron workflows.
- 9 site tests; suite 81 passed offline. Real build: 16 rows, balanced HTML, no script/http/apiKey.

Broke / fixed:
- `strftime("%-d")` is glibc-only; crashed on Windows. Dates formatted by hand.

Open:
- Visual check in a browser (none available in this session). Phone layout + gauge legibility.

## 2026-09-08 — Site redesign after review

- Two reviewers: the single-page ledger was cluttered and hard to read. Rebuilt: a card per game (pick in words, probability bar with Vegas marker, spreads in team terms, result + verdict once played), compact week table at the end, one page per week with a week strip, season page separate. Provenance collapsed.
- Multi-page output regenerated every run: `index.html` (current week), `weeks/<s>_<ww>.html`, `season.html`. Nothing manual for a new week.
- 10 site tests; all pages parse. Live site fetched and confirmed rendering Week 1 before the redesign.

## 2026-09-08 — Site, second review round

- Feedback: Season tab unreadable; ours-vs-Vegas unclear on cards; wanted team colours, plainer text, and a visual for the spread.
- Cards now: explicit "Our pick" / "Vegas favorite" rows with team-colour chips; probability bar filled in the picked team's colour; one-sentence plain-English call ("SEA should win by about 5. Vegas has SEA by 3."); a labelled spread number line (Us above, Vegas below, final margin as a bar once played); team nicknames in headers; team-colour top border.
- Season page rewritten as "Track record" in plain English; the backtest is a two-sentence dry-run summary with details collapsed.
- Team colours hardcoded from nflverse (32 teams) so the build needs no network.

## 2026-09-08 — Site, third review round

- Team colours were vanishing in dark mode (navy/black primaries on a dark panel). Added contrast-aware selection: per team, per surface, primary if it reaches 3:1, else secondary, else lightened; emitted as `--tl`/`--td` CSS variables so the page picks the right one for the colour scheme. Test asserts all 32 teams pass on both surfaces.
- Spread number line had colliding labels. Replaced with aligned rows (Us / Vegas / Final) — fixed text column, bar from a shared zero coloured by the favoured team. Labels are never positioned by value.
- No browser tools available this session; user reviewed live.

## 2026-09-08 — Schedule change and gate loosened to per-game

- Cron: predict-early Thu 21:00 UTC (was Tue 16:00), predict-late Sun 14:00 UTC (unchanged), grade Fri/Mon/Tue 12:00 UTC (was Tue only) so results appear through the week.
- **Gate is now per game.** `assert_before_kickoff` fails only once the week's *last* kickoff has passed; `WeekTarget.earliest_kickoff` is the next kickoff still ahead. A Thursday pass after a Wednesday opener covers the rest of the week instead of refusing. Games already started are excluded by `games_to_predict`, and the grader ignores post-kickoff predictions, so nothing loosens the invariant.
- **Re-running a published week is a green no-op**, not `PredictionExistsError` (this is what made the first scheduled `predict-early` red).
- Site: removed the disagreement summary line; fixed card overflow (grid `minmax(min(20rem,100%),1fr)`, `min-width:0` on flex children, bar flex-basis 0).
- No separate retrain job: every pass refits on all completed games.

## 2026-09-08 — Site layout: one week grid, wider page

- Per-slot grids made single-game slots (Wed, Thu, Sun night, Mon) look broken. Now one continuous `.week-grid`; slot labels span all columns as full-width dividers with a rule and a game count ("Sunday 1:00 pm — 9 games, 3 played"). Cards stay uniform and flow.
- Page widened 76rem -> 84rem, prose 66ch -> 74ch, header lede uncapped. Season page's dry-run and trust sections are now two-column (`.split`) instead of a narrow text column in a wide page.
- Backtest numbers moved from a prose sentence into a stat strip (ours bold, Vegas muted).

## 2026-09-08 — Vegas colour and card density

- **Vegas is pink (#F06BB0), not gold.** Gold sat ~15 CIELAB units from Vikings/Steelers/Packers/Chiefs gold. Measured every candidate against all 32 teams' primary, secondary and lightened-for-dark colours: pink is ~34 from its nearest neighbour and no NFL team uses pink at all, so the market marker can never read as a team. Test asserts it matches no team colour.
- **Cards decluttered:** dropped the "Spread | away | even | home" axis header (the values already name the team); the spread is now one grid with a "Points" caption spanning both rows. Card split into three zones separated by hairlines (matchup / pick / points), padding 1.15rem, card min-width 21rem, gap 1.1rem. Eight rows down to six with more air between them.

## 2026-09-08 — Verified in a browser at last

- localhost approved in the extension, so the site was finally reviewed by looking at it rather than by reading markup. Confirmed: spread rows stack and align (Us above Vegas, both growing from a shared centre), 3-column grid at 1440px, slot dividers reading correctly, pink Vegas markers distinct from every team colour on the page.
- The "spread still looks broken" report was a stale browser cache: the deployed page already had the fix (33 `spread-row` elements, pink `#F06BB0`). Hard refresh resolves it.
- Logos flashed as bare white discs: ESPN serves one 500px PNG per team and ignores resize params, so `loading="lazy"` left them empty for a moment after paint. Now eager with `decoding="async"`, disc softened to #F2F3F5. 32 distinct images per week, cached across pages.

## 2026-09-08 — Bye weeks checked; Thanksgiving gap found and closed

- **Byes need no code.** A team on bye is simply absent from that week's games: rolling form and Elo stay put (correct — nothing happened), `rest_diff` picks up the 13+ day rest, and weeks with 13-15 games render like any other. Nothing to change.
- **Real gap found instead.** A Thursday 21:00 UTC pass misses any game kicking off earlier that week. Measured 2002-2026: **31 weeks would have lost a game** — Thanksgiving's 12:30 PM ET game *every season*, plus Christmas (2025 wk17) and Wednesday openers (2012, 2026 wk1, 2026 wk12).
- Fix: second `predict-early` cron, Tuesday 16:00 UTC, with `--only-early-openers`. It publishes only when the week's first kickoff precedes the next scheduled Thursday pass; otherwise it exits without writing. Simulated over 2026 it fires for weeks 1 and 12 only.
- `next_scheduled_early_pass()` in schedule.py; 4 tests including the Thanksgiving-week case. Suite 94 passed.

- Added `docs/WEEKLY_FLOW.md`: the running schedule (what fires when, what each job writes, what is immutable, how the gate is enforced, manual commands, what a red/no-op job means). Linked from PLAN.md.
- Wrote `README.md` as the repo's front door: what the project is, a user-facing "when the site updates" table, what a game card shows, the honest 65.0% vs Vegas 66.5% comparison, why the record is trustworthy (with links straight into `data/predictions`, `data/odds`, `data/results` and Actions), the model in a paragraph, and how to run it. All relative links verified; the 30-feature claim checked against `FEATURE_COLUMNS`.

## 2026-09-08 — Pre-emptive fix: a crash that would have hit Sunday's pass

- Simulated Sunday morning (Wed/Thu games marked played, pbp release not yet up) and the pipeline **crashed**: `nfl.load_pbp(seasons=[2026])` raises `ValueError("Season must be between 1999 and 2025")`, and the loaders only caught `(ConnectionError, OSError)`.
- The trigger is exact: 2026 only enters the loader's season list once its first game is `is_played`. So the failure would have appeared for the first time on Sunday's live pass, not before.
- Fixed: `_FEED_NOT_READY = (ConnectionError, OSError, ValueError)` across all four per-season loaders. 6 new tests parametrised over all three exception types, plus an end-to-end "partly published season" test.
- Verified the simulation end to end: completed games 6499 -> 6501, the Sunday game's Elo as-of advances from Feb (last season) to Sep 11 (Thursday night), and pbp form still builds 13,542 rows with the feed refusing.
- Suite 101 passed.

## 2026-09-08 — Dress rehearsal of a graded week

- `tools/preview_graded_week.py` replays real completed weeks (2025 wk1-2) through the whole pipeline — walk-forward predict, freeze the real historical Vegas lines, grade against real finals, render the site — into a scratch directory. `data/` untouched (verified with git status).
- Confirms the parts that had only ever been exercised by test fixtures: the Final row in the spread block, the score + verdict line ("GB won, ✓ we were right and our side covered the spread"), the week results strip, slot labels counting played games, and the season page's weekly accuracy chart.
- Round trip is sound: 32 games graded, per-week and season summaries agree, history rebuilt.

## 2026-09-08 — Full-season simulator, and the bug it found

- `tools/preview_season.py` replays any real season through the whole pipeline at any moment in time: walk-forward predictions for both passes, real historical Vegas lines frozen beside them, grading of whatever has finished as of `--as-of`, and a rendered site. Writes to a scratch dir; `data/` untouched (verified).
- Replaced the single-purpose `preview_graded_week.py`.
- Verified states that had only ever existed as test fixtures: 6 weeks with **both passes** per week (the late pass superseding for Sunday games — wk5 graded 2 games from `early`, 12 from `late`), an 18-slot week strip with the current week marked, a **mid-week partial** (`Week 6, 2025, 1 of 15 played`), bye weeks (14- and 15-game weeks), and slot labels counting played games.
- **Confirmed: index.html always shows the newest week**, with older weeks at their own URLs and every page carrying the full strip.
- **Bug found and fixed.** In a part-played week the weekly accuracy chart plotted the current week's rate from a single graded game (0%), and 0% is below the chart's 30% floor, so the lines drew *outside the plot area*. Now `line_chart` clamps to the axis range, and only **complete** weeks are charted, with the caption saying so. Partial weeks still count in the season totals.
- 2 new tests; suite 103 passed.

## 2026-09-08 — Full-season simulation, and three bugs it caught

`tools/simulate_season.py` walks the real cron timeline across a whole season in order — every
Tuesday safety net, Thursday early pass, Sunday late pass, Friday/Monday/Tuesday grade — calling
the **real** `predict.run`, `grade.run` and `site_build.main` with only the clock, the nflreadpy
frame and the odds fetch moved back in time. It asserts eight invariants after every single job
and fails on the first violation.

2025 replay: **128,123 invariant checks passed**, 44 prediction files, 285/285 games given a
pre-kickoff pick, 285 graded, 24 site pages. 22 weeks including the playoffs. Both safety-net
weeks fired (Thanksgiving Nov 25, Christmas Dec 23). Final: 64.6% vs Vegas 66.0%, spread error
10.03 vs 9.67, ATS 138-146-1.

Bugs found and fixed:

1. **The late pass burned Week 1's slot eight days early.** The Sunday cron fires on the Sunday
   *before* the season too; it targeted Week 1 and published `2025_01_late.json` with all 16
   games, so Week 1 never got its real Sunday-morning refresh. Added `LATE_PASS_LEAD_LIMIT` —
   a late pass whose next kickoff is more than 24h away is not late and skips. Same hazard
   existed in the gap before the playoffs.
2. **Every grade run rewrote every result file.** `graded_at_utc` was a wall-clock stamp, so all
   22 files differed on each run and the scheduled job would have pushed a junk commit three
   times a week forever. Removed; results are now a pure function of predictions + scores.
   (Same bug class as `rebuilt_at_utc` in history.json, fixed earlier — this one survived.)
3. **A tie rendered as "it was a tie, a tie"**, and the week table's Vegas column showed the
   *home* team's probability while the card showed the *favourite's* — 26% next to 74% for the
   same game. The tie now reads "It was a tie, so the pick does not count"; both table columns
   now describe the team we picked ("Our odds" / "Vegas on our pick").

The 2025 season contained exactly one tie (GB 40, DAL 40 in week 4), which is why this only
surfaced under a full-season replay.

## 2026-09-08 — Hardened for an unattended season

Four failure modes that a person would otherwise have had to notice:

1. **Transient network failures cost a whole week.** Added `retry.py`; the nflverse load and the
   odds request now retry four times with backoff (~45s). Permanent failures (401/403/422/429 —
   bad key, exhausted quota) raise `OddsPermanentError` and fail fast instead.
2. **An odds outage killed the prediction.** Now the pass publishes anyway with `vegas: null`
   and records the reason in the snapshot. A missing baseline is one empty column; a missing
   prediction is a hole that cannot be filled, because a prediction made after kickoff is not a
   prediction.
3. **A rescheduled game broke the gate.** The grader used the kickoff *recorded in the
   prediction file*. If a game is moved forward, a pass that looked pre-kickoff when written
   might not be. It now judges against the schedule's actual kickoff and lists moved games.
4. **Nothing noticed a missed week.** `health.py` runs at the end of every grade job and exits
   non-zero on: a played week with no prediction file, a played game with no pre-kickoff pick, a
   game ungraded four days after finishing, or a prediction published without a baseline. Scoped
   to seasons we have actually published for, so last season is not reported as our hole.

13 new tests. Suite 125 passed; the full-season simulation still reports 128,123 checks clean.

## 2026-09-08 — A red CI run exposed silent data loss

CI failed on `test_no_schedule_team_code_is_missing_from_team_stats`: GitHub returned a **500**
for one nflverse parquet. The test failure was upstream noise, but chasing it found a real bug in
production code.

**Every per-season loader caught `(ConnectionError, OSError, ValueError)` alike and skipped that
season.** So a transient 500 on, say, 2005 silently removed 2005 from the training data — the
model would fit on a hole and nobody would ever know. Reproduced it: `_team_game_epa([2004, 2005,
2006])` with a 500 on 2005 returned only 2004 and 2006, no error.

This is the third instance of the same failure shape in this project (after the SD/STL/OAK
franchise-code nulls and the `load_pbp` ValueError). The pattern: an exception handler broad
enough to swallow a real problem alongside an expected one.

Fixed by separating the two meanings:

- **not published yet** — a `ValueError` (a season nflreadpy refuses) or a ConnectionError
  wrapping a **404**. That season contributes nothing; carry on. This is normal in September.
- **the download failed** — a 500, a timeout, a reset. Retried four times with backoff, and if
  it still fails the exception escapes and the job goes red. Training on a hole is worse than
  not training.

`_load_season_feed()` in features.py now wraps all four loaders. Draft picks degrade to
replacement level with a printed note, since that prior is not load-bearing.

The network test now **skips** rather than fails when nflverse is unavailable: it exists to catch
a franchise rename on our side, and a red build for someone else's outage only teaches us to
ignore red.

4 new tests pinning the distinction, including one proving a recovering server still yields the
season. Full suite (including network): 136 passed.

## 2026-09-08 — Publishing made race-proof; the site is no longer committed

Audited what remains for autonomy and found the worst failure mode yet in the commit step.

**Measured first:** a cold `build_frame()` on a fresh runner takes **38s** (nflreadpy caches in
memory only, so every run downloads). Well inside the 30-minute job timeout — that risk is clear.

**The bug.** Each job committed `site/` — 24 generated HTML files — *before* rebasing. Any
concurrent job that had landed produced a rebase conflict in generated files, and the job died
after making the prediction but before pushing it. On an ephemeral runner that prediction is
gone, and a prediction cannot be back-dated, so the week gets a permanent hole.

**The fix is architectural, and CLAUDE.md already called it:** "the site is a generated view over
those files and can always be rebuilt from them from scratch." So `site/` is now gitignored build
output. The deploy job installs, runs `site_build` against the data on `main`, and uploads
straight to Pages. Generated files can no longer take part in a merge at all.

`tools/publish.sh` now owns publishing for all three workflows. Building it surfaced three more
ways a shell step can strand committed work, each fixed and then verified against a real
three-repo git race:

- `git add data/odds` when that directory does not exist is **fatal** under `set -e` — after the
  prediction was already made. Now stages only paths that exist.
- `git pull --rebase` refuses to run with any unrelated dirty file in the tree. Now `--autostash`.
- A commit made by an attempt that then failed to push was reported as "nothing new to publish"
  on the next run and abandoned. Now an unpushed commit is detected and pushed.

Verified end to end with two clones racing against a bare origin: a grade commit lands first, the
predict job rebases onto it, and origin ends with both. A re-run is a clean no-op, exit 0.

## 2026-09-08 — weekday coverage, quota warning, completed-week preview

- Checked every weekday the NFL has ever used, 2002-2026 (6,771 games): Sun 5732, Mon 448,
  Thu 334, Sat 233, Fri 13, Wed 6, Tue 5. Every one of them is preceded by a publish pass —
  0 games uncovered. Saturday's 233 all fall to the Thursday pass (earliest Saturday kickoff is
  00:00 UTC, after the 21:00 UTC Thursday run); the stray Wed/Fri/Tue games are split between
  the Thursday pass and the Tuesday safety net, all with >=21h of lead.
- Confirmed weekly retraining from the simulated 2025 season's published files: training set
  grows on every single pass, 6214 games at week 1 early -> 6498 at week 22 late (+284 learned
  in-season).
- health.py: added an odds-quota check (warn under 100 requests remaining, read from the newest
  snapshot that recorded one). Moved it and the "published without a baseline" check into
  odds_problems(), ahead of the early returns in problems() - both previously sat after the
  "nothing played yet" return, so neither could fire before the season's first game finished.
  That was a real hole: an exhausted quota would have gone unreported exactly when there was
  still time to do something about it.
- site_build: the card legend still said the Vegas marker was "bronze". It has been pink since
  the colour-distance work. Fixed the wording, and the same stale claim in phase6's report.
- Rebuilt the full 2025 simulation's 24 site pages with current code to show what a completed
  week looks like.
- Card results were a sentence you had to read ("SEA won, ✓ we were right, and our side covered
  the spread"). Replaced with a tinted band: one large ✓/✗/= glyph, the score with the winner
  picked out and the loser muted, and a chip each for the pick and the spread. Colour is never
  the only signal — every state carries its own glyph too. Verified by rendering all 22
  simulated weeks: 183 hit bands, 101 miss, 1 tie, and every chip variant including the push.
  Two site tests asserted on the old sentence and were rewritten to assert the structure.

## 2026-09-08 — grading Vegas head to head

- Vegas was already scored on every game with the same metrics as us (accuracy, Brier, spread
  MAE, per game and per week) — it just was not visible on the cards. Added:
  - a Vegas chip on every graded card, in the Vegas pink, so the head-to-head is readable per
    game. Where we agreed it mirrors ours; where we disagreed the two glyphs differ, which is
    exactly the case worth finding.
  - `head_to_head()` in grade.py: of the games where we and Vegas named different winners, how
    many each side got right. Agreeing with the market says nothing either way, so this is the
    sharpest read on whether the model adds anything. Ties excluded (nobody can be right).
    Shown in the summary strip as "When we disagree — 14 of 32", hidden until there is one.
- Re-graded the simulated 2025 season with the new code: 285 games, us 64.6% / Vegas 66.0%,
  spread MAE 10.03 vs 9.67, and 32 disagreements which we lost 14–18. Consistent with the
  holdout backtest — a solid model that has not beaten the market.
- Fixed a latent crash found while doing it: the chip generator unpacked its tuples before the
  `if c` filter ran, so any absent chip raised TypeError. A graded game with no Vegas line would
  have taken down every week page. Now filters first; regression test covers the no-line card.
- Season chart: 22 weeks of x labels collided ("wk 10wk 11"), and both series ending at 100%
  stacked their end labels. Labels are now thinned to fit with the last week always kept, and
  colliding end labels are nudged apart.

## 2026-09-08 (late) — pre-season readiness check

Audited what has actually run on Actions versus what is only assumed to work.

Proven on real Actions: `deploy-site` (the live site serves the newest commit's markup), and
`grade` + `tools/publish.sh` (commit 733ba47, "grade 2026-09-08 results", authored by
github-actions[bot] — so checkout, GITHUB_TOKEN push and permissions are all good).

Never yet exercised on Actions: the odds fetch. Week 1's early prediction was committed by hand
during phase 4 development, using the local `.env` key — the `ODDS_API_KEY` *repository secret*
has never been read by a job. Worse, Thursday's run cannot test it: predict.py returns at the
"already published" check (predict.py:183) before it ever reaches fetch_snapshot (line 209), so
the no-op skips the odds path entirely. First real use would have been Sunday's late pass.

An outage there is non-fatal by design — the prediction publishes and the gap is recorded — so a
bad secret produces a *green* run with a silently baseline-less week. Added
`.github/workflows/check-odds-key.yml`: manual dispatch only, asserts the secret is present,
then fetches to `$RUNNER_TEMP` and prints the quota. Commits nothing, so it cannot burn the
week's slot.

Also confirmed: no cron could have fired today (the Tuesday cron landed on main at 16:50 UTC,
after its own 16:00 slot; every other cron landed later still). First scheduled runs are
Thursday 21:00 UTC (a no-op, week 1 is published) and Friday 12:00 UTC (the first real grading).
Retrain cost measured at 113s of a 30-minute budget: 106s downloading feeds, 5.6s fitting.
