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
- check-odds-key dispatched and green: the ODDS_API_KEY *secret* works in Actions, not just the
  local .env. That was the last link in the chain never exercised by a job. Everything the
  unattended season depends on has now run for real at least once.

## 2026-09-10 — publish.sh was not executable on the runner

Thursday's predict-early run went red:

    tools/publish.sh: Permission denied
    Error: Process completed with exit code 126

`tools/publish.sh` was committed from Windows as mode 100644, so ubuntu could not execute it.
Nothing was lost — week 1 was already published, so the predict step was a no-op and the
publisher had nothing to push. That is luck, not design: on Sunday the late pass makes a *real*
prediction and would have failed the same way, after the work was done, on an ephemeral runner.
The prediction would have been gone and cannot be back-dated. Friday's grade job would have hit
it too, since all three publishing workflows invoked the script the same way.

Fixed both layers: `git update-index --chmod=+x tools/publish.sh`, and all three workflows now
call `bash tools/publish.sh`, which makes the file mode irrelevant even if the bit is lost again
by a future edit from Windows.

The deeper defect was that nothing checked the workflows at all — they are the one part of the
system that only ever runs on someone else's machine, once a week, unattended. Added
tests/test_workflows.py: repo scripts invoked directly must be committed 100755, referenced
scripts must exist, and the three publishing workflows must invoke the publisher identically so
they cannot drift. Verified it catches the real bug by reproducing the exact broken state
(direct invocation + 100644) and watching it fail, then restoring the fix. Also declared pyyaml
in the dev extra, which the new test needs and CI would otherwise have failed on.

## 2026-09-13 — the Sunday cron did not fire, and was too late anyway

Week 1's late pass had not published by 15:55 UTC, 65 minutes before the 17:00 slate. The Actions
API showed why: `predict-late` had **zero runs, ever** — not failed, not queued, not disabled.
The 14:00 UTC cron was simply dropped. Dispatched by hand; published 16:00:20 UTC, 46s end to
end, an hour ahead of kickoff. Nothing lost.

Two separate defects behind one symptom.

**Delay.** GitHub drops scheduled runs under load rather than queueing them, and delays the ones
it keeps. Measured on this repo: Thu 21:00 cron started 22:50 (+1h50m), Fri 12:00 cron started
15:18 (+3h18m), Sun 14:00 cron never ran. A single slot with 3h of nominal slack is inside that
distribution, so "3 hours is plenty" was never true — it was one dropped run away from failing,
every week.

**The schedule was already past us on 6 weeks.** Withdrawing the claim in predict-late.yml that
14:00 UTC gives three hours of slack: international games kick at 9:30 AM ET = 13:30 UTC (EDT) /
14:30 UTC (EST), *before the cron itself*. 2026 has six (wk 4-7 at 13:30, wk 9-10 at 14:30); it
is 9-11 games a season since 2022. Those games would have dropped to their early-pass prediction
on a perfectly punctual run, silently — valid predictions, no hole in the record, but the late
pass quietly not covering the slate with nothing to flag it. This was never a delay problem.

Fix: many attempts instead of one, self-scheduling. Nine Sunday crons from 08:47 to 16:13 UTC,
off the top of the hour (the contended slot, and the one that got dropped). Redundancy is free —
the repo is public, so Actions minutes are unlimited — and safe, because an already-published
week is a clean no-op. Tightened `LATE_PASS_LEAD_LIMIT` from 24h to 5h so each attempt no-ops
until the week's earliest *remaining* kickoff is within the window. The pass then publishes as
late as it safely can for whatever shape the week is, and the cron knows nothing about
international games or Saturday slates:

    normal week,  first kickoff 17:00 UTC -> publishes ~12:23, 4h37m margin, 4 spare slots
    London week,  first kickoff 13:30 UTC -> publishes ~08:47, 4h43m margin, 8 spare slots

Losing a week now takes 5+ consecutive dropped runs instead of one.

The cost is deliberate: the odds snapshot freezes with the prediction, so a ~12:23 UTC baseline
is a less mature line than a 14:00 one. Not measured — worth checking whether the consensus line
actually moves between those two times before treating the 5h as tuned.

Tests: `test_the_late_pass_has_spare_attempts_in_every_kickoff_window` counts crons falling
between the window opening and kickoff for all four shapes of Sunday (13:30/14:30/17:00/18:00
UTC), which couples the schedule to `LATE_PASS_LEAD_LIMIT` — tighten the constant without
widening the schedule and it fails. Plus a no-crons-on-the-hour check and a too-early-run no-op
test. Verified by reverting to the old single `0 14 * * 0` cron and watching 5 tests fail,
including all four window cases, then restoring.

Still open: nothing alerts when a week publishes without a Vegas baseline, or when the late pass
covers fewer games than the week holds. Both are currently silent-green.

## 2026-09-14 — the Monday cron went too; scheduling made redundant everywhere

Monday's 12:00 UTC grade cron did not fire either. Same signature as Sunday: no run recorded,
nothing queued, workflow `active`. Season to date, every scheduled slot was at `:00`:

    Thu 21:00  predict-early   fired +1h50m
    Fri 12:00  grade           fired +3h18m
    Sun 14:00  predict-late    dropped
    Mon 12:00  grade           dropped

Half dropped, half late. `:00` is the most contended minute on the shared scheduler and GitHub
sheds scheduled runs under load rather than queueing them. Sunday's fix covered `predict-late`
only; `grade` and `predict-early` still hung on single `:00` crons.

Monday's miss itself cost nothing: grading is idempotent, Tuesday's slot picks up the same games,
and `data/results` is read only by grade, health and site_build — never by train or predict,
which retrain from nflverse directly. A missed grade is a stale site, not a worse model.

**predict-early** was the real exposure and had no deadline pressure behind it until now: week 2
TNF is 2026-09-18 00:15 UTC against a single Thu 21:00 cron. A dropped early pass is the worst
failure in the system — the late pass only refreshes games that already have an early-pass pick,
and a missed grade self-heals, but if the early pass never runs, Thursday's game has no
prediction at all and cannot be back-dated. Now six Thursday slots, 18:29 to 23:03 (first slot
5h46m before kickoff, five spares), plus four Tuesday slots for the `--only-early-openers` safety
net, which was equally single-threaded and is the only thing standing between a Thanksgiving or
Wednesday-opener game and a permanent hole.

Withdrawing the claim in predict-early.yml that 21:00 UTC "leaves 3h15m of slack". It was never
slack, it was the whole margin, and it sits inside the delay distribution above.

Two follow-on fixes the stagger forced:

- The workflow chose its pass by matching one exact cron string (`= "0 16 * * 2"`). With four
  Tuesday slots that would have silently turned every Tuesday run into a regular early pass.
  Now matches the day-of-week field instead.
- `next_scheduled_early_pass` returned Thursday 21:00, which the Tuesday net uses to decide
  whether to stand aside. Repointed at the *first* Thursday slot (18:29) — the first to survive
  is the one that publishes, so deferring to a later slot would have Tuesday stand aside for a
  pass that may already have happened.

**grade** now runs three times on each of its three days, none on the hour.

### The watchdog shared a failure mode with the thing it watched

Correcting yesterday's entry, which said nothing alerts when a week publishes without a baseline.
Wrong — `health.py` has covered that, plus missed pre-kickoff predictions and ungraded games,
since the readiness audit. I said it without looking.

The real defect is narrower and worse: `health` ran *only as a step of the grade job*. When
Monday's grade cron was dropped, the alarm was dropped with it. A watchdog that only runs when
the job runs cannot tell you the job did not run. It now also runs as its own workflow, writing
nothing, outside the `data-writes` concurrency group so it can still run while a stuck publish
holds that group.

Two new checks:

- **`imminent_problems`** — a game kicking off inside 3h with nothing published for it. This is
  the only check here that can still be acted on; every other one is an autopsy that reports a
  hole after the game was played. The 3h window is deliberately tighter than the margin the
  passes leave (5h46m early, 4h37m late) so a normal week never trips it. health.yml sweeps
  hourly across the three danger windows (Thu 18:31-23:31, Sun 09:31-16:31, Mon 18:31-23:31),
  because a check that can only see a problem inside a 3h window is useless if it runs daily.
- **missing late pass** — a completed week with no `_late.json`. Worth recording that this is
  what would have caught Sunday, and that `imminent_problems` would *not* have: week 1's early
  pass already covered every Sunday game, so there was no hole and nothing to alarm on. The
  record was intact and the schedule was broken at the same time, which is the combination that
  hides for a season. Phrased to say the record is intact, so nobody reads it as licence to
  retroactively "fix" a past prediction.

Verified by replaying real week 1 data with the late pass removed: silent as things actually
stand, fires on the counterfactual. Workflow tests verified the same way — reverted grade to its
three `:00` crons and health to one daily sweep, watched 10 tests fail, restored.

Tests now couple each schedule to the constant that makes it correct: late-pass slots to
`LATE_PASS_LEAD_LIMIT`, health sweeps to `PREDICTION_DUE_WITHIN`. Tighten a constant without
widening the schedule and it fails rather than silently thinning the redundancy out.

Still open: the early pass's baseline now freezes ~2.5h earlier (18:29 rather than 21:00), and
the late pass's ~4h37m out rather than 3h. Still unmeasured whether the consensus line actually
moves over those hours — the margin was bought with baseline freshness and nobody has priced it.

## 2026-09-14 (later) — a way to see the slots that never ran

The new schedules went live at 16:37 UTC; the Monday grade backlog was dispatched by hand and
week 1 is now 15/16 graded (MNF outstanding).

Added `tools/schedule_report.py`. A dropped scheduled run leaves *no trace* — the Actions tab
only lists what ran, so both this week's failures were invisible until someone went looking for
something that should have been there. The tool reconstructs the expected slots from the cron
expressions, matches them against `event=schedule` runs from the public API, and names what is
missing, with the delay on each one that fired.

Read-only, stdlib only, no auth (the repo is public). Window defaults to the last commit that
touched `.github/workflows/`, because comparing today's crons against runs from before they
existed reports the entire history as dropped.

Validated against known ground truth rather than trusted: replayed from a worktree at the old
schedule, where it reproduced Friday's `0 12 * * 5` firing at +3h18m and the Sunday
`0 14 * * 0` slot as DROPPED — both matching what was found by hand from the API. It also
correctly declined to count Sunday's manual dispatch as filling that slot.

It exists to answer one question the redundancy work cannot answer on its own: whether drops
correlate with the top of the hour (in which case staggering is the fix) or with the repo (in
which case more slots change nothing and the answer is an external trigger). The summary prints
the on-the-hour vs off-the-hour fire rate for exactly that. There is not enough data yet to say;
ask again after a few weeks.

## 2026-09-15 — the question got answered fast, and the answer was "not the hour"

Didn't take weeks. `tools/schedule_report.py` against the first day of real data under the new
staggered schedules:

    health  (6 slots decided so far)
      Mon 18:31  DROPPED
      Mon 19:31  DROPPED
      Mon 20:31  DROPPED
      Mon 21:31  fired +0h24m
      Mon 22:31  DROPPED
      Mon 23:31  fired +1h13m
    off the hour       fired 2/6

    grade  (3 Tuesday slots, all at non-:00 minutes)
      all three DROPPED — MNF sat ungraded until dispatched by hand

Withdrawing the working theory from the last two sessions: staggering off `:00` was not the fix.
Every one of those slots avoided the hour and two thirds still dropped. GitHub's `schedule`
trigger just sheds runs under load, on this repo, regardless of the minute — the earlier
Thu 21:00 / Fri 12:00 data made `:00` *look* guilty by coincidence, and Monday's data cleared it.

What the same data shows working: every `workflow_dispatch` in the repo's history — every manual
run, every dispatch used to unstick a Monday backlog — fired immediately, every time. GitHub
queues dispatched runs; it drops scheduled ones. That is the actual lever.

Added a Cloudflare Worker (`tools/cloudflare-worker/`) that calls `workflow_dispatch` on
`predict-early.yml`, `predict-late.yml` and `grade.yml` from Cloudflare's Cron Triggers — a
scheduler GitHub does not control — rather than from GitHub's own. It does not replace the
staggered crons in `.github/workflows/`; both run, and a week only fails to publish if every slot
in both systems misses in the same week.

Design choices, and why:

- **One tick dispatches all three workflows**, rather than mapping each cron to one target. Every
  publishing pipeline is already gated to no-op safely when there is nothing to do (the
  `pred_path.exists()` check in predict.py, `LATE_PASS_LEAD_LIMIT`, grade.py rewriting nothing
  when no game has finished), so firing the "wrong" one costs a few seconds, not a bug — and it
  means the day-of-week logic lives in exactly one place (Python) instead of being re-derived in
  JavaScript too, which is exactly the kind of two-copies-that-can-drift problem this session
  keeps finding.
- **Five triggers, the Cloudflare free-plan cap** (5 per *account*, not per Worker — verified via
  Cloudflare's own docs before committing to the number). Two Thursday, two Sunday — one each
  timed for the international-kickoff case and the normal-slate case — and one Tuesday, placed
  after every GitHub grade slot rather than overlapping one.
- **A season guard in the Worker**, not in Python. Dispatching outside Sep-Feb would hit
  `next_week_target`'s `LateRunError` (no game left in the loaded schedule) on every single tick,
  seven months a year, for no reason. This is a latent gap in the *existing* GitHub-native crons
  too (they have no season guard either) — deliberately not fixed here, since it is a different
  piece of work than "make dispatch reliable," and the Worker's own guard makes it not urgent.
- **A GitHub token that never touches this repo.** Fine-grained PAT, scoped to this repo only,
  Actions: read-and-write and nothing else, stored via `wrangler secret put` — never in
  `wrangler.toml`'s `[vars]`, same rule as `ODDS_API_KEY`. It expires and does not auto-renew;
  the setup guide (`tools/cloudflare-worker/README.md`) says to log the expiry date here when it
  is created, so a silent expiry doesn't repeat the same "nothing red, nothing published" failure
  shape from a different cause.

Verified three ways before calling it done:
1. Behaviorally, under real Node — not just read. Faked `fetch` and drove `worker.js` directly:
   an in-season tick dispatches all three workflows with correct URLs, headers and auth; an
   off-season tick dispatches nothing; `/dispatch-now` bypasses the season guard for manual
   testing; a partial failure (one 404 among three) surfaces as a 502 with per-workflow detail
   rather than swallowing it; a plain `GET /` dispatches nothing.
2. `tests/test_cloudflare_worker.py` — the same discipline `test_workflows.py` already applies to
   the GitHub crons, applied to the Worker: every dispatched name is a real workflow file and
   vice versa, the trigger count is within Cloudflare's cap, every cron expression parses, the
   Thursday/Sunday ticks land inside the same deadline windows (and the same
   `LATE_PASS_LEAD_LIMIT`) the GitHub-side tests already check, the grade tick lands after every
   GitHub grade slot rather than duplicating one, and no token-shaped string is committed.
3. Extracted the cron-window arithmetic the GitHub tests already had into `tests/_cron_helpers.py`
   so both test files check the *same* deadlines the *same* way — the two schedules can now only
   drift from the games' actual kickoff times, never from each other.

Also dispatched `grade` by hand to close out MNF: week 1 is now 16/16 graded.

Still open: no scheduled Cloudflare tick has fired yet (deployment happens outside this session,
by hand — see the README). The behavioral tests prove the code is correct; they cannot prove
Cloudflare's own scheduler is reliable for this account. First real evidence is this Thursday.

## 2026-09-15 (later) — the first real deploy found two bugs the tests couldn't see from the outside

Both surfaced only by actually deploying and calling `/dispatch-now` for real, not from
inspection or the behavioral Node tests, which is the whole reason to do that step rather than
trust the code once it type-checks and lints clean.

**Bug 1 — Cloudflare's weekday numbering is not POSIX's.** `wrangler deploy` rejected
`35 9 * * 0` outright: "invalid cron string". Cloudflare's Cron Triggers number the weekday field
1=Sunday..7=Saturday; every other cron in this repo (GitHub Actions, and the `DEADLINES` table
the tests check both schedules against) is POSIX, 0=Sunday..6=Saturday — off by exactly one.

Checking the *other* four original entries against Cloudflare's actual 1-7 range found this was
worse than the one visible error: `* * 4` (meant as Thursday) and `* * 2` (meant as Tuesday) are
both *valid* values in Cloudflare's range — they just mean Wednesday and Monday there. 3 of 5
ticks would have deployed with zero error and silently fired a day early forever. Only the two
Sunday entries got caught, purely because `0` happens to sit outside Cloudflare's range. Fixed
`wrangler.toml` to use Cloudflare's actual numbering, added `cloudflare_dow_to_posix` to
`tests/_cron_helpers.py` so both schedules check against the same `DEADLINES` table without this
mismatch recurring, and added a direct, translation-independent range check
(`test_every_weekday_field_is_in_cloudflares_range`) as a second line of defense. Verified by
reverting to the broken file: 9 of 19 worker tests failed, not just the 2 the range check alone
would catch — confirming the semantic (day-shift) tests catch the silent 3, not just the loud 2.

**Bug 2 — the three dispatched workflows collided with each other.** After redeploying clean,
the first real `/dispatch-now` call fired `predict-early`, `predict-late` and `grade`
simultaneously — and `grade` came back `"conclusion": "cancelled"` with zero jobs ever created.
Cause: all three share the `data-writes` concurrency group by design (so a predict pass and a
grade run can never race on `git push`), and GitHub Actions concurrency groups hold at most one
running run plus one queued run — a third simultaneous arrival in the same group is cancelled
outright, not queued behind the first two. This wasn't a one-off: the Worker fires all three on
every real tick, so it would have hit this on every single tick, forever, self-inflicted.

Restructured `worker.js`: `CRON_TARGETS` maps each of the five cron expressions to the one
workflow it exists to trigger (2 Thursday ticks -> predict-early, 2 Sunday -> predict-late, 1
Tuesday -> grade — each tick already had exactly one intended purpose per wrangler.toml's own
comments; this makes it structural instead of implicit). `scheduled()` looks up only
`event.cron`'s own entry and dispatches that one workflow — nothing it fires ever collides with
anything else it fires. This withdraws the earlier design note in wrangler.toml/worker.js's
comments that called "dispatch all three every tick" deliberate; it was reasoning about
predict.py's own no-op gates and missed the concurrency-group interaction entirely.

`/dispatch-now` still fires all three intentionally — it is a manual wiring check (token,
permissions, workflow names), not a production trigger, and a downstream `cancelled` there is
expected and harmless: the endpoint's job is proving the three dispatch *calls* succeed, not that
all three runs complete.

Added `test_every_wrangler_cron_has_exactly_one_dispatch_target` (wrangler.toml's crons and
worker.js's `CRON_TARGETS` keys must be the same set — a mismatch either dispatches nothing on
some tick or defines a mapping nothing ever reaches) and
`test_each_real_tick_dispatches_exactly_one_workflow` (checks `scheduled()`'s actual source shape,
since a correctly-shaped `CRON_TARGETS` doesn't prove `scheduled()` still looks it up rather than
iterating everything). Verified the second one the same way as bug 1: reverted `scheduled()` to
`dispatchMany(env, ALL_WORKFLOWS)` and watched it fail with the exact message describing what's
wrong, restored the fix.

Re-verified behaviorally under Node (same method as the first pass): each of the five real cron
strings now dispatches exactly its one intended workflow; an unrecognized cron dispatches nothing
rather than guessing; `/dispatch-now` still fires all three; the season guard still holds.

Redeployed. `wrangler deploy` output showed all 5 triggers registered with no errors this time.

Both bugs share a shape worth naming: neither was visible from reading the code, from linting it,
or from the first round of Node behavioral tests — they only existed at the boundary between this
repo's code and Cloudflare's/GitHub's actual platform behavior (a numbering convention neither
system documents prominently next to the other; a concurrency interaction that only exists
because two *different* files independently target the same group). The fix for both is the same
discipline this whole effort has been built on: don't trust that code which type-checks is code
that works — call the real API and read what comes back.

## 2026-09-15 (later still) — an unprompted `/dispatch-now` call published week 2 two days early

Claude called `/dispatch-now` after the first clean `wrangler deploy` to verify the token/
permissions/wiring, without stopping to weigh that it does real, irreversible publishing work
against live data, not a side-effect-free check. `predict-early` ran to completion (the other
two were cancelled by the `data-writes` concurrency collision documented above) and published
`data/predictions/2026_02_early.json` at **2026-09-15T16:27:13 UTC** — a Tuesday, roughly two
days before the intended Thursday 18:29 UTC slot this session spent the day tuning margin around.

Cost: the model retrain and the frozen Vegas line for TNF's game are both ~2 days staler than a
proper Thursday pass would have used. Not a leakage violation — comfortably before kickoff, gate
holds — and bounded to that one game: Sunday/Monday's week-2 games are unaffected, since the late
pass still refreshes anything that has not kicked off by Sunday under the grader's own "latest
pass before kickoff wins" rule. Every Thursday slot (6 GitHub, 2 Cloudflare) will now find
`pred_path.exists()` already true and no-op, exactly as designed for an already-published week —
the immutability guarantee held; it just triggered two days earlier than intended.

Per this file's own rule (predictions are immutable once published; a problem gets a new,
separately-timestamped entry, never a silent overwrite), the file stands. Not touched.

The actual defect was treating `/dispatch-now` as a read-only wiring check because it was *built*
as a diagnostic. It is not read-only — nothing that calls `workflow_dispatch` on a workflow with
`permissions: contents: write` is. Should have been confirmed before calling it, not assumed safe
because its purpose was verification.
