# Phase 7 — Automation

**Status:** wired; exit criterion (one full unattended live week) pending Week 1. **Date:** 2026-09-08.

## The scheduled jobs

| workflow | cron (UTC) | ET | does |
|---|---|---|---|
| `predict-early` | `0 21 * * 4` (Thu) | 5 PM EDT / 4 PM EST | predicts every game still ahead → commit → rebuild site → deploy |
| `predict-early` | `0 16 * * 2` (Tue) | 12 PM EDT / 11 AM EST | safety net: publishes **only** if the week opens before Thursday's pass |
| `predict-late` | `0 14 * * 0` (Sun) | 10 AM EDT / 9 AM EST | re-predicts the Sunday/Monday slate with Thursday's result in the model |
| `grade` | `0 12 * * 5`, `0 12 * * 1`, `0 12 * * 2` | 8 AM EDT | grades what has finished, updates history, rebuilds the site |

Thursday 21:00 UTC is 3h15m before the earliest possible TNF kickoff (8:15 PM ET) in the tighter
EDT half of the season; Sunday 14:00 UTC is three hours before the 1 PM ET slate. Grading runs
three times a week — Friday after Thursday night, Monday after the Sunday slate, Tuesday after
Monday night — so the cards fill in with results as the week goes rather than all at once.

**The Tuesday safety net.** A Thursday-afternoon pass misses any game that kicks off earlier in
its own week, and that is not rare — it happens **every season since 2002**: Thanksgiving always
has a 12:30 PM ET game, Christmas sometimes does, and 2012, 2025 and 2026 opened weeks on a
Wednesday. Measured across 2002–2026, 31 weeks would have lost a game. So `predict-early` also
runs Tuesday with `--only-early-openers`, which publishes only when the target week's first
kickoff falls before the next scheduled Thursday pass and otherwise exits without writing
anything. Simulated over 2026 it fires for exactly two weeks — week 1 (Wednesday opener) and
week 12 (Thanksgiving eve) — and stays out of the way for the other sixteen, which keep the
freshest possible Thursday model.

**There is no separate retrain job.** Every prediction pass refits on every completed game, so
Thursday's pass already carries the whole previous week. The ET→UTC conversion and the DST
reasoning live in each workflow's header.

## Design decisions

**The bot commits data, and only data.** `tools/publish.sh` stages `data/predictions` +
`data/odds` (predict) or `data/results` (grade), commits as `github-actions[bot]` with a message
derived from the file just written, rebases with `--autostash`, and pushes with three retries.

**The site is never committed.** It is gitignored build output; the deploy job rebuilds it from
whatever data is on `main` and uploads it straight to Pages. This was a correctness fix, not
tidiness: committing 24 generated HTML files that change on every run meant any concurrent job
produced a rebase conflict in files nobody merges, and the job died *after* making the prediction
but before pushing it. On an ephemeral runner that prediction is gone, and a prediction cannot be
back-dated. Data files are new files with unique names, so a rebase between jobs is
conflict-free.

The script is defensive about the specific ways a shell step strands committed work: it stages
only paths that exist (`git add` on a missing directory is fatal under `set -e`), autostashes so
an unrelated dirty file cannot block the rebase, and pushes an unpushed commit from a previous
attempt rather than reporting "nothing to do".

**Deploy is called, not triggered.** Pushes made with the built-in `GITHUB_TOKEN` never fire
other workflows, so a data commit would not have triggered `deploy-site`'s `push` trigger.
`deploy-site.yml` is now also `workflow_call`, and each cron workflow runs it as a downstream
job. Its checkout pins `ref: main` so it deploys the commit the caller just pushed rather than
the commit the caller started from.

**One data write at a time.** All three workflows share a `concurrency` group so two jobs can
never race on a push. `cancel-in-progress: false` means a queued job waits rather than being
dropped.

**The gate is per game, and only a truly late run fails.** A pass predicts only games whose
kickoff is still ahead, so a delayed Thursday run simply covers fewer games. `LateRunError` and
a red job are reserved for a run after the entire week has started, when there is nothing valid
left to publish. A week that is already published exits green without touching the files —
immutability means there is nothing to do, and nothing to do is not a failure.

**The key** enters exactly one step, as `${{ secrets.ODDS_API_KEY }}`, on the predict step.
GitHub masks it in logs; the code side scrubs it from every exception (Phase 3).

## Built to survive an unattended season

- **Transient failures are retried** — four attempts with backoff, ~45s, on the nflverse load
  and the odds request. A 30-second outage no longer costs a week's prediction, which can never
  be back-filled.
- **Permanent failures fail fast** — a bad key, a 403, an exhausted quota. Retrying proves
  nothing and delays the job.
- **An odds outage does not cost the prediction.** The pass publishes with `vegas: null` and
  records why in the snapshot. The Vegas line is the comparison, not the product.
- **Rescheduled games are judged against reality.** The grader compares each prediction to the
  game's actual kickoff from the schedule, not the kickoff recorded in the prediction file, and
  lists moved games under `rescheduled_games`.
- **The record audits itself.** `python -m nfl_predict.health` runs at the end of every grade
  job and fails it if anything is missing, which sends GitHub's failure email.

## Verified

- All five workflow files parse; triggers and job graphs confirmed programmatically.
- **A whole season replayed through the real cron timeline** (`tools/simulate_season.py`):
  128,123 invariant checks across 2025's 22 weeks, 44 prediction files, 285/285 games given a
  pre-kickoff pick and graded, 24 pages built. It found three bugs, all fixed — see docs/LOG.md.
- **Not yet verified: an actual scheduled run.** That is the phase's exit criterion and can
  only happen live.

## Dispatching by hand

Safe. Each `(week, pass)` file is immutable and a re-run for a week already published is a
no-op. Dispatching `predict-late` early would still burn that week's late slot with an early
prediction, so only do it deliberately.

~~**A week that opens before Thursday needs a manual pass.**~~ Withdrawn — the Tuesday safety-net
cron handles those automatically now. 2026 Week 1 was published by hand before that existed.

## Deferred

- ~~`site_build` does not exist yet (Phase 6); the deploy job republishes the placeholder page.~~
  Withdrawn — Phase 6 shipped. All three workflows now run `python -m nfl_predict.site_build`
  after their data step and commit `site/` alongside the data, so the page is rebuilt on every
  publish and every grade.
- Bye weeks / the postseason: `next_week_target` targets the week of the next kickoff, so a bye
  week simply predicts the following week early. Postseason (single elimination, one game per
  round for some teams) is Phase 8.
