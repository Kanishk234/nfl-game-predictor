# Phase 7 — Automation

**Status:** wired; exit criterion (one full unattended live week) pending Week 1. **Date:** 2026-09-08.

## The scheduled jobs

| workflow | cron (UTC) | ET | does |
|---|---|---|---|
| `predict-early` | `0 21 * * 4` (Thu) | 5 PM EDT / 4 PM EST | predicts every game still ahead → commit → rebuild site → deploy |
| `predict-late` | `0 14 * * 0` (Sun) | 10 AM EDT / 9 AM EST | re-predicts the Sunday/Monday slate with Thursday's result in the model |
| `grade` | `0 12 * * 5`, `0 12 * * 1`, `0 12 * * 2` | 8 AM EDT | grades what has finished, updates history, rebuilds the site |

Thursday 21:00 UTC is 3h15m before the earliest possible TNF kickoff (8:15 PM ET) in the tighter
EDT half of the season; Sunday 14:00 UTC is three hours before the 1 PM ET slate. Grading runs
three times a week — Friday after Thursday night, Monday after the Sunday slate, Tuesday after
Monday night — so the cards fill in with results as the week goes rather than all at once.

**There is no separate retrain job.** Every prediction pass refits on every completed game, so
Thursday's pass already carries the whole previous week. The ET→UTC conversion and the DST
reasoning live in each workflow's header.

## Design decisions

**The bot commits data; that is the only thing that pushes to `main` besides you.** Unattended
pre-kickoff publishing is impossible otherwise. The commit step is narrow: it stages only
`data/predictions`, `data/odds` (predict) or `data/results` (grade), exits cleanly when there is
nothing new, and commits as `github-actions[bot]` with a message derived from the file it just
wrote (`publish 2026_01_late prediction`). It rebases on `main` before pushing, on a
full-history checkout, so a race with a human push does not fail the run.

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

## Verified

- All five workflow files parse; triggers and job graphs confirmed programmatically.
- The commit step's shell was simulated locally end to end (stage → detect → derive message).
- **Not yet verified: an actual scheduled run.** That is the phase's exit criterion and can
  only happen live. The safe way to exercise the full chain before Sunday is to dispatch
  `grade` by hand: it is idempotent, writes nothing for a week with no completed games, and
  still runs checkout → install → run → commit-skip → deploy.

## Dispatching by hand

Safe. Each `(week, pass)` file is immutable and a re-run for a week already published is a
no-op. Dispatching `predict-late` early would still burn that week's late slot with an early
prediction, so only do it deliberately.

**A week that opens before Thursday needs a manual pass.** 2026 Week 1 opened on Wednesday; its
prediction was published by hand on the Tuesday. The Thursday cron covers every normal week.

## Deferred

- `site_build` does not exist yet (Phase 6). The deploy job currently republishes the
  placeholder page; once the builder exists, the cron workflows gain a build step before commit
  and the `site/` output is committed with the data.
- Bye weeks / the postseason: `next_week_target` targets the week of the next kickoff, so a bye
  week simply predicts the following week early. Postseason (single elimination, one game per
  round for some teams) is Phase 8.
