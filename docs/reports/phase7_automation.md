# Phase 7 — Automation

**Status:** wired; exit criterion (one full unattended live week) pending Week 1. **Date:** 2026-09-08.

## The three scheduled jobs

| workflow | cron (UTC) | ET | does |
|---|---|---|---|
| `grade` | `0 12 * * 2` (Tue) | 8 AM EDT / 7 AM EST | `python -m nfl_predict.grade` → commit `data/results` → deploy |
| `predict-early` | `0 16 * * 2` (Tue) | 12 PM EDT / 11 AM EST | `predict --pass early` → commit `data/predictions` + `data/odds` → deploy |
| `predict-late` | `0 14 * * 0` (Sun) | 10 AM EDT / 9 AM EST | `predict --pass late` → same |

Ordering on Tuesday is deliberate: grade at 12:00, predict at 16:00, so the new week's model
trains on a fully graded previous week. The ET→UTC conversion and the DST reasoning are
documented in each file's header so they are never re-derived.

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

**The gate fails the job.** If the runner is late and the week's first kickoff has passed,
`predict.py` raises `LateRunError` before writing anything, and the job goes red. A red job is
the correct outcome; a late prediction is not.

**The key** enters exactly one step, as `${{ secrets.ODDS_API_KEY }}`, on the predict step.
GitHub masks it in logs; the code side scrubs it from every exception (Phase 3).

## Verified

- All five workflow files parse; triggers and job graphs confirmed programmatically.
- The commit step's shell was simulated locally end to end (stage → detect → derive message).
- **Not yet verified: an actual scheduled run.** That is the phase's exit criterion and can
  only happen live. The safe way to exercise the full chain before Sunday is to dispatch
  `grade` by hand: it is idempotent, writes nothing for a week with no completed games, and
  still runs checkout → install → run → commit-skip → deploy.

## Do not dispatch the predict workflows early

Each `(week, pass)` file is immutable. A manual Tuesday dispatch of `predict-late` would burn
Week 1's late slot with a Tuesday prediction and the real Sunday run would refuse to overwrite
it. The headers say so in capitals.

## Deferred

- `site_build` does not exist yet (Phase 6). The deploy job currently republishes the
  placeholder page; once the builder exists, the cron workflows gain a build step before commit
  and the `site/` output is committed with the data.
- Bye weeks / the postseason: `next_week_target` targets the week of the next kickoff, so a bye
  week simply predicts the following week early. Postseason (single elimination, one game per
  round for some teams) is Phase 8.
