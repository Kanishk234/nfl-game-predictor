"""Check the record for holes, and fail loudly if there are any.

    python -m nfl_predict.health

Runs after every grade job. The point of an unattended season is that nobody is watching, so
something has to watch on their behalf: a job that silently publishes nothing is worse than one
that goes red, because a missing prediction can never be back-filled — a prediction made after
kickoff is not a prediction.

Exits non-zero when the record is incomplete, which turns the workflow red and sends GitHub's
failure email. Everything it reports is actionable.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta

import polars as pl

from nfl_predict.data.games import load_games
from nfl_predict.data.schedule import GAME_DURATION
from nfl_predict.grade import RESULTS_DIR, load_predictions, official_predictions
from nfl_predict.predict import PREDICTIONS_DIR

#: How long after a game finishes before an ungraded result counts as a problem. The grade job
#: runs three times a week, so anything unscored after four days has been missed, not delayed.
GRADING_GRACE = timedelta(days=4)

#: The free tier allows 500 requests a month and a snapshot costs three (one per market), so a
#: season needs roughly 25. Warn well before it matters: a pass that cannot fetch odds still
#: publishes, but the track record loses its baseline, which is half the point of the project.
QUOTA_WARN_BELOW = 100

#: How close a kickoff may get with nothing published before that counts as an emergency.
#:
#: Every other check here is an autopsy: it reports a hole once the game has been played, when
#: nothing can be done about it. This one is the only check that can still be acted on — it
#: fires while a human can dispatch the job by hand, which is exactly how 2026 week 1's late
#: pass was saved (noticed 65 minutes before kickoff, dispatched, published with an hour to
#: spare) and exactly what nothing would have told us.
#:
#: Deliberately tighter than the margin the passes are scheduled to leave — the early pass
#: publishes about 5h46m before TNF, the late pass about 4h37m before the Sunday slate — so a
#: normal week never trips it, and a fire means attempts have genuinely been dropped rather than
#: merely delayed.
PREDICTION_DUE_WITHIN = timedelta(hours=3)


def operating_seasons() -> set[int]:
    """Seasons we have actually published for. A season we never ran is not a hole in our
    record — it is simply not our record."""
    return {int(p.name[:4]) for p in PREDICTIONS_DIR.glob("*_*_*.json")}


def odds_problems() -> list[str]:
    """Checks that do not depend on any game having been played yet — an exhausted quota is
    worth hearing about on a Tuesday in the off-week, not only after a game finishes."""
    found: list[str] = []
    snapshots = sorted(PREDICTIONS_DIR.parent.glob("odds/*.json"))

    # a snapshot that recorded a provider failure, so the gap is explained and not silent
    for path in snapshots:
        snap = json.loads(path.read_text())
        if snap.get("unavailable"):
            found.append(f"{path.name}: published without a Vegas baseline ({snap['unavailable'][:80]})")

    # the quota, read from the most recent snapshot that recorded one. Running out mid season
    # would quietly cost every remaining week its baseline.
    for path in reversed(snapshots):
        remaining = json.loads(path.read_text()).get("source", {}).get("quota", {}).get("requests_remaining")
        if remaining is None:
            continue
        if int(remaining) < QUOTA_WARN_BELOW:
            found.append(f"odds API quota down to {remaining} requests (as of {path.name}); "
                         f"a season needs about 25")
        break
    return found


def imminent_problems(games: pl.DataFrame, now: datetime, seasons: set[int]) -> list[str]:
    """Games about to kick off with no prediction published for them yet.

    Scoped to seasons we have actually published for, like every other check: a season we never
    ran is not a hole in our record, and this should stay silent in the off-season.
    """
    found: list[str] = []
    due = games.filter(
        pl.col("season").is_in(list(seasons))
        & (pl.col("kickoff_utc") > now)
        & (pl.col("kickoff_utc") <= now + PREDICTION_DUE_WITHIN)
    )
    for season, week in sorted({(r["season"], r["week"]) for r in due.iter_rows(named=True)}):
        week_games = games.filter((pl.col("season") == season) & (pl.col("week") == week))
        kickoffs = dict(week_games.select("game_id", "kickoff_utc").iter_rows())
        official = official_predictions(load_predictions(season, week), kickoffs)
        block = due.filter((pl.col("season") == season) & (pl.col("week") == week))
        missing = sorted(set(block["game_id"]) - set(official))
        if missing:
            soonest = block.filter(pl.col("game_id").is_in(missing))["kickoff_utc"].min()
            found.append(
                f"{season} week {week}: kicks off {soonest.isoformat()} "
                f"({(soonest - now).total_seconds() / 3600:.1f}h away) with no prediction "
                f"published for {missing} — dispatch the pass NOW; after kickoff it is gone"
            )
    return found


def problems(games: pl.DataFrame, now: datetime) -> list[str]:
    found: list[str] = []
    seasons = operating_seasons()
    if not seasons:
        return found
    found += odds_problems()
    found += imminent_problems(games, now, seasons)
    season = max(seasons)
    played = games.filter(
        (pl.col("season") == season) & (pl.col("kickoff_utc") + GAME_DURATION < now)
    )
    if played.is_empty():
        return found

    for week in sorted(played["week"].unique().to_list()):
        week_games = games.filter((pl.col("season") == season) & (pl.col("week") == week))
        started = week_games.filter(pl.col("kickoff_utc") + GAME_DURATION < now)
        passes = load_predictions(season, week)

        # 1. a week whose games have been played with nothing published at all
        if not passes:
            found.append(f"{season} week {week}: {started.height} games played, no prediction file")
            continue

        # 2. individual games that never got a prediction before they kicked off
        kickoffs = dict(week_games.select("game_id", "kickoff_utc").iter_rows())
        official = official_predictions(passes, kickoffs)
        missed = sorted(set(started["game_id"]) - set(official))
        if missed:
            found.append(f"{season} week {week}: no pre-kickoff prediction for {missed}")

        # 3. the late pass never ran at all for a week that is now over. This is not a hole in
        #    the record — the early pass covers those games and its predictions are valid — so
        #    it is deliberately not phrased as one. It is here because it is the only trace a
        #    dropped Sunday cron leaves behind: on 2026-09-13 the late cron was never fired and
        #    nothing in the system noticed, because every game still had an early-pass pick.
        #    Every regular-season and playoff week has a Sunday slate, so a completed week with
        #    no late pass means the Sunday schedule failed, not that there was nothing to do.
        if "late" not in passes and started.height == week_games.height:
            found.append(f"{season} week {week}: completed with no late pass — the early pass "
                         f"covered it, so the record is intact, but the Sunday schedule did not "
                         f"run; check for dropped scheduled runs")

        # 4. games finished long enough ago that they should have been graded by now
        path = RESULTS_DIR / f"{season}_{week:02d}.json"
        graded = {g["game_id"] for g in json.loads(path.read_text())["games"]} if path.exists() else set()
        overdue = sorted(
            g["game_id"] for g in started.iter_rows(named=True)
            if g["game_id"] not in graded and g["kickoff_utc"] + GRADING_GRACE < now
        )
        if overdue:
            found.append(f"{season} week {week}: finished but ungraded after "
                         f"{GRADING_GRACE.days} days: {overdue}")

    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--warn-only", action="store_true", help="report but always exit 0")
    args = ap.parse_args(argv)

    found = problems(load_games(), datetime.now(UTC))
    if not found:
        seasons = operating_seasons()
        scope = f"season {max(seasons)}" if seasons else "nothing published yet"
        print(f"record is complete ({scope}): every played game has a pre-kickoff "
              "prediction and a grade")
        return 0
    print(f"{len(found)} problem(s) with the record:")
    for p in found:
        print(f"  - {p}")
    return 0 if args.warn_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
