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


def operating_seasons() -> set[int]:
    """Seasons we have actually published for. A season we never ran is not a hole in our
    record — it is simply not our record."""
    return {int(p.name[:4]) for p in PREDICTIONS_DIR.glob("*_*_*.json")}


def problems(games: pl.DataFrame, now: datetime) -> list[str]:
    found: list[str] = []
    seasons = operating_seasons()
    if not seasons:
        return found
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

        # 3. games finished long enough ago that they should have been graded by now
        path = RESULTS_DIR / f"{season}_{week:02d}.json"
        graded = {g["game_id"] for g in json.loads(path.read_text())["games"]} if path.exists() else set()
        overdue = sorted(
            g["game_id"] for g in started.iter_rows(named=True)
            if g["game_id"] not in graded and g["kickoff_utc"] + GRADING_GRACE < now
        )
        if overdue:
            found.append(f"{season} week {week}: finished but ungraded after "
                         f"{GRADING_GRACE.days} days: {overdue}")

    # 4. an odds snapshot that recorded a provider failure, so the gap is explained not silent
    for path in sorted(PREDICTIONS_DIR.parent.glob("odds/*.json")):
        snap = json.loads(path.read_text())
        if snap.get("unavailable"):
            found.append(f"{path.name}: published without a Vegas baseline ({snap['unavailable'][:80]})")
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
