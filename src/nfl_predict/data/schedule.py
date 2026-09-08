"""Week targeting and the kickoff guard.

The cron schedule alone cannot be trusted to keep a prediction pass legal: NFL weeks do not
reliably open on Thursday (since 2002 there have been Wednesday, Friday, Saturday and Tuesday
regular-season games), and GitHub Actions' scheduled runs can be delayed under load. So the
guard here is mechanical — a pass computes the earliest kickoff in the week it covers and
refuses to publish if that kickoff has already passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import polars as pl

#: Conservative upper bound on wall-clock game length (regulation + overtime + stoppages).
#: A game's *result* is not knowable until roughly this long after its kickoff, so this is what
#: separates "a prior game we may learn from" from "a game still in progress".
GAME_DURATION = timedelta(hours=4)


class LateRunError(RuntimeError):
    """Raised when a prediction pass would publish at or after a kickoff it claims to cover."""


@dataclass(frozen=True)
class WeekTarget:
    """The (season, week) a pass covers, plus the deadline that makes it legal."""

    season: int
    week: int
    earliest_kickoff: datetime
    n_games: int

    def deadline_gap(self, now: datetime) -> timedelta:
        return self.earliest_kickoff - now


def utcnow() -> datetime:
    return datetime.now(UTC)


def next_week_target(games: pl.DataFrame, now: datetime | None = None) -> WeekTarget:
    """The next (season, week) with at least one kickoff still in the future.

    Weeks are targeted by kickoff time rather than by date arithmetic on the calendar, so a
    Wednesday opener, a Saturday slate or a rescheduled game all fall out correctly.
    """
    now = now or utcnow()
    upcoming = games.filter(pl.col("kickoff_utc") > now)
    if upcoming.is_empty():
        raise LateRunError(f"no games with a kickoff after {now.isoformat()}")

    first = upcoming.sort("kickoff_utc").row(0, named=True)
    season, week = first["season"], first["week"]
    in_week = games.filter((pl.col("season") == season) & (pl.col("week") == week))
    return WeekTarget(
        season=season,
        week=week,
        earliest_kickoff=in_week["kickoff_utc"].min(),
        n_games=in_week.height,
    )


def assert_before_kickoff(target: WeekTarget, now: datetime | None = None) -> None:
    """The gate. Refuse to proceed if the week's first game has already started."""
    now = now or utcnow()
    if now >= target.earliest_kickoff:
        raise LateRunError(
            f"pass for {target.season} week {target.week} would run at {now.isoformat()}, "
            f"at or after the week's first kickoff {target.earliest_kickoff.isoformat()} — "
            "a prediction published now is not valid (see CLAUDE.md, 'The gate')"
        )


def games_in_week(games: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    return games.filter((pl.col("season") == season) & (pl.col("week") == week)).sort("kickoff_utc")
