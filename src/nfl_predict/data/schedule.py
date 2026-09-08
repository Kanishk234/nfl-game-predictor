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

#: When the regular early pass runs: Thursday 21:00 UTC (see .github/workflows/predict-early.yml).
#: The Tuesday safety-net run uses this to decide whether a week opens too early for it.
EARLY_PASS_WEEKDAY, EARLY_PASS_HOUR = 3, 21

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
    latest_kickoff: datetime
    n_games: int

    def deadline_gap(self, now: datetime) -> timedelta:
        """Time until the next kickoff the pass still has to beat."""
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
        # The earliest kickoff *still ahead*: a Thursday pass after a Wednesday opener is
        # measured against Thursday's game, not the one already played.
        earliest_kickoff=first["kickoff_utc"],
        latest_kickoff=in_week["kickoff_utc"].max(),
        n_games=in_week.height,
    )


def assert_before_kickoff(target: WeekTarget, now: datetime | None = None) -> None:
    """The gate. Refuse to proceed if there is no game left in the week to predict.

    The gate is per game: a pass only ever predicts games whose kickoff is still ahead
    (`predict.games_to_predict`), and the grader ignores any prediction generated after its
    game's kickoff. So a pass run after some of the week's games have started is fine for the
    rest of the week; a pass run after all of them have started has nothing valid to publish.
    """
    now = now or utcnow()
    if now >= target.latest_kickoff:
        raise LateRunError(
            f"pass for {target.season} week {target.week} would run at {now.isoformat()}, "
            f"at or after the week's last kickoff {target.latest_kickoff.isoformat()} — "
            "there is no game left to predict (see CLAUDE.md, 'The gate')"
        )


def next_scheduled_early_pass(now: datetime) -> datetime:
    """The next Thursday 21:00 UTC at or after `now`.

    Used by the Tuesday safety-net run: if the target week's first kickoff is later than this,
    the regular Thursday pass will cover the whole week and Tuesday should stay out of the way.
    """
    d = now.replace(hour=EARLY_PASS_HOUR, minute=0, second=0, microsecond=0)
    if d < now:
        d += timedelta(days=1)
    while d.weekday() != EARLY_PASS_WEEKDAY:
        d += timedelta(days=1)
    return d


def games_in_week(games: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    return games.filter((pl.col("season") == season) & (pl.col("week") == week)).sort("kickoff_utc")
