"""Game-level ingestion from nflreadpy, normalized to a UTC kickoff timestamp.

This is the spine of the whole project: every feature's "as-of" time is checked against
`kickoff_utc` from this table, so the conversion here has to be exactly right. nflreadpy's
schedules give `gameday` (date) and `gametime` (HH:MM) as *local Eastern* strings with no
timezone attached; we attach America/New_York and convert, which handles the mid-season
EDT->EST transition that a fixed offset would silently get wrong by an hour.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import nflreadpy as nfl
import polars as pl

#: First season of the current 32-team, 8-division era. See CLAUDE.md — this bound is a
#: deliberate modeling choice, not a data-availability limit.
FIRST_SEASON = 2002

EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

#: Columns kept from the schedules table. `spread_line`/`total_line`/moneylines are the
#: historical closing market, used as the backtest baseline in Phase 2 — they are NOT features.
_COLUMNS = [
    "game_id", "season", "game_type", "week",
    "gameday", "weekday", "gametime",
    "away_team", "home_team", "away_score", "home_score", "result", "total",
    "location", "roof", "surface", "div_game", "overtime",
    "away_rest", "home_rest",
    "spread_line", "total_line", "away_moneyline", "home_moneyline",
]


def load_games(first_season: int = FIRST_SEASON) -> pl.DataFrame:
    """Load all games from `first_season` onward with a UTC kickoff timestamp.

    Includes not-yet-played games (`is_played` is False for those) — those are the rows the
    weekly prediction pass runs against.
    """
    games = nfl.load_schedules().filter(pl.col("season") >= first_season).select(_COLUMNS)

    missing_time = games.filter(pl.col("gameday").is_null() | pl.col("gametime").is_null())
    if missing_time.height:
        raise ValueError(
            f"{missing_time.height} game(s) from {first_season}+ lack a date or kickoff time; "
            "kickoff_utc cannot be derived and the leakage gate would be unenforceable: "
            f"{missing_time['game_id'].to_list()[:10]}"
        )

    return (
        games.with_columns(
            kickoff_utc_expr().alias("kickoff_utc"),
            pl.col("result").is_not_null().alias("is_played"),
        )
        .sort("kickoff_utc", "game_id")
    )


def kickoff_utc_expr() -> pl.Expr:
    """Expression turning nflreadpy's local-Eastern `gameday` + `gametime` into UTC.

    Split out from `load_games` so the timezone conversion can be tested on synthetic rows
    without a network round-trip.
    """
    return (
        pl.concat_str([pl.col("gameday"), pl.lit(" "), pl.col("gametime")])
        .str.to_datetime(format="%Y-%m-%d %H:%M", time_unit="us")
        .dt.replace_time_zone(str(EASTERN))
        .dt.convert_time_zone(str(UTC))
    )


def completed_games(games: pl.DataFrame) -> pl.DataFrame:
    """Only games with a final result — the rows eligible for training and grading."""
    return games.filter(pl.col("is_played"))
