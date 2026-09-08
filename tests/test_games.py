"""Ingestion tests. The kickoff conversion is tested offline; coverage tests hit the network."""

from datetime import UTC, datetime

import polars as pl
import pytest

from nfl_predict.data.games import FIRST_SEASON, kickoff_utc_expr, load_games


def _kickoffs(rows):
    df = pl.DataFrame(rows, schema={"gameday": pl.String, "gametime": pl.String})
    return df.select(kickoff_utc_expr().alias("k"))["k"].to_list()


def test_kickoff_conversion_handles_dst_transition():
    """Same 13:00 ET kickoff is 17:00 UTC in September (EDT) but 18:00 UTC in December (EST).

    A fixed UTC offset would get one of these wrong by an hour, which is exactly the size of
    error that turns a valid pre-kickoff prediction into a late one.
    """
    sept, dec = _kickoffs([
        {"gameday": "2025-09-07", "gametime": "13:00"},
        {"gameday": "2025-12-14", "gametime": "13:00"},
    ])
    assert sept == datetime(2025, 9, 7, 17, 0, tzinfo=UTC)
    assert dec == datetime(2025, 12, 14, 18, 0, tzinfo=UTC)


def test_kickoff_conversion_rolls_late_games_to_next_utc_day():
    """A 20:20 ET Thursday kickoff is already Friday in UTC — the date must roll, not truncate."""
    (kick,) = _kickoffs([{"gameday": "2026-09-10", "gametime": "20:20"}])
    assert kick == datetime(2026, 9, 11, 0, 20, tzinfo=UTC)


@pytest.fixture(scope="module")
def games():
    return load_games()


@pytest.mark.network
class TestLoadedSchedule:
    def test_starts_at_first_season(self, games):
        assert games["season"].min() == FIRST_SEASON

    def test_every_game_has_a_kickoff(self, games):
        assert games["kickoff_utc"].null_count() == 0

    def test_kickoffs_are_utc(self, games):
        assert games.schema["kickoff_utc"].time_zone == "UTC"

    def test_game_ids_are_unique(self, games):
        assert games["game_id"].n_unique() == games.height

    def test_played_games_have_scores(self, games):
        played = games.filter(pl.col("is_played"))
        assert played["home_score"].null_count() == 0
        assert played["away_score"].null_count() == 0

    def test_vegas_spread_is_complete_for_completed_seasons(self, games):
        """The Phase 2 backtest baseline depends on this being gap-free."""
        completed = games.filter(pl.col("is_played") & (pl.col("season") < 2026))
        assert completed["spread_line"].null_count() == 0
