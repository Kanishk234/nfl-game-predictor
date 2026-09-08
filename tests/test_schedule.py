"""Week targeting and the kickoff guard."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nfl_predict.data.schedule import (
    LateRunError,
    assert_before_kickoff,
    games_in_week,
    next_week_target,
)


def _games(rows):
    return pl.DataFrame(
        rows, schema={"game_id": pl.String, "season": pl.Int32, "week": pl.Int32,
                      "kickoff_utc": pl.Datetime("us", "UTC")},
    )


def _week1_with_wednesday_opener():
    """2026 Week 1's real shape: a Wednesday opener, then Thursday, then the Sunday slate."""
    return _games([
        {"game_id": "2026_01_NE_SEA", "season": 2026, "week": 1,
         "kickoff_utc": datetime(2026, 9, 10, 0, 20, tzinfo=UTC)},
        {"game_id": "2026_01_SF_LA", "season": 2026, "week": 1,
         "kickoff_utc": datetime(2026, 9, 11, 0, 35, tzinfo=UTC)},
        {"game_id": "2026_01_ATL_PIT", "season": 2026, "week": 1,
         "kickoff_utc": datetime(2026, 9, 13, 17, 0, tzinfo=UTC)},
        {"game_id": "2026_02_AAA_BBB", "season": 2026, "week": 2,
         "kickoff_utc": datetime(2026, 9, 20, 17, 0, tzinfo=UTC)},
    ])


def test_target_is_the_week_of_the_next_kickoff():
    target = next_week_target(_week1_with_wednesday_opener(), datetime(2026, 9, 8, tzinfo=UTC))
    assert (target.season, target.week, target.n_games) == (2026, 1, 3)
    assert target.earliest_kickoff == datetime(2026, 9, 10, 0, 20, tzinfo=UTC)


def test_earliest_kickoff_is_the_wednesday_game_not_the_thursday_one():
    """The deadline is the week's *first* game, whatever weekday it lands on."""
    target = next_week_target(_week1_with_wednesday_opener(), datetime(2026, 9, 8, tzinfo=UTC))
    assert target.earliest_kickoff.weekday() == 3  # Wed 8:20pm ET is already Thu in UTC
    assert target.earliest_kickoff < datetime(2026, 9, 11, 0, 35, tzinfo=UTC)


def test_thursday_run_after_a_wednesday_opener_targets_the_rest_of_the_week():
    """The gate is per game: Wednesday's game is out of reach, Thursday's and Sunday's are not."""
    now = datetime(2026, 9, 10, 21, 0, tzinfo=UTC)
    target = next_week_target(_week1_with_wednesday_opener(), now)
    assert (target.season, target.week) == (2026, 1)
    assert target.earliest_kickoff == datetime(2026, 9, 11, 0, 35, tzinfo=UTC)  # Thursday, not Wednesday
    assert_before_kickoff(target, now)


def test_guard_rejects_a_run_after_the_whole_week_has_kicked_off():
    target = next_week_target(_week1_with_wednesday_opener(), datetime(2026, 9, 8, tzinfo=UTC))
    with pytest.raises(LateRunError, match="no game left"):
        assert_before_kickoff(target, datetime(2026, 9, 13, 17, 0, tzinfo=UTC))


def test_guard_accepts_the_tuesday_cron():
    target = next_week_target(_week1_with_wednesday_opener(), datetime(2026, 9, 8, tzinfo=UTC))
    assert_before_kickoff(target, datetime(2026, 9, 8, 16, 0, tzinfo=UTC))


def test_guard_rejects_a_run_exactly_at_the_last_kickoff():
    """'Before kickoff' is strict — equal timestamps are late, not on time."""
    target = next_week_target(_week1_with_wednesday_opener(), datetime(2026, 9, 8, tzinfo=UTC))
    with pytest.raises(LateRunError):
        assert_before_kickoff(target, target.latest_kickoff)


def test_target_rolls_to_next_week_once_every_game_has_kicked_off():
    target = next_week_target(_week1_with_wednesday_opener(), datetime(2026, 9, 14, tzinfo=UTC))
    assert (target.season, target.week) == (2026, 2)


def test_no_future_games_raises():
    with pytest.raises(LateRunError, match="no games"):
        next_week_target(_week1_with_wednesday_opener(), datetime(2027, 1, 1, tzinfo=UTC))


def test_deadline_gap():
    target = next_week_target(_week1_with_wednesday_opener(), datetime(2026, 9, 8, tzinfo=UTC))
    assert target.deadline_gap(datetime(2026, 9, 9, 0, 20, tzinfo=UTC)) == timedelta(days=1)


def test_games_in_week_is_sorted_by_kickoff():
    week = games_in_week(_week1_with_wednesday_opener(), 2026, 1)
    assert week["game_id"].to_list() == ["2026_01_NE_SEA", "2026_01_SF_LA", "2026_01_ATL_PIT"]


class TestSafetyNetTiming:
    """`next_scheduled_early_pass` decides whether the Tuesday run should stay out of the way."""

    def test_finds_the_coming_thursday_from_a_tuesday(self):
        from nfl_predict.data.schedule import next_scheduled_early_pass
        tue = datetime(2026, 11, 24, 16, 0, tzinfo=UTC)
        assert next_scheduled_early_pass(tue) == datetime(2026, 11, 26, 21, 0, tzinfo=UTC)

    def test_after_thursdays_pass_it_rolls_to_the_next_week(self):
        from nfl_predict.data.schedule import next_scheduled_early_pass
        just_after = datetime(2026, 11, 26, 21, 30, tzinfo=UTC)
        assert next_scheduled_early_pass(just_after) == datetime(2026, 12, 3, 21, 0, tzinfo=UTC)

    def test_thanksgiving_week_opens_before_the_thursday_pass(self):
        """The case this exists for: 2026 week 12 opens Wed 8pm ET, and Thanksgiving's first
        game kicks at 12:30pm ET — both before Thursday 21:00 UTC."""
        from nfl_predict.data.schedule import next_scheduled_early_pass
        tue = datetime(2026, 11, 24, 16, 0, tzinfo=UTC)
        week_opens = datetime(2026, 11, 26, 1, 0, tzinfo=UTC)   # Wed Nov 25, 8:00 pm ET
        assert week_opens < next_scheduled_early_pass(tue)      # so Tuesday must publish

    def test_a_normal_week_opens_after_the_thursday_pass(self):
        from nfl_predict.data.schedule import next_scheduled_early_pass
        tue = datetime(2026, 9, 15, 16, 0, tzinfo=UTC)
        week_opens = datetime(2026, 9, 18, 0, 15, tzinfo=UTC)   # Thu Sep 17, 8:15 pm ET
        assert week_opens > next_scheduled_early_pass(tue)      # so Tuesday stays out of the way
