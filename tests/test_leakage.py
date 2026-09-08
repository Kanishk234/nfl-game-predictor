"""The gate: no feature may use information not knowable before its own game's kickoff.

Two layers here. The synthetic tests pin the *mechanism* - they construct the exact situations
that leak (simultaneous kickoffs, a team's own result folded into its form) and assert the
pipeline refuses them. The network-marked tests then run the same assertion over the whole real
2002-present frame.

Synthetic first on purpose: a green run over real data proves only that no leak happened to
occur, while these prove the guard actually fires when one does.
"""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nfl_predict.data.features import ELO_START, elo_ratings, rolling_form
from nfl_predict.data.pipeline import (
    BASELINE_COLUMNS,
    FEATURE_COLUMNS,
    LeakageError,
    assert_no_leakage,
    build_frame,
)
from nfl_predict.data.schedule import GAME_DURATION

pytestmark = pytest.mark.leakage

T0 = datetime(2025, 9, 7, 17, 0, tzinfo=UTC)

_GAMES_SCHEMA = {
    "game_id": pl.String,
    "season": pl.Int32,
    "week": pl.Int32,
    "kickoff_utc": pl.Datetime("us", "UTC"),
    "home_team": pl.String,
    "away_team": pl.String,
    "home_score": pl.Int32,
    "away_score": pl.Int32,
    "is_played": pl.Boolean,
}


def _games(rows):
    return pl.DataFrame(rows, schema=_GAMES_SCHEMA)


def _game(gid, kickoff, home, away, hs, as_, week=1):
    return {
        "game_id": gid, "season": 2025, "week": week, "kickoff_utc": kickoff,
        "home_team": home, "away_team": away, "home_score": hs, "away_score": as_,
        "is_played": True,
    }


class TestEloCannotSeeSimultaneousGames:
    """The subtle leak: Elo is league-wide, so a naive walk lets 1:00pm games read each other."""

    def test_simultaneous_games_start_from_the_same_ratings(self):
        elo = elo_ratings(_games([
            _game("g1", T0, "AAA", "BBB", 35, 0),
            _game("g2", T0, "CCC", "DDD", 21, 20),
        ])).sort("game_id")
        # Neither game had any concluded result available to it.
        assert elo["elo_as_of_utc"].null_count() == 2
        assert elo["home_elo_pre"].to_list() == [ELO_START, ELO_START]
        assert elo["away_elo_pre"].to_list() == [ELO_START, ELO_START]

    def test_a_later_game_does_absorb_the_earlier_result(self):
        """The mirror of the above - the guard must not simply refuse to ever learn."""
        elo = elo_ratings(_games([
            _game("g1", T0, "AAA", "BBB", 35, 0),
            _game("g2", T0 + timedelta(hours=5), "AAA", "CCC", 14, 10),
        ])).sort("game_id")
        later = elo.filter(pl.col("game_id") == "g2").row(0, named=True)
        assert later["elo_as_of_utc"] == T0 + GAME_DURATION
        assert later["home_elo_pre"] > ELO_START  # AAA's blowout win is now priced in

    def test_a_game_kicking_off_exactly_at_completion_does_not_absorb_it(self):
        """Strictness matters: a 4:15pm ET game 'completes' exactly at the 8:15pm kickoff."""
        elo = elo_ratings(_games([
            _game("early", T0, "AAA", "BBB", 35, 0),
            _game("night", T0 + GAME_DURATION, "AAA", "CCC", 14, 10),
        ]))
        night = elo.filter(pl.col("game_id") == "night").row(0, named=True)
        assert night["elo_as_of_utc"] is None
        assert night["home_elo_pre"] == ELO_START


class TestRollingFormIsBackwardLooking:
    def test_a_teams_own_result_is_not_in_its_form(self):
        form = rolling_form(_games([
            _game("g1", T0, "AAA", "BBB", 40, 0),
            _game("g2", T0 + timedelta(days=7), "AAA", "CCC", 0, 3, week=2),
        ]))
        first = form.filter((pl.col("game_id") == "g1") & (pl.col("team") == "AAA")).row(0, named=True)
        second = form.filter((pl.col("game_id") == "g2") & (pl.col("team") == "AAA")).row(0, named=True)
        # Nothing is known before a team's first game.
        assert first["margin_form"] is None
        assert first["form_as_of_utc"] is None
        # The second game sees the +40 from week 1, and not its own -3.
        assert second["margin_form"] == 40.0
        assert second["form_as_of_utc"] == T0 + GAME_DURATION

    def test_form_as_of_is_always_before_the_game_it_describes(self):
        games = _games([
            _game("g1", T0, "AAA", "BBB", 40, 0),
            _game("g2", T0 + timedelta(days=4), "CCC", "AAA", 7, 21, week=2),
            _game("g3", T0 + timedelta(days=11), "AAA", "DDD", 3, 6, week=3),
        ])
        form = rolling_form(games).join(
            games.select("game_id", "kickoff_utc"), on="game_id", how="left"
        )
        late = form.filter(
            pl.col("form_as_of_utc").is_not_null()
            & (pl.col("form_as_of_utc") >= pl.col("kickoff_utc"))
        )
        assert late.height == 0


class TestTheGuardItselfFires:
    """A gate that cannot fail is not a gate."""

    def test_assert_no_leakage_rejects_a_leaky_frame(self):
        leaky = pl.DataFrame({
            "game_id": ["bad"],
            "kickoff_utc": [T0],
            "as_of_utc": [T0 + timedelta(hours=1)],
        })
        with pytest.raises(LeakageError, match="not knowable before kickoff"):
            assert_no_leakage(leaky)

    def test_assert_no_leakage_rejects_as_of_exactly_at_kickoff(self):
        borderline = pl.DataFrame({
            "game_id": ["bad"], "kickoff_utc": [T0], "as_of_utc": [T0],
        })
        with pytest.raises(LeakageError):
            assert_no_leakage(borderline)

    def test_assert_no_leakage_accepts_a_clean_frame(self):
        clean = pl.DataFrame({
            "game_id": ["ok", "no-history"],
            "kickoff_utc": [T0, T0],
            "as_of_utc": [T0 - timedelta(seconds=1), None],
        })
        assert_no_leakage(clean)


def test_the_vegas_line_is_not_a_feature():
    """The baseline exists to be beaten, not learned from."""
    assert not set(BASELINE_COLUMNS) & set(FEATURE_COLUMNS)
    assert not any("spread" in c or "moneyline" in c or "total" in c for c in FEATURE_COLUMNS)


def test_targets_are_not_features():
    assert not any(c in FEATURE_COLUMNS for c in ("home_win", "margin", "result", "home_score"))


@pytest.mark.network
def test_real_frame_has_no_leakage():
    """The gate over the whole 2002-present dataset."""
    frame = build_frame()
    assert_no_leakage(frame)
    assert frame.height > 6000
