"""Feature construction details that are easy to get silently wrong."""

import nflreadpy as nfl
import polars as pl
import pytest

from nfl_predict.data.features import TEAM_CODE_ALIASES, canonical_team
from nfl_predict.data.games import load_games


def test_canonical_team_maps_relocated_franchises():
    df = pl.DataFrame({"team": ["SD", "STL", "OAK", "KC"]})
    assert df.select(canonical_team())["team"].to_list() == ["LAC", "LA", "LV", "KC"]


@pytest.mark.network
def test_no_schedule_team_code_is_missing_from_team_stats():
    """Regression: an unmapped code yields null EPA silently rather than raising.

    This originally cost every San Diego, St. Louis and Oakland game its EPA features (11.5% of
    completed games had null offensive form). If nflverse renames another franchise, this fails
    here instead of quietly degrading the model.
    """
    games = load_games().filter(pl.col("is_played"))
    scheduled = set(games["home_team"].to_list()) | set(games["away_team"].to_list())
    stats_teams = set(
        nfl.load_team_stats(seasons=list(range(2002, 2026)), summary_level="week")["team"].to_list()
    )
    unmapped = {t for t in scheduled if TEAM_CODE_ALIASES.get(t, t) not in stats_teams}
    assert not unmapped, f"team codes with no stats counterpart: {sorted(unmapped)}"


def test_qb_draft_score_maps_first_overall_to_one_and_undrafted_to_zero(monkeypatch):
    from datetime import UTC, datetime

    from nfl_predict.data import features as F

    monkeypatch.setattr(F, "_qb_draft_scores", lambda: pl.DataFrame(
        {"player_id": ["FIRST", "LATE"], "qb_draft": [1.0, 1 / 256]}))
    games = pl.DataFrame({
        "game_id": ["g1"], "home_qb_id": ["FIRST"], "away_qb_id": ["UDFA"],
        "kickoff_utc": [datetime(2025, 9, 7, tzinfo=UTC)],
    })
    row = F.qb_draft_features(games).row(0, named=True)
    assert row["home_qb_draft"] == 1.0
    assert row["away_qb_draft"] == 0.0


class TestFeedsThatAreNotReadyYet:
    """A season's data arrives in pieces. Every loader must treat "not there yet" as empty.

    The failure this guards against is specific and was caught before it happened: the moment
    the first game of a new season is marked played, that season joins the list passed to the
    loaders — and `load_pbp` raises ValueError ("Season must be between 1999 and N") until
    nflverse publishes its release, which would have crashed the whole prediction pass.
    """

    @staticmethod
    def _boom(exc):
        def raise_it(*a, **k):
            raise exc
        return raise_it

    @pytest.mark.parametrize("exc", [
        ConnectionError("404"), OSError("unreadable"), ValueError("Season must be between 1999 and 2025"),
    ])
    def test_pbp_loader_survives(self, monkeypatch, exc):
        import nflreadpy

        from nfl_predict.data import features as F
        monkeypatch.setattr(nflreadpy, "load_pbp", self._boom(exc))
        out = F._team_game_pbp([2026])
        assert out.height == 0 and "epa_noto" in out.columns

    @pytest.mark.parametrize("exc", [
        ConnectionError("404"), OSError("unreadable"), ValueError("Season must be between 1999 and 2025"),
    ])
    def test_team_stats_and_qb_loaders_survive(self, monkeypatch, exc):
        import nflreadpy

        from nfl_predict.data import features as F
        monkeypatch.setattr(nflreadpy, "load_team_stats", self._boom(exc))
        monkeypatch.setattr(nflreadpy, "load_player_stats", self._boom(exc))
        assert F._team_game_epa([2026]).height == 0
        assert F._qb_game_log([2026]).height == 0

    def test_a_partly_published_season_still_builds_features(self, monkeypatch):
        """Scores are in the schedule but the pbp release has not landed: Elo and margin form
        update from the score, the pbp-derived features simply hold their last value."""
        import nflreadpy

        from nfl_predict.data import features as F
        monkeypatch.setattr(nflreadpy, "load_pbp", self._boom(ValueError("not yet")))
        games = load_games()
        just_played = ["2026_01_NE_SEA", "2026_01_SF_LA"]
        sim = games.with_columns(
            pl.when(pl.col("game_id").is_in(just_played)).then(True).otherwise(pl.col("is_played")).alias("is_played"),
            pl.when(pl.col("game_id").is_in(just_played)).then(27).otherwise(pl.col("home_score")).alias("home_score"),
            pl.when(pl.col("game_id").is_in(just_played)).then(20).otherwise(pl.col("away_score")).alias("away_score"),
        )
        assert F.elo_ratings(sim).height > 6499        # the new results are rated
        assert F.pbp_form(sim).height > 0              # and pbp form still builds
