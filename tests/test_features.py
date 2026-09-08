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
