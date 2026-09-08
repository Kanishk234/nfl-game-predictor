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
