"""Odds fetch: parsing, matching, immutability, and that the key can never leak."""

import json
from datetime import UTC, datetime
from typing import ClassVar

import polars as pl
import pytest

from nfl_predict.odds import fetch as F

KEY = "sk-test-DO-NOT-LEAK-1234567890"

EVENT = {
    "id": "abc", "commence_time": "2026-09-10T00:20:00Z",
    "home_team": "Seattle Seahawks", "away_team": "New England Patriots",
    "bookmakers": [
        {"key": "draftkings", "markets": [
            {"key": "h2h", "last_update": "2026-09-08T17:05:44Z", "outcomes": [
                {"name": "New England Patriots", "price": 142}, {"name": "Seattle Seahawks", "price": -170}]},
            {"key": "spreads", "last_update": "2026-09-08T17:05:44Z", "outcomes": [
                {"name": "New England Patriots", "price": -102, "point": 3.0},
                {"name": "Seattle Seahawks", "price": -118, "point": -3.0}]},
            {"key": "totals", "last_update": "2026-09-08T17:05:44Z", "outcomes": [
                {"name": "Over", "price": -108, "point": 44.5}, {"name": "Under", "price": -112, "point": 44.5}]},
        ]},
        {"key": "fanduel", "markets": [
            {"key": "spreads", "last_update": "2026-09-08T17:00:00Z", "outcomes": [
                {"name": "New England Patriots", "price": -110, "point": 3.5},
                {"name": "Seattle Seahawks", "price": -110, "point": -3.5}]},
        ]},
    ],
}
ABBR = {"Seattle Seahawks": "SEA", "New England Patriots": "NE"}


class TestParsing:
    def test_spread_sign_matches_nflreadpy_convention(self):
        """Home favoured by 3 -> spread_line +3, like `spread_line` in the schedule."""
        p = F._parse_event(EVENT, ABBR)
        assert p["books"]["draftkings"]["spread_home"] == -3.0
        assert p["consensus"]["spread_line"] == 3.25  # median of 3.0 and 3.5, sign flipped
        assert p["consensus"]["total_line"] == 44.5
        assert p["consensus"]["n_books"] == 2

    def test_moneyline_devigged(self):
        p = F._parse_event(EVENT, ABBR)
        raw_home, raw_away = F.american_to_prob(-170), F.american_to_prob(142)
        assert raw_home + raw_away > 1.0  # vig present
        assert p["consensus"]["p_home_moneyline"] == pytest.approx(raw_home / (raw_home + raw_away))
        assert 0.6 < p["consensus"]["p_home_moneyline"] < 0.65

    def test_unknown_team_name_is_skipped_not_guessed(self):
        assert F._parse_event({**EVENT, "home_team": "Seattle Sea Hawks"}, ABBR) is None

    def test_american_to_prob(self):
        assert F.american_to_prob(100) == 0.5
        assert F.american_to_prob(-100) == 0.5
        assert F.american_to_prob(-200) == pytest.approx(2 / 3)
        assert F.american_to_prob(200) == pytest.approx(1 / 3)


@pytest.mark.network
def test_team_name_map_is_unambiguous_for_schedule_codes():
    """Regression: the Rams are listed as LA and LAR upstream; only the schedule's code survives."""
    m = F.team_name_map({"LA", "SF", "KC"})
    assert m["Los Angeles Rams"] == "LA"
    assert set(m.values()) == {"LA", "SF", "KC"}


class TestMatching:
    def _week(self):
        return pl.DataFrame({
            "game_id": ["2026_01_NE_SEA", "2026_01_SF_LA"],
            "home_team": ["SEA", "LA"], "away_team": ["NE", "SF"],
            "kickoff_utc": [datetime(2026, 9, 10, 0, 20, tzinfo=UTC), datetime(2026, 9, 11, 0, 35, tzinfo=UTC)],
        })

    def test_matches_on_teams_and_kickoff(self):
        lines = F.match_to_schedule([F._parse_event(EVENT, ABBR)], self._week())
        assert [r["game_id"] for r in lines] == ["2026_01_NE_SEA"]

    def test_wrong_week_same_teams_is_not_matched(self):
        far = F._parse_event({**EVENT, "commence_time": "2026-12-10T00:20:00Z"}, ABBR)
        assert F.match_to_schedule([far], self._week()) == []


class TestSecrets:
    def test_scrub_removes_key(self):
        assert KEY not in F.scrub(f"GET ...?apiKey={KEY}&regions=us failed", KEY)

    def test_failed_request_never_raises_the_key(self, monkeypatch):
        import requests

        def boom(*a, **k):
            raise requests.ConnectionError(f"https://x/?apiKey={KEY} unreachable")
        monkeypatch.setattr(requests, "get", boom)
        with pytest.raises(F.OddsFetchError) as e:
            F._request(KEY)
        assert KEY not in str(e.value)

    def test_non_200_never_raises_the_key(self, monkeypatch):
        import requests

        class R:
            status_code = 401
            text = f"Unauthorized for key {KEY}"
            headers: ClassVar[dict] = {}
        monkeypatch.setattr(requests, "get", lambda *a, **k: R())
        with pytest.raises(F.OddsFetchError) as e:
            F._request(KEY)
        assert KEY not in str(e.value)

    def test_snapshot_never_contains_request_parameters(self, tmp_path):
        with pytest.raises(F.OddsFetchError):
            F.write_snapshot({"source": {"apiKey": KEY}}, tmp_path / "x.json")

    def test_missing_key_is_a_clear_error(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ODDS_API_KEY", raising=False)
        monkeypatch.setattr(F, "_ENV_FILE", tmp_path / "nope")
        with pytest.raises(F.OddsFetchError, match="ODDS_API_KEY is not set"):
            F.load_api_key()

    def test_env_file_is_gitignored(self):
        from pathlib import Path
        assert ".env" in Path(".gitignore").read_text().splitlines()


class TestImmutability:
    def test_second_write_to_same_path_is_refused(self, tmp_path):
        p = tmp_path / "2026_01_early.json"
        F.write_snapshot({"lines": []}, p)
        with pytest.raises(F.SnapshotExistsError):
            F.write_snapshot({"lines": [1]}, p)
        assert json.loads(p.read_text()) == {"lines": []}
