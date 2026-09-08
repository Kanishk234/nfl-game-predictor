"""What happens when the world misbehaves. An unattended season depends on these."""

import json
from datetime import UTC, datetime, timedelta
from typing import ClassVar

import polars as pl
import pytest

from nfl_predict import health as H
from nfl_predict import predict as P
from nfl_predict.odds import fetch as F
from nfl_predict.retry import with_retries

T0 = datetime(2026, 9, 10, 0, 20, tzinfo=UTC)


class TestRetries:
    def test_a_transient_failure_is_retried_then_succeeds(self):
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("502")
            return "ok"

        assert with_retries(flaky, what="thing", sleep=lambda s: None) == "ok"
        assert len(calls) == 3

    def test_it_gives_up_and_says_what_failed(self):
        with pytest.raises(RuntimeError, match="thing failed after 4 attempts"):
            with_retries(lambda: (_ for _ in ()).throw(ConnectionError("down")),
                         what="thing", sleep=lambda s: None)

    def test_a_permanent_failure_is_not_retried(self):
        calls = []

        def bad_key():
            calls.append(1)
            raise F.OddsPermanentError("401")

        with pytest.raises(F.OddsPermanentError):
            with_retries(bad_key, what="odds", retry_on=(F.OddsFetchError,),
                         give_up_on=(F.OddsPermanentError,), sleep=lambda s: None)
        assert len(calls) == 1, "a bad key must fail fast, not spend a minute proving it"

    @pytest.mark.parametrize(("status", "permanent"), [(401, True), (403, True), (429, True),
                                                       (500, False), (502, False)])
    def test_http_status_decides_transient_versus_permanent(self, monkeypatch, status, permanent):
        import requests

        class R:
            status_code = status
            text = "nope"
            headers: ClassVar[dict] = {}
        monkeypatch.setattr(requests, "get", lambda *a, **k: R())
        with pytest.raises(F.OddsFetchError) as e:
            F._request("key")
        assert isinstance(e.value, F.OddsPermanentError) is permanent


class TestOddsOutageDoesNotLoseThePrediction:
    """The Vegas line is the comparison, not the product."""

    def _frame(self):
        rows = [{"game_id": f"2025_01_G{i}", "season": 2025, "week": 1,
                 "kickoff_utc": T0 - timedelta(days=360 + i), "home_team": "AAA", "away_team": "BBB",
                 "is_played": True, "as_of_utc": None, "elo_diff": 50.0 * (i - 3), "week_": 1,
                 "home_win": int(i >= 3), "spread_line": 3.0 * (i - 3)} for i in range(6)]
        rows.append({"game_id": "2026_01_NE_SEA", "season": 2026, "week": 1, "kickoff_utc": T0,
                     "home_team": "SEA", "away_team": "NE", "is_played": False,
                     "as_of_utc": T0 - timedelta(days=200), "elo_diff": 10.0, "week_": 1,
                     "home_win": None, "spread_line": None})
        return pl.DataFrame(rows).drop("week_").with_columns(
            pl.col("kickoff_utc").cast(pl.Datetime("us", "UTC")),
            pl.col("as_of_utc").cast(pl.Datetime("us", "UTC")))

    def test_the_prediction_is_still_published_and_the_gap_is_recorded(self, monkeypatch, tmp_path):
        import numpy as np

        class Const:
            def predict_proba(self, X):
                return np.column_stack([np.full(len(X), 0.4), np.full(len(X), 0.6)])

            def predict(self, X):
                return np.full(len(X), 3.0)

        monkeypatch.setattr(P, "build_frame", self._frame)
        monkeypatch.setattr(P, "assert_no_leakage", lambda f: None)
        monkeypatch.setattr(P, "PROCESSED_PATH", tmp_path / "g.parquet")
        monkeypatch.setattr(P, "PREDICTIONS_DIR", tmp_path / "predictions")
        monkeypatch.setattr(P, "snapshot_path", lambda t, n: tmp_path / "odds" / f"{t.season}_{t.week:02d}_{n}.json")
        monkeypatch.setattr(P, "fit_final", lambda frame, spec: {
            "classifier": Const(), "regressor": Const(), "features": ["elo_diff"],
            "spec": {"learner": "linear", "params": {}}, "n_train": 6,
            "trained_through_game": "x", "trained_through_utc": "", "trained_at_utc": "",
            "code_version": "abc"})

        def dead(*a, **k):
            raise F.OddsFetchError("the odds provider is down")
        monkeypatch.setattr(P, "fetch_snapshot", dead)

        paths = P.run("early", now=T0 - timedelta(hours=5))
        assert paths is not None, "an odds outage must not cost us the prediction"
        rec = json.loads(paths[0].read_text())
        assert rec["n_games"] == 1
        assert rec["predictions"][0]["vegas"] is None      # shown as missing, never invented
        snap = json.loads(paths[1].read_text())
        assert "down" in snap["unavailable"]               # and the reason is on the record
        assert snap["fetched_at_utc"] == rec["generated_at_utc"]


class TestHealthCheck:
    def _games(self, played_through: int):
        rows = []
        for wk in (1, 2):
            for i in range(2):
                rows.append({"game_id": f"2026_{wk:02d}_G{i}", "season": 2026, "week": wk,
                             "kickoff_utc": T0 + timedelta(days=7 * (wk - 1), hours=i)})
        return pl.DataFrame(rows).with_columns(pl.col("kickoff_utc").cast(pl.Datetime("us", "UTC")))

    def test_a_season_we_never_ran_is_not_our_problem(self, monkeypatch, tmp_path):
        """Last season was played without us. That is not a hole in our record."""
        (tmp_path / "predictions").mkdir(); (tmp_path / "results").mkdir(); (tmp_path / "odds").mkdir()
        monkeypatch.setattr(H, "PREDICTIONS_DIR", tmp_path / "predictions")
        monkeypatch.setattr(H, "RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr("nfl_predict.grade.PREDICTIONS_DIR", tmp_path / "predictions")
        assert H.problems(self._games(1), now=T0 + timedelta(days=1)) == []

    def test_a_week_played_with_no_prediction_is_reported(self, monkeypatch, tmp_path):
        (tmp_path / "predictions").mkdir(); (tmp_path / "results").mkdir(); (tmp_path / "odds").mkdir()
        # we published week 1, so this season is ours to keep complete
        (tmp_path / "predictions" / "2026_01_early.json").write_text(json.dumps(
            {"season": 2026, "week": 1, "pass": "early", "generated_at_utc": (T0 - timedelta(days=1)).isoformat(),
             "predictions": []}))
        monkeypatch.setattr(H, "PREDICTIONS_DIR", tmp_path / "predictions")
        monkeypatch.setattr(H, "RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr("nfl_predict.grade.PREDICTIONS_DIR", tmp_path / "predictions")
        # far enough along that week 2 has been played, but only week 1 was ever published
        found = H.problems(self._games(1), now=T0 + timedelta(days=8))
        assert any("week 2" in p and "no prediction file" in p for p in found), found

    def test_a_game_that_never_got_a_pre_kickoff_pick_is_reported(self, monkeypatch, tmp_path):
        (tmp_path / "predictions").mkdir(); (tmp_path / "results").mkdir(); (tmp_path / "odds").mkdir()
        games = self._games(1)
        wk1 = games.filter(pl.col("week") == 1)
        # a pass covering only the first of week 1's two games
        (tmp_path / "predictions" / "2026_01_early.json").write_text(json.dumps(
            {"season": 2026, "week": 1, "pass": "early",
             "generated_at_utc": (T0 - timedelta(days=1)).isoformat(),
             "predictions": [{"game_id": wk1["game_id"][0], "kickoff_utc": wk1["kickoff_utc"][0].isoformat()}]}))
        monkeypatch.setattr(H, "PREDICTIONS_DIR", tmp_path / "predictions")
        monkeypatch.setattr(H, "RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr("nfl_predict.grade.PREDICTIONS_DIR", tmp_path / "predictions")
        found = H.problems(games, now=T0 + timedelta(hours=6))
        assert any("no pre-kickoff prediction" in p and wk1["game_id"][1] in p for p in found), found

    def test_a_clean_record_reports_nothing(self, monkeypatch, tmp_path):
        (tmp_path / "predictions").mkdir(); (tmp_path / "results").mkdir(); (tmp_path / "odds").mkdir()
        games = self._games(1)
        rec = {"season": 2026, "week": 1, "pass": "early",
               "generated_at_utc": (T0 - timedelta(days=1)).isoformat(),
               "predictions": [{"game_id": g, "kickoff_utc": k.isoformat()}
                               for g, k in games.filter(pl.col("week") == 1).select("game_id", "kickoff_utc").iter_rows()]}
        (tmp_path / "predictions" / "2026_01_early.json").write_text(json.dumps(rec))
        (tmp_path / "results" / "2026_01.json").write_text(json.dumps(
            {"games": [{"game_id": g} for g in games.filter(pl.col("week") == 1)["game_id"]]}))
        monkeypatch.setattr(H, "PREDICTIONS_DIR", tmp_path / "predictions")
        monkeypatch.setattr(H, "RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr("nfl_predict.grade.PREDICTIONS_DIR", tmp_path / "predictions")
        assert H.problems(games, now=T0 + timedelta(hours=6)) == []


class TestQuotaWarning:
    def _dirs(self, monkeypatch, tmp_path):
        for d in ("predictions", "results", "odds"):
            (tmp_path / d).mkdir()
        (tmp_path / "predictions" / "2026_01_early.json").write_text(json.dumps(
            {"season": 2026, "week": 1, "pass": "early",
             "generated_at_utc": (T0 - timedelta(days=1)).isoformat(), "predictions": []}))
        monkeypatch.setattr(H, "PREDICTIONS_DIR", tmp_path / "predictions")
        monkeypatch.setattr(H, "RESULTS_DIR", tmp_path / "results")
        monkeypatch.setattr("nfl_predict.grade.PREDICTIONS_DIR", tmp_path / "predictions")

    def _snapshot(self, tmp_path, name, remaining):
        (tmp_path / "odds" / name).write_text(json.dumps(
            {"source": {"quota": {"requests_remaining": remaining}}, "lines": []}))

    def _games(self):
        return pl.DataFrame({"game_id": ["2026_01_G0"], "season": [2026], "week": [1],
                             "kickoff_utc": [T0]}).with_columns(
            pl.col("kickoff_utc").cast(pl.Datetime("us", "UTC")))

    def test_a_healthy_quota_says_nothing(self, monkeypatch, tmp_path):
        self._dirs(monkeypatch, tmp_path)
        self._snapshot(tmp_path, "2026_01_early.json", "480")
        assert not any("quota" in p for p in H.problems(self._games(), now=T0 - timedelta(hours=1)))

    def test_a_low_quota_is_reported(self, monkeypatch, tmp_path):
        self._dirs(monkeypatch, tmp_path)
        self._snapshot(tmp_path, "2026_01_early.json", "42")
        found = H.problems(self._games(), now=T0 - timedelta(hours=1))
        assert any("quota down to 42" in p for p in found), found

    def test_the_most_recent_snapshot_wins(self, monkeypatch, tmp_path):
        """An old low reading must not outlive a renewed quota."""
        self._dirs(monkeypatch, tmp_path)
        self._snapshot(tmp_path, "2026_01_early.json", "12")
        self._snapshot(tmp_path, "2026_02_early.json", "495")
        assert not any("quota" in p for p in H.problems(self._games(), now=T0 - timedelta(hours=1)))
