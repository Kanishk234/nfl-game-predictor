"""The prediction pass: what it predicts, what it writes, and what it refuses to do."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from nfl_predict import predict as P
from nfl_predict.data.schedule import WeekTarget

T0 = datetime(2026, 9, 10, 0, 20, tzinfo=UTC)


class _Const:
    def __init__(self, v):
        self.v = v

    def predict_proba(self, X):
        return np.column_stack([1 - np.full(len(X), self.v), np.full(len(X), self.v)])

    def predict(self, X):
        return np.full(len(X), self.v * 10)


def _bundle():
    return {
        "classifier": _Const(0.62), "regressor": _Const(0.62), "features": ["elo_diff", "week"],
        "spec": {"learner": "linear", "params": {"C": 0.1}}, "n_train": 10,
        "trained_through_game": "2025_22_SEA_NE", "trained_through_utc": "2026-02-08T23:30:00+00:00",
        "trained_at_utc": "2026-09-08T00:00:00+00:00", "code_version": "abc123",
    }


def _frame():
    rows = []
    for i in range(6):  # six played games to fit the spread->prob curve on
        rows.append({"game_id": f"2025_01_G{i}", "season": 2025, "week": 1, "kickoff_utc": T0 - timedelta(days=360 + i),
                     "home_team": "AAA", "away_team": "BBB", "is_played": True, "as_of_utc": None,
                     "elo_diff": 50.0 * (i - 3), "home_win": int(i >= 3), "spread_line": 3.0 * (i - 3)})
    for gid, k in (("2026_01_NE_SEA", T0), ("2026_01_SF_LA", T0 + timedelta(days=1)), ("2026_01_ATL_PIT", T0 + timedelta(days=3))):
        rows.append({"game_id": gid, "season": 2026, "week": 1, "kickoff_utc": k, "home_team": gid[-3:].strip("_"),
                     "away_team": gid.split("_")[2], "is_played": False, "as_of_utc": T0 - timedelta(days=200),
                     "elo_diff": 10.0, "home_win": None, "spread_line": None})
    return pl.DataFrame(rows).with_columns(pl.col("kickoff_utc").cast(pl.Datetime("us", "UTC")), pl.col("as_of_utc").cast(pl.Datetime("us", "UTC")))


def _odds():
    return {"lines": [
        {"game_id": "2026_01_NE_SEA", "consensus": {"spread_line": 3.0, "total_line": 44.5, "p_home_moneyline": 0.618, "n_books": 3}},
        {"game_id": "2026_01_SF_LA", "consensus": {"spread_line": -1.5, "total_line": 47.0, "p_home_moneyline": 0.46, "n_books": 3}},
    ]}


def _target():
    return WeekTarget(season=2026, week=1, earliest_kickoff=T0, latest_kickoff=T0 + timedelta(days=3), n_games=3)


class TestWhatGetsPredicted:
    def test_early_pass_gets_the_whole_week(self):
        rows = P.games_to_predict(_frame(), _target(), now=T0 - timedelta(days=1))
        assert rows["game_id"].to_list() == ["2026_01_NE_SEA", "2026_01_SF_LA", "2026_01_ATL_PIT"]

    def test_late_pass_excludes_games_that_already_kicked_off(self):
        rows = P.games_to_predict(_frame(), _target(), now=T0 + timedelta(hours=2))
        assert rows["game_id"].to_list() == ["2026_01_SF_LA", "2026_01_ATL_PIT"]

    def test_a_game_at_exactly_kickoff_is_not_predicted(self):
        rows = P.games_to_predict(_frame(), _target(), now=T0)
        assert "2026_01_NE_SEA" not in rows["game_id"].to_list()


class TestRecord:
    def test_prediction_rows_carry_both_baselines_and_provenance(self):
        frame = _frame(); rows = P.games_to_predict(frame, _target(), now=T0 - timedelta(days=1))
        preds = P.make_predictions(_bundle(), frame, rows, _odds())
        sea = next(p for p in preds if p["game_id"] == "2026_01_NE_SEA")
        assert sea["p_home"] == 0.62 and sea["pred_margin"] == 6.2 and sea["pick"] == "SEA"
        assert sea["vegas"]["spread_line"] == 3.0
        assert sea["vegas"]["p_home_moneyline"] == 0.618
        assert 0.5 < sea["vegas"]["p_home_from_spread"] < 1.0  # positive spread -> home favoured
        pit = next(p for p in preds if p["game_id"] == "2026_01_ATL_PIT")
        assert pit["vegas"] is None  # no line: recorded as missing, not invented
        rec = P.build_record(_target(), "early", T0 - timedelta(hours=30), _bundle(), preds)
        assert rec["gate"]["seconds_before_first_kickoff"] == 30 * 3600
        assert rec["model"]["trained_through_game"] == "2025_22_SEA_NE"
        assert rec["model"]["code_version"] == "abc123"
        assert rec["n_games"] == 3

    def test_record_is_immutable(self, tmp_path):
        p = tmp_path / "2026_01_early.json"
        P.write_record({"a": 1}, p)
        with pytest.raises(P.PredictionExistsError):
            P.write_record({"a": 2}, p)
        assert json.loads(p.read_text()) == {"a": 1}


class TestGateInRun:
    def test_run_refuses_once_the_whole_week_has_kicked_off(self, monkeypatch, tmp_path):
        monkeypatch.setattr(P, "build_frame", _frame)
        monkeypatch.setattr(P, "assert_no_leakage", lambda f: None)
        monkeypatch.setattr(P, "PROCESSED_PATH", tmp_path / "games.parquet")
        from nfl_predict.data.schedule import LateRunError
        with pytest.raises(LateRunError):
            P.run("late", now=T0 + timedelta(days=3, minutes=5))

    def test_run_after_the_opener_predicts_only_the_rest(self, monkeypatch, tmp_path):
        monkeypatch.setattr(P, "build_frame", _frame)
        monkeypatch.setattr(P, "assert_no_leakage", lambda f: None)
        monkeypatch.setattr(P, "PROCESSED_PATH", tmp_path / "games.parquet")
        monkeypatch.setattr(P, "PREDICTIONS_DIR", tmp_path / "predictions")
        monkeypatch.setattr(P, "snapshot_path", lambda t, n: tmp_path / "odds" / f"{t.season}_{t.week:02d}_{n}.json")
        monkeypatch.setattr(P, "fetch_snapshot", lambda t, n, f, now=None: {**_odds(), "fetched_at_utc": now.isoformat()})
        monkeypatch.setattr(P, "fit_final", lambda frame, spec: _bundle())
        pred_path, _ = P.run("early", now=T0 + timedelta(hours=5))  # Wednesday's game is under way
        rec = json.loads(pred_path.read_text())
        assert [p["game_id"] for p in rec["predictions"]] == ["2026_01_SF_LA", "2026_01_ATL_PIT"]

    def test_run_leaves_an_existing_pass_untouched_and_does_not_fail(self, monkeypatch, tmp_path):
        monkeypatch.setattr(P, "build_frame", _frame)
        monkeypatch.setattr(P, "assert_no_leakage", lambda f: None)
        monkeypatch.setattr(P, "PROCESSED_PATH", tmp_path / "games.parquet")
        monkeypatch.setattr(P, "PREDICTIONS_DIR", tmp_path)
        (tmp_path / "2026_01_early.json").write_text('{"published": "earlier"}')
        pred_path, _ = P.run("early", now=T0 - timedelta(days=1))
        assert json.loads(pred_path.read_text()) == {"published": "earlier"}

    def test_run_end_to_end_with_stubbed_odds_and_model(self, monkeypatch, tmp_path):
        monkeypatch.setattr(P, "build_frame", _frame)
        monkeypatch.setattr(P, "assert_no_leakage", lambda f: None)
        monkeypatch.setattr(P, "PROCESSED_PATH", tmp_path / "games.parquet")
        monkeypatch.setattr(P, "PREDICTIONS_DIR", tmp_path / "predictions")
        monkeypatch.setattr(P, "snapshot_path", lambda t, n: tmp_path / "odds" / f"{t.season}_{t.week:02d}_{n}.json")
        monkeypatch.setattr(P, "fetch_snapshot", lambda t, n, f, now=None: {**_odds(), "fetched_at_utc": now.isoformat()})
        monkeypatch.setattr(P, "fit_final", lambda frame, spec: _bundle())
        pred_path, odds_path = P.run("early", now=T0 - timedelta(days=1))
        rec = json.loads(pred_path.read_text())
        assert rec["n_games"] == 3 and odds_path.exists()
        # frozen together: the odds timestamp is the prediction timestamp
        assert json.loads(odds_path.read_text())["fetched_at_utc"] == rec["generated_at_utc"]
