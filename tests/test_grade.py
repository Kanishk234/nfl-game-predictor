"""Grading: which prediction counts, the scoring math, and idempotency."""

import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from nfl_predict import grade as G

T_THU = datetime(2026, 9, 11, 0, 20, tzinfo=UTC)
T_SUN = datetime(2026, 9, 13, 17, 0, tzinfo=UTC)
T_EARLY = T_THU - timedelta(days=2)
T_LATE = T_SUN - timedelta(hours=3)


def _row(gid, home, away, kickoff, p_home, margin, line, pv):
    return {"game_id": gid, "kickoff_utc": kickoff.isoformat(), "home_team": home, "away_team": away,
            "p_home": p_home, "pred_margin": margin, "pick": home if p_home >= 0.5 else away,
            "vegas": {"spread_line": line, "total_line": 44.0, "p_home_moneyline": pv,
                      "p_home_from_spread": pv, "n_books": 3} if line is not None else None}


def _passes():
    early = {"pass": "early", "generated_at_utc": T_EARLY.isoformat(), "predictions": [
        _row("2026_01_A_B", "B", "A", T_THU, 0.70, 6.0, 3.0, 0.60),
        _row("2026_01_C_D", "D", "C", T_SUN, 0.40, -2.0, -1.0, 0.45),
        _row("2026_01_E_F", "F", "E", T_SUN, 0.55, 1.0, None, None),
    ]}
    late = {"pass": "late", "generated_at_utc": T_LATE.isoformat(), "predictions": [
        _row("2026_01_C_D", "D", "C", T_SUN, 0.52, 3.0, 3.0, 0.50),   # supersedes early for C_D
        _row("2026_01_E_F", "F", "E", T_SUN, 0.60, 2.0, None, None),
    ]}
    return {"early": early, "late": late}


def _finals(played=("2026_01_A_B", "2026_01_C_D", "2026_01_E_F")):
    rows = [
        {"game_id": "2026_01_A_B", "home_score": 24, "away_score": 17},   # home wins by 7
        {"game_id": "2026_01_C_D", "home_score": 20, "away_score": 17},   # home wins by 3 = late line -> push
        {"game_id": "2026_01_E_F", "home_score": 10, "away_score": 13},   # home loses
    ]
    kicks = {"2026_01_A_B": T_THU, "2026_01_C_D": T_SUN, "2026_01_E_F": T_SUN}
    return pl.DataFrame([{**r, "season": 2026, "week": 1, "is_played": r["game_id"] in played,
                          "kickoff_utc": kicks[r["game_id"]],
                          "home_score": r["home_score"] if r["game_id"] in played else None,
                          "away_score": r["away_score"] if r["game_id"] in played else None} for r in rows])


class TestRescheduledGames:
    """A prediction records the kickoff that was scheduled when it was written. Games move."""

    def test_a_game_brought_forward_invalidates_a_pass_that_is_now_late(self):
        passes = _passes()
        # The Sunday game is moved to Saturday, before the late pass was generated.
        moved = {"2026_01_C_D": T_LATE - timedelta(hours=2)}
        off = G.official_predictions(passes, moved)
        assert off["2026_01_C_D"][0] == "early"   # the late pass is now after kickoff

    def test_a_game_pushed_back_keeps_the_later_pass(self):
        off = G.official_predictions(_passes(), {"2026_01_C_D": T_SUN + timedelta(days=1)})
        assert off["2026_01_C_D"][0] == "late"

    def test_the_recorded_kickoff_is_used_when_the_schedule_says_nothing(self):
        assert G.official_predictions(_passes(), {})["2026_01_C_D"][0] == "late"


class TestOfficialRule:
    def test_latest_pass_before_kickoff_wins(self):
        off = G.official_predictions(_passes())
        assert off["2026_01_A_B"][0] == "early"     # only in early
        assert off["2026_01_C_D"][0] == "late"      # late supersedes
        assert off["2026_01_E_F"][0] == "late"

    def test_a_pass_generated_after_kickoff_is_ignored(self):
        p = _passes()
        p["late"]["generated_at_utc"] = (T_SUN + timedelta(minutes=1)).isoformat()
        off = G.official_predictions(p)
        assert off["2026_01_C_D"][0] == "early"


class TestScoring:
    def test_per_game_grades(self):
        graded = {g["game_id"]: g for g in G.grade_rows(list(G.official_predictions(_passes()).values()), _finals())}
        ab = graded["2026_01_A_B"]
        assert ab["pick"] == "B" and ab["winner"] == "B"
        assert ab["model"]["correct"] == 1 and ab["vegas"]["correct"] == 1
        assert ab["model"]["brier"] == pytest.approx(0.09) and ab["model"]["abs_error"] == 1.0
        assert ab["model"]["ats"] == "win"       # we said +6 vs line +3, home won by 7: home covers
        cd = graded["2026_01_C_D"]
        assert cd["pass"] == "late"
        assert cd["model"]["ats"] == "push"      # margin 3 == line 3
        ef = graded["2026_01_E_F"]
        assert ef["pick"] == "F" and ef["winner"] == "E"
        assert ef["model"]["correct"] == 0 and ef["vegas"] is None and "ats" not in ef["model"]

    def test_a_tie_has_no_winner_and_is_not_scored_as_a_pick(self):
        finals = _finals().with_columns(
            pl.when(pl.col("game_id") == "2026_01_A_B").then(24).otherwise(pl.col("away_score")).alias("away_score"))
        graded = {g["game_id"]: g for g in G.grade_rows(list(G.official_predictions(_passes()).values()), finals)}
        assert graded["2026_01_A_B"]["winner"] == "tie"
        assert graded["2026_01_A_B"]["model"]["correct"] is None

    def test_summary_aggregates(self):
        graded = G.grade_rows(list(G.official_predictions(_passes()).values()), _finals())
        s = G.summarise(graded)
        assert s["n"] == 3 and s["n_with_line"] == 2
        assert s["model"]["accuracy"] == pytest.approx(2 / 3)
        assert s["model"]["ats"]["ats_w"] == 1 and s["model"]["ats"]["ats_push"] == 1
        assert s["vegas"]["accuracy"] == 1.0

    def test_unplayed_games_are_skipped_not_scored(self):
        graded = G.grade_rows(list(G.official_predictions(_passes()).values()), _finals(played=("2026_01_A_B",)))
        assert [g["game_id"] for g in graded] == ["2026_01_A_B"]


class TestIdempotency:
    def _setup(self, monkeypatch, tmp_path, played):
        pred_dir, res_dir = tmp_path / "predictions", tmp_path / "results"
        pred_dir.mkdir()
        for name, rec in _passes().items():
            (pred_dir / f"2026_01_{name}.json").write_text(json.dumps(rec))
        monkeypatch.setattr(G, "PREDICTIONS_DIR", pred_dir)
        monkeypatch.setattr(G, "RESULTS_DIR", res_dir)
        monkeypatch.setattr(G, "HISTORY_PATH", res_dir / "history.json")
        monkeypatch.setattr(G, "load_games", lambda: _finals(played))

    def test_every_written_file_is_byte_identical_on_a_second_run(self, monkeypatch, tmp_path):
        """Not just the history: a re-grade must produce no diff at all, or the scheduled job
        pushes a junk commit every time it runs."""
        self._setup(monkeypatch, tmp_path, ("2026_01_A_B", "2026_01_C_D", "2026_01_E_F"))
        G.run(2026)
        first = {p.name: p.read_text() for p in (tmp_path / "results").glob("*.json")}
        G.run(2026)
        second = {p.name: p.read_text() for p in (tmp_path / "results").glob("*.json")}
        assert first == second, f"changed: {[k for k in first if first[k] != second.get(k)]}"

    def test_second_run_is_identical_and_does_not_duplicate(self, monkeypatch, tmp_path):
        self._setup(monkeypatch, tmp_path, ("2026_01_A_B", "2026_01_C_D", "2026_01_E_F"))
        G.run(2026)
        first = json.loads((tmp_path / "results" / "history.json").read_text())
        G.run(2026)
        second = json.loads((tmp_path / "results" / "history.json").read_text())
        assert first == second  # byte-identical: a no-op run must not produce a commit
        assert len(first["seasons"]["2026"]["weeks"]) == 1
        assert first["seasons"]["2026"]["summary"]["n"] == 3

    def test_partial_week_is_marked_incomplete_then_completes(self, monkeypatch, tmp_path):
        self._setup(monkeypatch, tmp_path, ("2026_01_A_B",))
        G.run(2026)
        r = json.loads((tmp_path / "results" / "2026_01.json").read_text())
        assert r["n_graded"] == 1 and r["complete"] is False
        monkeypatch.setattr(G, "load_games", lambda: _finals())
        G.run(2026)
        r = json.loads((tmp_path / "results" / "2026_01.json").read_text())
        assert r["n_graded"] == 3 and r["complete"] is True

    def test_week_with_nothing_played_writes_no_result_file(self, monkeypatch, tmp_path):
        self._setup(monkeypatch, tmp_path, ())
        G.run(2026)
        assert not (tmp_path / "results" / "2026_01.json").exists()
