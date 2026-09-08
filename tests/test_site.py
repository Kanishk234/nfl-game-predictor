"""The site is a view over data/: it must render every state the data can be in."""

from datetime import UTC, datetime, timedelta

from nfl_predict import site_build as S

T0 = datetime(2026, 9, 10, 0, 20, tzinfo=UTC)


def _pred(week, pass_name, games, gen_offset_h=-30):
    return {
        "season": 2026, "week": week, "pass": pass_name,
        "generated_at_utc": (T0 + timedelta(days=7 * (week - 1), hours=gen_offset_h)).isoformat(),
        "gate": {"earliest_kickoff_utc": T0.isoformat(), "seconds_before_first_kickoff": -gen_offset_h * 3600},
        "model": {"n_train": 6499, "trained_through_game": "2025_22_SEA_NE", "code_version": "abc1234",
                  "learner": "linear", "params": {}, "n_features": 30, "feature_hash": "x",
                  "trained_through_utc": "", "trained_at_utc": ""},
        "n_games": len(games), "predictions": games,
    }


def _row(gid, home, away, week=1, p=0.62, m=3.1, line=2.5, pv=0.58):
    return {"game_id": gid, "kickoff_utc": (T0 + timedelta(days=7 * (week - 1))).isoformat(), "home_team": home,
            "away_team": away, "p_home": p, "pred_margin": m, "pick": home if p >= 0.5 else away,
            "vegas": {"spread_line": line, "total_line": 44.0, "p_home_moneyline": pv, "p_home_from_spread": pv, "n_books": 3} if line is not None else None}


def _result(week, games):
    return {"season": 2026, "week": week, "n_games": len(games), "n_graded": len(games), "complete": True,
            "summary": {"n": len(games), "n_with_line": len(games),
                        "model": {"accuracy": 0.5, "brier": 0.24, "log_loss": 0.68, "auc": 0.5, "ece": 0.1, "spread_mae": 9.0,
                                  "ats": {"ats_w": 1, "ats_l": 1, "ats_push": 0, "ats_pct": 0.5}},
                        "vegas": {"accuracy": 1.0, "brier": 0.2, "log_loss": 0.6, "auc": 0.5, "ece": 0.1, "spread_mae": 8.0}},
            "games": games}


def _graded(gid, home, away, winner, correct):
    return {"game_id": gid, "home_team": home, "away_team": away, "home_score": 24, "away_score": 17,
            "margin": 7.0, "home_win": 1, "winner": winner, "pick": home,
            "model": {"p_home": 0.62, "pred_margin": 3.1, "correct": correct, "brier": 0.14, "abs_error": 3.9, "ats": "win"},
            "vegas": {"p_home": 0.58, "spread_line": 2.5, "correct": 1, "brier": 0.18, "abs_error": 4.5}}


NOW = T0 - timedelta(hours=20)


class TestStates:
    def test_no_predictions_at_all(self):
        page = S.render([], {}, None, None, NOW)
        assert "No predictions published yet" in page

    def test_upcoming_week_with_no_results(self):
        preds = [_pred(1, "early", [_row("2026_01_NE_SEA", "SEA", "NE"), _row("2026_01_SF_LA", "LA", "SF", line=None)])]
        page = S.render(preds, {}, None, None, NOW)
        assert "Week 1, 2026" in page and "not played" in page
        assert "NE at <strong>SEA</strong>" in page
        assert "no line" in page                     # missing line is shown as missing, not invented
        assert "30 hours before the first kickoff" in page
        assert "abc1234" in page                     # model provenance is on the page
        assert "Earlier weeks" not in page

    def test_graded_week_shows_winner_and_marks(self):
        preds = [_pred(1, "early", [_row("2026_01_NE_SEA", "SEA", "NE")])]
        results = {(2026, 1): _result(1, [_graded("2026_01_NE_SEA", "SEA", "NE", "SEA", 1)])}
        page = S.render(preds, results, None, None, NOW)
        assert '<span class="winner">SEA</span>' in page and 'class="hit"' in page
        assert "ATS win" in page and "complete" in page

    def test_late_pass_supersedes_early_for_its_games(self):
        early = _pred(1, "early", [_row("2026_01_NE_SEA", "SEA", "NE", p=0.62), _row("2026_01_ATL_PIT", "PIT", "ATL", p=0.55)])
        late = _pred(1, "late", [_row("2026_01_ATL_PIT", "PIT", "ATL", p=0.40)], gen_offset_h=-3)
        page = S.render([early, late], {}, None, None, NOW)
        # one row per game; PIT's row is the late pass (40%, pick ATL)
        assert page.count("2026_01_ATL_PIT") == 0 and page.count("ATL at <strong>PIT</strong>") == 1
        assert "<strong>ATL</strong>" in page and "late pass" in page

    def test_past_weeks_collapse_and_latest_is_open(self):
        preds = [_pred(1, "early", [_row("2026_01_NE_SEA", "SEA", "NE")]),
                 _pred(2, "early", [_row("2026_02_KC_DEN", "DEN", "KC", week=2)])]
        page = S.render(preds, {}, None, None, NOW + timedelta(days=7))
        assert '<section id="this-week"><h2>Week 2' in page
        assert '<details class="past-week"><summary><h3>Week 1' in page

    def test_backtest_section_when_file_exists(self):
        bt = {"holdout_seasons": [2021, 2022], "n_folds": 40, "n_games": 500,
              "overall": {"model": {"win": {"accuracy": 0.646, "brier": 0.22}, "spread": {"mae": 10.07}},
                          "vegas": {"win": {"accuracy": 0.665, "brier": 0.21}, "spread": {"mae": 9.76}}},
              "by_season": {"2021": {"model": {"win": {"accuracy": 0.62, "brier": 0.23}, "spread": {"mae": 10.5}},
                                     "vegas": {"win": {"accuracy": 0.63, "brier": 0.22}, "spread": {"mae": 10.1}}}},
              "calibration": {"model": [{"bin_low": 0.5, "bin_high": 0.6, "n": 50, "mean_predicted": 0.55, "observed": 0.52}],
                              "vegas": [{"bin_low": 0.5, "bin_high": 0.6, "n": 5, "mean_predicted": 0.55, "observed": 0.6}]}}
        page = S.render([], {}, None, bt, NOW)
        assert "The model does not beat Vegas" in page and "Before going live: 2021, 2022" in page
        assert 'fill="none"' in page  # the n=5 bin is hollow


class TestGauge:
    def test_gauge_clamps_and_labels(self):
        svg = S.spread_gauge(30.0, -3.0, 7.0)
        assert 'aria-label="our margin +30.0, line −3.0, actual +7"' in svg
        assert "gauge-model" in svg and "gauge-vegas" in svg and "gauge-actual" in svg

    def test_gauge_without_line_or_result(self):
        svg = S.spread_gauge(2.5, None, None)
        assert "gauge-vegas" not in svg and "gauge-actual" not in svg


def test_real_build_writes_index_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SITE_DIR", tmp_path)
    S.main()
    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "<title>nfl-predict</title>" in page
    assert "apiKey" not in page
    assert "<script" not in page  # no JavaScript, by design
    assert page.count("http://") == 0  # relative or https only
    assert "2026_01_early.json" in page  # the real Week 1 file is linked
