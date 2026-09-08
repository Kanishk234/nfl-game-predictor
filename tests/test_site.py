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


class TestPages:
    def test_no_predictions_at_all(self):
        pages = S.render_site([], {}, None, None, NOW)
        assert set(pages) == {"index.html", "season.html"}
        assert "No predictions published yet" in pages["index.html"]

    def test_one_page_per_week_and_index_is_the_latest(self):
        preds = [_pred(1, "early", [_row("2026_01_NE_SEA", "SEA", "NE")]),
                 _pred(2, "early", [_row("2026_02_KC_DEN", "DEN", "KC", week=2)])]
        pages = S.render_site(preds, {}, None, None, NOW + timedelta(days=7))
        assert set(pages) == {"index.html", "weeks/2026_01.html", "weeks/2026_02.html", "season.html"}
        assert "<h2>Week 2, 2026" in pages["index.html"] and "<h2>Week 1, 2026" in pages["weeks/2026_01.html"]
        # the strip lists both weeks on every page, with relative links that work from a subfolder
        for p in pages.values():
            assert "Week 1</a>" in p and "Week 2</a>" in p
        assert 'href="../weeks/2026_02.html"' in pages["weeks/2026_01.html"]
        assert 'href="weeks/2026_02.html"' in pages["index.html"]

    def test_upcoming_game_card_reads_in_plain_words(self):
        preds = [_pred(1, "early", [_row("2026_01_NE_SEA", "SEA", "NE", p=0.67, m=5.1, line=3.0, pv=0.618),
                                    _row("2026_01_SF_LA", "LA", "SF", line=None)])]
        page = S.render_site(preds, {}, None, None, NOW)["index.html"]
        assert "<h3>Patriots at Seahawks</h3>" in page
        assert '<span class="who">Our pick</span>' in page and "<strong>SEA</strong>" in page and "67%" in page
        assert '<span class="who">Vegas favorite</span>' in page and "62%" in page
        assert "SEA should win by about 5. Vegas has SEA by 3." in page
        assert '<span class="spread-who">Us</span><span class="spread-val">SEA by 5.1</span>' in page
        assert '<span class="spread-who">Vegas</span><span class="spread-val">SEA by 3.0</span>' in page
        assert "no line yet" in page                       # missing line is shown as missing
        assert 'style="--tl:#002244;--td:' in page          # picked team's colour, one per surface
        assert "All games this week" in page               # the table at the end
        assert "30 hours before the first kickoff" in page  # provenance, collapsed
        assert "abc1234" in page

    def test_away_pick_shows_away_confidence(self):
        preds = [_pred(1, "early", [_row("2026_01_NE_SEA", "SEA", "NE", p=0.40, m=-2.0)])]
        page = S.render_site(preds, {}, None, None, NOW)["index.html"]
        assert "<strong>NE</strong><span class=\"conf\">60%</span>" in page
        assert "NE should win by about 2." in page

    def test_graded_card_shows_score_winner_and_verdict(self):
        preds = [_pred(1, "early", [_row("2026_01_NE_SEA", "SEA", "NE")])]
        results = {(2026, 1): _result(1, [_graded("2026_01_NE_SEA", "SEA", "NE", "SEA", 1)])}
        page = S.render_site(preds, results, None, None, NOW)["index.html"]
        assert '<span class="score">NE 17, SEA 24</span>' in page
        assert "SEA won, <span class=\"hit\">✓ we were right</span> and our side covered the spread." in page
        assert "Week 1, 2026, complete" in page

    def test_late_pass_supersedes_early_for_its_games(self):
        early = _pred(1, "early", [_row("2026_01_NE_SEA", "SEA", "NE", p=0.62), _row("2026_01_ATL_PIT", "PIT", "ATL", p=0.55)])
        late = _pred(1, "late", [_row("2026_01_ATL_PIT", "PIT", "ATL", p=0.40)], gen_offset_h=-3)
        page = S.render_site([early, late], {}, None, None, NOW)["index.html"]
        assert page.count("<h3>Falcons at Steelers</h3>") == 1
        assert 'style="--tl:#A71930;--td:' in page          # ATL's colour on the card

    def test_season_page_with_backtest(self):
        bt = {"holdout_seasons": [2021, 2022], "n_folds": 40, "n_games": 500,
              "overall": {"model": {"win": {"accuracy": 0.646, "brier": 0.22}, "spread": {"mae": 10.07}},
                          "vegas": {"win": {"accuracy": 0.665, "brier": 0.21}, "spread": {"mae": 9.76}}},
              "by_season": {"2021": {"model": {"win": {"accuracy": 0.62, "brier": 0.23}, "spread": {"mae": 10.5}},
                                     "vegas": {"win": {"accuracy": 0.63, "brier": 0.22}, "spread": {"mae": 10.1}}}},
              "calibration": {"model": [{"bin_low": 0.5, "bin_high": 0.6, "n": 50, "mean_predicted": 0.55, "observed": 0.52}],
                              "vegas": [{"bin_low": 0.5, "bin_high": 0.6, "n": 5, "mean_predicted": 0.55, "observed": 0.6}]}}
        page = S.render_site([], {}, None, bt, NOW)["season.html"]
        assert "picked the winner 64.6% of the\n     time; Vegas picked 66.5%." in page and "The dry run: 2021 to 2022" in page
        assert "Nothing graded yet" in page
        assert 'fill="none"' in page  # the n=5 bin is hollow


class TestTeamColours:
    def test_dark_primary_is_swapped_or_lightened_on_dark_surface(self):
        # Seahawks navy is fine on white, invisible on the dark panel: the green takes over.
        assert S.readable_team_color("SEA", S.LIGHT_PANEL) == "#002244"
        assert S.readable_team_color("SEA", S.DARK_PANEL) == "#69BE28"
        # Raiders: black and silver. Silver reads on dark; on white neither is strong but black passes.
        assert S._contrast(S.readable_team_color("LV", S.DARK_PANEL), S.DARK_PANEL) >= 3.0
        assert S._contrast(S.readable_team_color("LV", S.LIGHT_PANEL), S.LIGHT_PANEL) >= 3.0

    def test_every_team_reads_on_both_surfaces(self):
        for abbr in S.TEAM_COLORS:
            for surface in (S.LIGHT_PANEL, S.DARK_PANEL):
                assert S._contrast(S.readable_team_color(abbr, surface), surface) >= 3.0, abbr


class TestBar:
    def test_bar_splits_by_probability_and_marks_vegas(self):
        svg = S.prob_bar(0.75, 0.6, "SEA", "NE")
        assert 'aria-label="NE 25%, SEA 75%, Vegas has SEA at 60%"' in svg and "bar-vegas" in svg

    def test_bar_without_line(self):
        assert "bar-vegas" not in S.prob_bar(0.5, None, "A", "B")


def test_real_build_writes_pages_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SITE_DIR", tmp_path)
    S.main()
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert (tmp_path / "weeks" / "2026_01.html").exists() and (tmp_path / "season.html").exists()
    assert "apiKey" not in index and "<script" not in index and "http://" not in index
    assert "2026_01_early.json" in index
