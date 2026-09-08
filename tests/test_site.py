"""The site is a view over data/: it must render every state the data can be in."""

import re
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
        assert 'Patriots <small>at</small> Seahawks' in page
        assert '<strong>SEA</strong><span class="conf">67%</span>' in page and "to win" in page
        assert '<span class="who">Vegas</span>' in page and "62%" in page
        assert "teamlogos/nfl/500/sea.png" in page and "teamlogos/nfl/500/ne.png" in page
        assert '<span class="spread-who">Us</span><span class="spread-val">SEA by 5.1</span>' in page
        assert '<span class="spread-who">Vegas</span><span class="spread-val">SEA by 3.0</span>' in page
        # every spread row carries the same three cells, so the bars cannot fall out of line
        assert page.count('class="spread-row ') == page.count('class="spread-track"')
        assert "no line yet" in page                       # missing line is shown as missing
        assert 'style="--team:#' in page                    # picked team's colour on the card
        assert '<h3 class="slot"><span>Wednesday night</span>' in page
        assert page.count('<div class="week-grid">') == 1  # one grid, labels span it
        assert "All games this week" in page               # the table at the end
        assert "30 hours before the first kickoff" in page  # provenance, collapsed
        assert "abc1234" in page

    def test_away_pick_shows_away_confidence(self):
        preds = [_pred(1, "early", [_row("2026_01_NE_SEA", "SEA", "NE", p=0.40, m=-2.0)])]
        page = S.render_site(preds, {}, None, None, NOW)["index.html"]
        assert '<strong>NE</strong><span class="conf">60%</span>' in page
        assert "we disagree" in page  # Vegas (58% SEA) and we (NE) differ

    def test_graded_card_shows_score_winner_and_verdict(self):
        preds = [_pred(1, "early", [_row("2026_01_NE_SEA", "SEA", "NE")])]
        results = {(2026, 1): _result(1, [_graded("2026_01_NE_SEA", "SEA", "NE", "SEA", 1)])}
        page = S.render_site(preds, results, None, None, NOW)["index.html"]
        assert '<span class="score">NE 17, SEA 24</span>' in page
        assert "SEA won, <span class=\"hit\">✓ we were right</span> and our side covered the spread." in page
        assert "Week 1, 2026, complete" in page
        assert "1 game, 1 played" in page  # the slot label counts what is done

    def test_late_pass_supersedes_early_for_its_games(self):
        early = _pred(1, "early", [_row("2026_01_NE_SEA", "SEA", "NE", p=0.62), _row("2026_01_ATL_PIT", "PIT", "ATL", p=0.55)])
        late = _pred(1, "late", [_row("2026_01_ATL_PIT", "PIT", "ATL", p=0.40)], gen_offset_h=-3)
        page = S.render_site([early, late], {}, None, None, NOW)["index.html"]
        assert page.count("Falcons <small>at</small> Steelers") == 1
        assert 'id="2026_01_ATL_PIT" style="--team:#' in page

    def test_season_page_with_backtest(self):
        bt = {"holdout_seasons": [2021, 2022], "n_folds": 40, "n_games": 500,
              "overall": {"model": {"win": {"accuracy": 0.646, "brier": 0.22}, "spread": {"mae": 10.07}},
                          "vegas": {"win": {"accuracy": 0.665, "brier": 0.21}, "spread": {"mae": 9.76}}},
              "by_season": {"2021": {"model": {"win": {"accuracy": 0.62, "brier": 0.23}, "spread": {"mae": 10.5}},
                                     "vegas": {"win": {"accuracy": 0.63, "brier": 0.22}, "spread": {"mae": 10.1}}}},
              "calibration": {"model": [{"bin_low": 0.5, "bin_high": 0.6, "n": 50, "mean_predicted": 0.55, "observed": 0.52}],
                              "vegas": [{"bin_low": 0.5, "bin_high": 0.6, "n": 5, "mean_predicted": 0.55, "observed": 0.6}]}}
        page = S.render_site([], {}, None, bt, NOW)["season.html"]
        assert "The dry run: 2021 to 2022" in page
        assert "<dd>64.6%</dd>" in page and '<dd class="muted">66.5%</dd>' in page   # us, then Vegas
        assert "<dd>10.1 pts</dd>" in page and '<dd class="muted">9.8 pts</dd>' in page
        assert "Nothing graded yet" in page
        assert 'fill="none"' in page  # the n=5 bin is hollow


class TestTeamColours:
    def test_hue_is_kept_when_lightening(self):
        # Rams navy on white stays as is; on the dark panel it becomes a lighter blue, not yellow.
        assert S.readable_team_color("LA", S.LIGHT_PANEL) == "#003594"
        dark = S.readable_team_color("LA", S.DARK_PANEL)
        r, g, b = (int(dark[i:i + 2], 16) for i in (1, 3, 5))
        assert b > r and b > g and S._contrast(dark, S.DARK_PANEL) >= 3.0

    def test_black_primary_falls_back_to_the_iconic_secondary(self):
        assert S.readable_team_color("PIT", S.DARK_PANEL).upper().startswith("#FFB6")  # Steelers gold
        assert S.readable_team_color("LV", S.DARK_PANEL).upper() == "#A5ACAF"  # Raiders silver, not grey

    def test_every_team_reads_on_both_surfaces(self):
        for abbr in S.TEAM_COLORS:
            for surface in (S.LIGHT_PANEL, S.DARK_PANEL):
                assert S._contrast(S.readable_team_color(abbr, surface), surface) >= 3.0, abbr


class TestSlots:
    def test_slot_labels(self):
        assert S.slot_label("2026-09-10T00:20:00+00:00") == "Wednesday night"
        assert S.slot_label("2026-09-13T17:00:00+00:00") == "Sunday 1:00 pm"
        assert S.slot_label("2026-09-13T20:25:00+00:00") == "Sunday 4:25 pm"
        assert S.slot_label("2026-09-15T00:15:00+00:00") == "Monday night"


def test_vegas_colour_is_not_any_team_colour():
    """Pink on purpose: no NFL team uses it, so the market marker is never mistaken for a team."""
    every_team = {c.upper() for p1, p2, _ in S.TEAM_COLORS.values() for c in (p1, p2)}
    every_team |= {S.readable_team_color(a, S.DARK_PANEL).upper() for a in S.TEAM_COLORS}
    assert S.VEGAS_DARK.upper() not in every_team


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


class TestSeasonChartHonesty:
    """A week in progress must not distort the weekly chart, and nothing may draw off-plot."""

    @staticmethod
    def _history(weeks):
        return {"seasons": {"2025": {
            "summary": {"n": sum(w["n_graded"] for w in weeks), "n_with_line": 1,
                        "model": {"accuracy": 0.6, "brier": 0.2, "log_loss": 0.6, "auc": 0.6, "ece": 0.1,
                                  "spread_mae": 10.0, "ats": {"ats_w": 5, "ats_l": 4, "ats_push": 0, "ats_pct": 0.55}},
                        "vegas": {"accuracy": 0.65, "brier": 0.2, "log_loss": 0.6, "auc": 0.6, "ece": 0.1, "spread_mae": 9.7}},
            "weeks": weeks, "calibration": {"model": [], "vegas": []}}}}

    @staticmethod
    def _week(n, acc, complete, graded=16):
        return {"season": 2025, "week": n, "n_games": 16, "n_graded": graded, "complete": complete,
                "summary": {"n": graded,
                            "model": {"accuracy": acc, "brier": 0.2, "log_loss": 0.6, "auc": 0.6, "ece": 0.1, "spread_mae": 10.0},
                            "vegas": {"accuracy": acc, "brier": 0.2, "log_loss": 0.6, "auc": 0.6, "ece": 0.1, "spread_mae": 9.7}}}

    def test_a_part_played_week_is_left_off_the_chart(self):
        hist = self._history([self._week(1, 0.75, True), self._week(2, 0.62, True),
                              self._week(3, 0.0, False, graded=1)])   # one game in, and it was wrong
        # the season page follows the season of the latest published week
        preds = [{**_pred(3, "early", [_row("2025_03_A_B", "B", "A", week=3)]), "season": 2025}]
        page = S.render_site(preds, {}, hist, None, NOW)["season.html"]
        assert "wk 1" in page and "wk 2" in page
        assert "wk 3" not in page                        # not plotted
        assert "joins the line once all its games are played" in page
        assert hist["seasons"]["2025"]["summary"]["n"] == 33   # but its games still count in the total

    def test_values_below_the_axis_are_clamped_into_the_plot(self):
        svg = S.line_chart({"Model": [(1, 0.0), (2, 1.0)]}, "picks right", 0.3, 1.0)
        ys = [float(m) for m in re.findall(r'cy="([\d.]+)"', svg)]
        assert ys, "expected plotted points"
        assert all(16 <= y <= 208 for y in ys), f"points drawn outside the plot area: {ys}"
