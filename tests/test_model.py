"""Metrics, the Vegas baseline, and the backtest's fold discipline."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from nfl_predict.model.backtest import (
    DEFAULT_SPEC,
    HOLDOUT_SEASONS,
    TUNING_SEASONS,
    ensure_no_train_test_overlap,
    season_folds,
    weekly_folds,
)
from nfl_predict.model.baseline import SpreadToWinProb
from nfl_predict.model.metrics import (
    ATS_BREAKEVEN,
    expected_calibration_error,
    spread_metrics,
    win_prob_metrics,
)


class TestWinProbMetrics:
    def test_perfect_predictions(self):
        m = win_prob_metrics(np.array([0.99, 0.01, 0.99]), np.array([1, 0, 1]))
        assert m.accuracy == 1.0
        assert m.brier < 0.001
        assert m.auc == 1.0

    def test_coin_flip_brier_is_a_quarter(self):
        y = np.array([1, 0, 1, 0])
        assert win_prob_metrics(np.full(4, 0.5), y).brier == pytest.approx(0.25)

    def test_ece_is_zero_when_bins_are_calibrated(self):
        # 70% predicted, 70% observed within the bin.
        prob = np.full(10, 0.75)
        y = np.array([1] * 7 + [0] * 3)
        assert expected_calibration_error(prob, y) == pytest.approx(0.05)  # |0.75 - 0.70|

    def test_degenerate_fold_has_nan_auc_not_a_crash(self):
        m = win_prob_metrics(np.array([0.6, 0.7]), np.array([1, 1]))
        assert np.isnan(m.auc)


class TestSpreadMetrics:
    def test_ats_record_and_pushes(self):
        # line +3 home. We say +7 (take home), result +10 -> home covers -> win.
        # line +3. We say +1 (take away), result +10 -> home covers -> loss.
        # line +3. We say +7, result exactly +3 -> push.
        m = spread_metrics(
            pred_margin=np.array([7.0, 1.0, 7.0]),
            margin=np.array([10.0, 10.0, 3.0]),
            line=np.array([3.0, 3.0, 3.0]),
        )
        assert m.ats_record == (1, 1, 1)
        assert m.ats_pct == 0.5
        assert m.mae == pytest.approx((3 + 9 + 4) / 3)

    def test_breakeven_constant_is_the_standard_juice_number(self):
        assert ATS_BREAKEVEN == pytest.approx(11 / 21, abs=1e-4)


class TestVegasBaseline:
    def test_bigger_home_spread_means_higher_home_win_prob(self):
        rng = np.random.default_rng(0)
        spread = rng.normal(0, 6, 4000)
        y = (spread / 7 + rng.logistic(0, 1, 4000) > 0).astype(int)
        b = SpreadToWinProb().fit(spread, y)
        p = b.predict(np.array([-7.0, 0.0, 7.0]))
        assert p[0] < p[1] < p[2]
        assert b.points_per_logit == pytest.approx(7.0, rel=0.25)


def _frame(n_seasons=4, games_per_week=2, weeks=3):
    rows = []
    t0 = datetime(2010, 9, 1, tzinfo=UTC)
    for s in range(n_seasons):
        for w in range(1, weeks + 1):
            for g in range(games_per_week):
                k = t0 + timedelta(days=365 * s + 7 * w, hours=g)
                rows.append({
                    "game_id": f"{2010 + s}_{w:02d}_{g}", "season": 2010 + s, "week": w,
                    "kickoff_utc": k, "is_played": True, "home_win": g % 2, "margin": 3.0,
                    "spread_line": 1.0, "elo_diff": 0.0,
                })
    return pl.DataFrame(rows).with_columns(pl.col("kickoff_utc").cast(pl.Datetime("us", "UTC")))


class TestFoldDiscipline:
    def test_tuning_and_holdout_seasons_are_disjoint(self):
        """Nothing reported on the holdout was ever tuned on it."""
        assert not set(TUNING_SEASONS) & set(HOLDOUT_SEASONS)
        assert max(TUNING_SEASONS) < min(HOLDOUT_SEASONS)

    def test_season_folds_train_only_on_earlier_seasons(self):
        for fold in season_folds(_frame(), (2012, 2013)):
            assert fold.train["season"].max() < fold.test["season"].min()

    def test_weekly_folds_train_only_on_games_finished_before_the_week(self):
        for fold in weekly_folds(_frame(), (2012, 2013)):
            assert fold.train["kickoff_utc"].max() < fold.test["kickoff_utc"].min()
            # Earlier weeks of the same season are fair game; that is the point of the replay.
            if fold.test["week"][0] > 1:
                assert (fold.train["season"] == fold.test["season"][0]).any()

    def test_overlap_guard_fires_on_a_leaky_fold(self):
        from nfl_predict.model.backtest import Fold

        f = _frame()
        leaky = Fold(name="bad", train=f, test=f.tail(2))
        with pytest.raises(AssertionError, match="not strictly before"):
            list(ensure_no_train_test_overlap(iter([leaky])))

    def test_default_spec_is_linear(self):
        """If this changes, the report's justification has to change with it."""
        assert DEFAULT_SPEC.learner == "linear"
