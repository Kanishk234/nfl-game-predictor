"""Walk-forward backtesting that mirrors how the model is actually used.

Production retrains every pass on every game completed so far and predicts the coming week.
So the honest backtest is the same loop replayed over history: for each (season, week) in the
test window, fit on every game that had *finished* before that week's first kickoff, predict
that week, move on. Nothing in a fold's training set postdates the games it is scored on.

Two fold generators:

- `weekly_folds` - the faithful replay above. One fit per week; used for reported numbers.
- `season_folds` - one fit per season (train on all prior seasons, score the whole season).
  ~20x cheaper, so it is what hyperparameter search runs on. It is slightly pessimistic (the
  model never sees the current season's early weeks) but the features already carry in-season
  information, so it ranks configurations the same way.

Tuning and reporting must use disjoint seasons: see `TUNING_SEASONS` / `HOLDOUT_SEASONS`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_predict.data.pipeline import FEATURE_COLUMNS
from nfl_predict.data.schedule import GAME_DURATION
from nfl_predict.model.baseline import SpreadToWinProb
from nfl_predict.model.metrics import (
    SpreadMetrics,
    WinProbMetrics,
    spread_metrics,
    win_prob_metrics,
)

#: Hyperparameters are chosen on walk-forward results over these seasons...
TUNING_SEASONS = tuple(range(2012, 2020))
#: ...and reported on these, which nothing is ever tuned against. 2020 sits between the two so
#: it is training data for the holdout but never a scored season itself.
HOLDOUT_SEASONS = (2021, 2022, 2023, 2024, 2025)


@dataclass(frozen=True)
class Fold:
    name: str
    train: pl.DataFrame
    test: pl.DataFrame


def season_folds(frame: pl.DataFrame, seasons: tuple[int, ...]) -> Iterator[Fold]:
    played = frame.filter(pl.col("is_played"))
    for season in seasons:
        yield Fold(
            name=str(season),
            train=played.filter(pl.col("season") < season),
            test=played.filter(pl.col("season") == season),
        )


def weekly_folds(frame: pl.DataFrame, seasons: tuple[int, ...]) -> Iterator[Fold]:
    played = frame.filter(pl.col("is_played"))
    weeks = (
        played.filter(pl.col("season").is_in(seasons))
        .group_by("season", "week")
        .agg(pl.col("kickoff_utc").min().alias("first_kickoff"))
        .sort("season", "week")
    )
    for season, week, first_kickoff in weeks.iter_rows():
        # Only games that had *finished* before the week opened are known at train time.
        train = played.filter(pl.col("kickoff_utc") + GAME_DURATION < first_kickoff)
        test = played.filter((pl.col("season") == season) & (pl.col("week") == week))
        yield Fold(name=f"{season}w{week:02d}", train=train, test=test)


@dataclass(frozen=True)
class ModelSpec:
    """A configuration to backtest: which learner, its parameters, which features.

    `learner` is "linear" (logistic regression for win probability, ridge for margin, both on
    imputed + standardised features) or "lgbm" (gradient-boosted trees, both targets).
    """

    learner: str = "linear"
    params: dict = field(default_factory=dict)
    features: tuple[str, ...] = tuple(FEATURE_COLUMNS)

    def classifier(self):
        if self.learner == "linear":
            return make_pipeline(
                SimpleImputer(), StandardScaler(),
                LogisticRegression(C=self.params.get("C", 0.1), max_iter=5000),
            )
        return LGBMClassifier(**self.params, verbose=-1)

    def regressor(self):
        if self.learner == "linear":
            return make_pipeline(
                SimpleImputer(), StandardScaler(), Ridge(alpha=self.params.get("alpha", 10.0))
            )
        return LGBMRegressor(**self.params, verbose=-1)


#: What ships. Linear, because on this feature set it is what backtests best: on the tuning
#: seasons logistic regression scores log loss 0.6190 against 0.6249 for the best of 60 boosted
#: configurations (which collapsed to depth-1 stumps under the search), and blending the two did
#: not beat the linear model alone. See docs/reports/phase2_model_backtest.md. C and alpha are
#: flat across an order of magnitude; these are mid-range picks.
DEFAULT_SPEC = ModelSpec(learner="linear", params={"C": 0.1, "alpha": 10.0})

#: The boosted-tree contender, kept so the comparison stays reproducible. Heavily regularised:
#: the random search preferred stumps with large leaves, i.e. barely-nonlinear models.
LGBM_SPEC = ModelSpec(
    learner="lgbm",
    params={
        "n_estimators": 800, "learning_rate": 0.01, "num_leaves": 2, "min_child_samples": 200,
        "subsample": 0.5, "subsample_freq": 1, "colsample_bytree": 0.4, "reg_lambda": 1.0,
        "max_bin": 255,
    },
)


def _xy(df: pl.DataFrame, features: tuple[str, ...]):
    X = df.select(features).to_numpy().astype(float)
    return X, df["home_win"].to_numpy(), df["margin"].to_numpy().astype(float)


def _fit_predict(spec: ModelSpec, fold: Fold) -> pl.DataFrame:
    X_tr, y_tr, m_tr = _xy(fold.train, spec.features)
    X_te, _, _ = _xy(fold.test, spec.features)

    clf = spec.classifier().fit(X_tr, y_tr)
    reg = spec.regressor().fit(X_tr, m_tr)
    vegas = SpreadToWinProb().fit(fold.train["spread_line"].to_numpy(), y_tr)

    return fold.test.select(
        "game_id", "season", "week", "kickoff_utc", "home_team", "away_team",
        "home_win", "margin", "spread_line",
    ).with_columns(
        pl.Series("p_home", clf.predict_proba(X_te)[:, 1]),
        pl.Series("pred_margin", reg.predict(X_te)),
        pl.Series("p_home_vegas", vegas.predict(fold.test["spread_line"].to_numpy())),
        pl.lit(fold.name).alias("fold"),
    )


@dataclass
class BacktestResult:
    predictions: pl.DataFrame

    def _scores(self, df: pl.DataFrame) -> dict:
        y = df["home_win"].to_numpy()
        margin = df["margin"].to_numpy()
        line = df["spread_line"].to_numpy()
        return {
            "model": {
                "win": win_prob_metrics(df["p_home"].to_numpy(), y),
                "spread": spread_metrics(df["pred_margin"].to_numpy(), margin, line),
            },
            "vegas": {
                "win": win_prob_metrics(df["p_home_vegas"].to_numpy(), y),
                # The line is its own margin prediction; ATS against itself is meaningless.
                "spread": spread_metrics(line, margin, line),
            },
        }

    def overall(self) -> dict:
        return self._scores(self.predictions)

    def by_season(self) -> dict[int, dict]:
        return {
            s: self._scores(self.predictions.filter(pl.col("season") == s))
            for s in sorted(self.predictions["season"].unique().to_list())
        }

    def summary_table(self) -> pl.DataFrame:
        rows = []
        for season, s in [*self.by_season().items(), ("all", self.overall())]:
            for who in ("model", "vegas"):
                w: WinProbMetrics = s[who]["win"]
                sp: SpreadMetrics = s[who]["spread"]
                rows.append({
                    "season": str(season), "who": who, "n": w.n,
                    "acc": w.accuracy, "brier": w.brier, "logloss": w.log_loss,
                    "auc": w.auc, "ece": w.ece, "mae": sp.mae,
                    "ats": sp.ats_pct if who == "model" else None,
                })
        return pl.DataFrame(rows)


def run_backtest(frame: pl.DataFrame, folds: Iterator[Fold], spec: ModelSpec) -> BacktestResult:
    preds = [_fit_predict(spec, fold) for fold in folds]
    return BacktestResult(predictions=pl.concat(preds).sort("kickoff_utc", "game_id"))


def objective(result: BacktestResult) -> float:
    """What tuning minimises: model log loss. Chosen over accuracy because the site displays
    probabilities, so calibration is the product, not a nice-to-have."""
    return result.overall()["model"]["win"].log_loss


def edge_vs_vegas(result: BacktestResult) -> dict[str, float]:
    """Model minus Vegas on each headline metric, sign-adjusted so positive = model better."""
    o = result.overall()
    m, v = o["model"], o["vegas"]
    return {
        "accuracy": m["win"].accuracy - v["win"].accuracy,
        "brier": v["win"].brier - m["win"].brier,
        "log_loss": v["win"].log_loss - m["win"].log_loss,
        "mae": v["spread"].mae - m["spread"].mae,
    }


def ensure_no_train_test_overlap(folds: Iterator[Fold]) -> Iterator[Fold]:
    """Belt-and-braces: assert every fold's training set is entirely earlier than its test set."""
    for fold in folds:
        latest_train = fold.train["kickoff_utc"].max()
        earliest_test = fold.test["kickoff_utc"].min()
        if latest_train is not None and latest_train >= earliest_test:
            raise AssertionError(
                f"fold {fold.name}: training data ({latest_train}) is not strictly before "
                f"test data ({earliest_test})"
            )
        if np.intersect1d(fold.train["game_id"].to_numpy(), fold.test["game_id"].to_numpy()).size:
            raise AssertionError(f"fold {fold.name}: a game appears in both train and test")
        yield fold
