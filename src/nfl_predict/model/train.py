"""Retrain on everything completed, save the artifact, and replay the holdout backtest.

Run as `python -m nfl_predict.model.train`. Writes:

- `data/models/latest.joblib` - the fitted classifier + regressor plus provenance (what data
  it was trained through, when, which code). The weekly prediction pass loads this.
- `data/processed/backtest.json` - the weekly walk-forward replay over the holdout seasons:
  per-season and overall metrics for the model and for Vegas, plus calibration bins. This is
  the number the site's "how good is this model" page is built from, and it is regenerable.

Both directories are gitignored. Nothing here is a source of truth; the committed data under
`data/predictions`, `data/odds` and `data/results` is.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import polars as pl

from nfl_predict.data.pipeline import PROCESSED_PATH
from nfl_predict.model.backtest import (
    DEFAULT_SPEC,
    HOLDOUT_SEASONS,
    ModelSpec,
    edge_vs_vegas,
    ensure_no_train_test_overlap,
    run_backtest,
    weekly_folds,
)
from nfl_predict.model.metrics import calibration_table

MODEL_PATH = Path("data/models/latest.joblib")
BACKTEST_PATH = Path("data/processed/backtest.json")


def code_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fit_final(frame: pl.DataFrame, spec: ModelSpec = DEFAULT_SPEC) -> dict:
    """Fit on every completed game and bundle with the provenance a prediction file must carry."""
    train = frame.filter(pl.col("is_played"))
    X = train.select(spec.features).to_numpy().astype(float)
    clf = spec.classifier().fit(X, train["home_win"].to_numpy())
    reg = spec.regressor().fit(X, train["margin"].to_numpy().astype(float))
    return {
        "classifier": clf,
        "regressor": reg,
        "spec": asdict(spec),
        "features": list(spec.features),
        "n_train": train.height,
        "trained_through_utc": train["kickoff_utc"].max().isoformat(),
        "trained_through_game": train.sort("kickoff_utc").row(-1, named=True)["game_id"],
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "code_version": code_version(),
    }


def holdout_report(frame: pl.DataFrame, spec: ModelSpec = DEFAULT_SPEC) -> dict:
    result = run_backtest(
        frame, ensure_no_train_test_overlap(weekly_folds(frame, HOLDOUT_SEASONS)), spec
    )
    preds = result.predictions

    def scores(s: dict) -> dict:
        return {who: {"win": s[who]["win"].row(), "spread": s[who]["spread"].row()}
                for who in ("model", "vegas")}

    return {
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "scheme": "weekly walk-forward; each fold trains on games completed before that week's "
                  "first kickoff",
        "n_folds": preds["fold"].n_unique(),
        "n_games": preds.height,
        "spec": asdict(spec),
        "overall": scores(result.overall()),
        "by_season": {str(s): scores(v) for s, v in result.by_season().items()},
        "edge_vs_vegas": edge_vs_vegas(result),
        "calibration": {
            "model": calibration_table(preds["p_home"].to_numpy(), preds["home_win"].to_numpy()),
            "vegas": calibration_table(
                preds["p_home_vegas"].to_numpy(), preds["home_win"].to_numpy()
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-backtest", action="store_true", help="only refit and save the model")
    args = ap.parse_args(argv)

    frame = pl.read_parquet(PROCESSED_PATH)

    bundle = fit_final(frame)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"model  -> {MODEL_PATH}  n_train={bundle['n_train']}  "
          f"through {bundle['trained_through_game']}  code={bundle['code_version']}")

    if args.skip_backtest:
        return 0

    report = holdout_report(frame)
    BACKTEST_PATH.write_text(json.dumps(report, indent=2))
    m, v = report["overall"]["model"], report["overall"]["vegas"]
    print(f"backtest -> {BACKTEST_PATH}  {report['n_folds']} weekly folds, {report['n_games']} games")
    print(f"  model  acc {m['win']['accuracy']:.4f}  logloss {m['win']['log_loss']:.4f}  "
          f"brier {m['win']['brier']:.4f}  ece {m['win']['ece']:.4f}  "
          f"mae {m['spread']['mae']:.3f}  ats {m['spread']['ats_pct']:.4f}")
    print(f"  vegas  acc {v['win']['accuracy']:.4f}  logloss {v['win']['log_loss']:.4f}  "
          f"brier {v['win']['brier']:.4f}  ece {v['win']['ece']:.4f}  "
          f"mae {v['spread']['mae']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
