"""The prediction pass. Refresh data, retrain, predict the coming week, freeze the odds beside it.

    python -m nfl_predict.predict --pass early     # Tuesday: the whole upcoming week
    python -m nfl_predict.predict --pass late      # Sunday: whatever has not kicked off yet

Writes two files that are immutable once written:

- data/predictions/<season>_<week>_<pass>.json
- data/odds/<season>_<week>_<pass>.json

The gate runs first: the pass computes the target week's earliest remaining kickoff and refuses
to proceed if it has passed. Every prediction row records the UTC time it was generated and the
model it came from, so a late or leaky prediction is provable after the fact.

"Retrain" means what CLAUDE.md says it means: refit the same model on the expanding set of
completed games. The Sunday pass retrains too, so Thursday's result is in Sunday's model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from nfl_predict.data.pipeline import PROCESSED_PATH, assert_no_leakage, build_frame
from nfl_predict.data.schedule import (
    WeekTarget,
    assert_before_kickoff,
    next_week_target,
)
from nfl_predict.model.backtest import DEFAULT_SPEC
from nfl_predict.model.baseline import SpreadToWinProb
from nfl_predict.model.train import fit_final
from nfl_predict.odds.fetch import fetch_snapshot, snapshot_path, write_snapshot

PREDICTIONS_DIR = Path("data/predictions")
PASSES = ("early", "late")


class PredictionExistsError(FileExistsError):
    """Predictions are immutable. A corrected run gets a new file, never an overwrite."""


def prediction_path(target: WeekTarget, pass_name: str) -> Path:
    return PREDICTIONS_DIR / f"{target.season}_{target.week:02d}_{pass_name}.json"


def games_to_predict(frame: pl.DataFrame, target: WeekTarget, now: datetime) -> pl.DataFrame:
    """The target week's games that have not kicked off. The early pass normally gets the whole
    week; the late pass gets the Sunday/Monday slate, and games already played keep the early
    pass's prediction."""
    return frame.filter(
        (pl.col("season") == target.season)
        & (pl.col("week") == target.week)
        & (pl.col("kickoff_utc") > now)
    ).sort("kickoff_utc", "game_id")


def make_predictions(bundle: dict, frame: pl.DataFrame, rows: pl.DataFrame, odds: dict) -> list[dict]:
    X = rows.select(bundle["features"]).to_numpy().astype(float)
    p_home = bundle["classifier"].predict_proba(X)[:, 1]
    margin = bundle["regressor"].predict(X)

    # The backtest's Vegas win probability was spread-derived; keep that alongside the live
    # moneyline so the two eras of the track record stay comparable.
    played = frame.filter(pl.col("is_played"))
    spread_to_prob = SpreadToWinProb().fit(played["spread_line"].to_numpy(), played["home_win"].to_numpy())
    lines = {line["game_id"]: line["consensus"] for line in odds["lines"]}

    out = []
    for i, g in enumerate(rows.iter_rows(named=True)):
        c = lines.get(g["game_id"])
        vegas = None
        if c and c["spread_line"] is not None:
            vegas = {
                "spread_line": c["spread_line"],
                "total_line": c["total_line"],
                "p_home_moneyline": c["p_home_moneyline"],
                "p_home_from_spread": float(spread_to_prob.predict([c["spread_line"]])[0]),
                "n_books": c["n_books"],
            }
        out.append({
            "game_id": g["game_id"],
            "kickoff_utc": g["kickoff_utc"].isoformat(),
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "p_home": round(float(p_home[i]), 4),
            "pred_margin": round(float(margin[i]), 2),
            "pick": g["home_team"] if p_home[i] >= 0.5 else g["away_team"],
            "vegas": vegas,
            "feature_as_of_utc": g["as_of_utc"].isoformat() if g["as_of_utc"] else None,
        })
    return out


def build_record(target: WeekTarget, pass_name: str, now: datetime, bundle: dict, predictions: list[dict]) -> dict:
    feature_hash = hashlib.sha256(",".join(bundle["features"]).encode()).hexdigest()[:12]
    return {
        "season": target.season,
        "week": target.week,
        "pass": pass_name,
        "generated_at_utc": now.isoformat(),
        "gate": {
            "earliest_kickoff_utc": min(p["kickoff_utc"] for p in predictions) if predictions else None,
            "seconds_before_first_kickoff": int(target.deadline_gap(now).total_seconds()),
        },
        "model": {
            "learner": bundle["spec"]["learner"],
            "params": bundle["spec"]["params"],
            "n_features": len(bundle["features"]),
            "feature_hash": feature_hash,
            "n_train": bundle["n_train"],
            "trained_through_game": bundle["trained_through_game"],
            "trained_through_utc": bundle["trained_through_utc"],
            "trained_at_utc": bundle["trained_at_utc"],
            "code_version": bundle["code_version"],
        },
        "n_games": len(predictions),
        "predictions": predictions,
    }


def write_record(record: dict, path: Path) -> Path:
    if path.exists():
        raise PredictionExistsError(
            f"{path} already exists; predictions are immutable once written (see CLAUDE.md)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return path


def run(pass_name: str, now: datetime | None = None, retrain: bool = True) -> tuple[Path, Path]:
    now = now or datetime.now(UTC)

    frame = build_frame()
    assert_no_leakage(frame)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(PROCESSED_PATH)

    target = next_week_target(frame, now)
    assert_before_kickoff(target, now)
    pred_path, odds_path = prediction_path(target, pass_name), snapshot_path(target, pass_name)
    for p in (pred_path, odds_path):
        if p.exists():
            raise PredictionExistsError(f"{p} already exists; a pass is never re-run in place")

    if retrain:
        bundle = fit_final(frame, DEFAULT_SPEC)
    else:
        import joblib

        from nfl_predict.model.train import MODEL_PATH
        bundle = joblib.load(MODEL_PATH)

    rows = games_to_predict(frame, target, now)
    if rows.is_empty():
        raise RuntimeError(f"no games left to predict in {target.season} week {target.week}")
    missing = [c for c in bundle["features"] if rows[c].null_count() == rows.height]
    if missing:
        raise RuntimeError(f"features entirely null for the target week: {missing}")

    # Same instant: the baseline is frozen with the prediction, never before or after.
    stamp = datetime.now(UTC)
    odds = fetch_snapshot(target, pass_name, frame, now=stamp)
    predictions = make_predictions(bundle, frame, rows, odds)
    record = build_record(target, pass_name, stamp, bundle, predictions)

    write_snapshot(odds, odds_path)
    write_record(record, pred_path)
    return pred_path, odds_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pass", dest="pass_name", choices=PASSES, required=True)
    ap.add_argument("--no-retrain", action="store_true", help="reuse data/models/latest.joblib")
    args = ap.parse_args(argv)

    pred_path, odds_path = run(args.pass_name, retrain=not args.no_retrain)
    record = json.loads(pred_path.read_text())
    print(f"wrote {pred_path} and {odds_path}")
    print(f"  {record['season']} week {record['week']} ({record['pass']}): {record['n_games']} games, "
          f"generated {record['gate']['seconds_before_first_kickoff'] / 3600:.1f}h before first kickoff, "
          f"model trained through {record['model']['trained_through_game']} ({record['model']['code_version']})")
    for p in record["predictions"]:
        v = p["vegas"]
        vs = f"vegas {v['spread_line']:+.1f} / {v['p_home_moneyline']:.3f}" if v else "no line"
        print(f"  {p['game_id']:18} home {p['p_home']:.3f}  margin {p['pred_margin']:+.1f}  | {vs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
