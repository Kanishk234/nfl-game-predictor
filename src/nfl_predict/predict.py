"""The prediction pass. Refresh data, retrain, predict the coming week, freeze the odds beside it.

    python -m nfl_predict.predict --pass early     # Thursday: everything not yet kicked off
    python -m nfl_predict.predict --pass late      # Sunday: whatever is still ahead

Writes two files that are immutable once written:

- data/predictions/<season>_<week>_<pass>.json
- data/odds/<season>_<week>_<pass>.json

The gate is per game and enforced twice: the pass only ever selects games whose kickoff is
still ahead, and it refuses to run at all once every game in the week has started. Every
prediction row records the UTC time it was generated, so the grader can ignore anything written
after its game began. A week already published is a no-op, not an error: files are immutable.

"Retrain" means what CLAUDE.md says it means: refit the same model on the expanding set of
completed games. Every pass retrains, so the Sunday pass has Thursday's result in its model and
the Thursday pass has all of the previous week.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from nfl_predict.data.pipeline import PROCESSED_PATH, assert_no_leakage, build_frame
from nfl_predict.data.schedule import (
    WeekTarget,
    assert_before_kickoff,
    next_scheduled_early_pass,
    next_week_target,
)
from nfl_predict.model.backtest import DEFAULT_SPEC
from nfl_predict.model.baseline import SpreadToWinProb
from nfl_predict.model.train import fit_final
from nfl_predict.odds.fetch import fetch_snapshot, snapshot_path, write_snapshot

PREDICTIONS_DIR = Path("data/predictions")
PASSES = ("early", "late")

#: How close the next kickoff must be for the late pass to be a *late* pass.
#:
#: The Sunday cron fires every Sunday, including the ones before a week has begun — the Sunday
#: before the season opener, and the Sunday inside a gap between the regular season and the
#: playoffs. Without this, that firing targets the coming week and burns its late slot with a
#: prediction made eight days early; the real Sunday-morning refresh then finds the file already
#: published and does nothing. A late pass only makes sense once its games are hours away.
LATE_PASS_LEAD_LIMIT = timedelta(hours=24)


def _utcnow() -> datetime:
    """Wall clock, as a seam. Production always uses the real clock; the season simulator in
    tools/ replaces this so a replayed pass stamps itself with the time it is pretending to be."""
    return datetime.now(UTC)


class PredictionExistsError(FileExistsError):
    """Predictions are immutable. A corrected run gets a new file, never an overwrite."""


def prediction_path(target: WeekTarget, pass_name: str) -> Path:
    return PREDICTIONS_DIR / f"{target.season}_{target.week:02d}_{pass_name}.json"


def games_to_predict(frame: pl.DataFrame, target: WeekTarget, now: datetime) -> pl.DataFrame:
    """The target week's games that have not kicked off.

    The early (Thursday) pass gets everything still ahead of it; the late (Sunday) pass gets
    whatever remains, which is the Sunday/Monday slate. Games already played keep the earlier
    pass's prediction, and the grader treats the latest pass before each kickoff as official."""
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


def run(pass_name: str, now: datetime | None = None, retrain: bool = True,
        only_early_openers: bool = False) -> tuple[Path, Path] | None:
    now = now or _utcnow()

    frame = build_frame()
    assert_no_leakage(frame)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(PROCESSED_PATH)

    target = next_week_target(frame, now)
    assert_before_kickoff(target, now)

    if pass_name == "late" and target.earliest_kickoff - now > LATE_PASS_LEAD_LIMIT:
        print(f"{target.season} week {target.week} does not start for "
              f"{(target.earliest_kickoff - now).total_seconds() / 3600:.0f}h; a late pass now would "
              "not be late. Leaving the slot for the Sunday of that week.")
        return None

    if only_early_openers and target.earliest_kickoff > next_scheduled_early_pass(now):
        # The safety-net run. This week opens after the regular Thursday pass, so that pass
        # will cover all of it with a fresher model; publishing now would only burn the slot.
        print(f"{target.season} week {target.week} opens {target.earliest_kickoff.isoformat()}, "
              f"after the next scheduled pass; leaving it to Thursday")
        return None
    pred_path, odds_path = prediction_path(target, pass_name), snapshot_path(target, pass_name)
    if pred_path.exists():
        # Already published (a manual run, or a re-triggered job). Immutability means there is
        # nothing to do, and nothing to do is not a failure.
        print(f"{pred_path} already published; leaving it untouched")
        return pred_path, odds_path
    if odds_path.exists():
        raise PredictionExistsError(f"{odds_path} exists without its prediction; refusing to guess")

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
    stamp = _utcnow()
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
    ap.add_argument("--only-early-openers", action="store_true",
                    help="publish only if the week opens before the next scheduled Thursday pass "
                         "(the Tuesday safety net for Thanksgiving, Christmas and Wednesday openers)")
    args = ap.parse_args(argv)

    paths = run(args.pass_name, retrain=not args.no_retrain, only_early_openers=args.only_early_openers)
    if paths is None:
        return 0
    pred_path, odds_path = paths
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
