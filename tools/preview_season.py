"""Replay a real season through the whole pipeline and render the site, at any moment in time.

    python tools/preview_season.py OUT --through-week 6
    python tools/preview_season.py OUT --through-week 2 --as-of 2025-09-12T18:00

Takes a season that has already happened, predicts each week walk-forward (the model only ever
sees games that had finished before that pass ran), freezes the real historical Vegas lines
beside each prediction, grades whatever has finished as of a chosen moment, and builds the site.

`--as-of` is the point of it: the live site spends most of its life *mid-week*, with Thursday
graded and Sunday still pending, and with two passes published for the same week. That state is
hard to reason about and easy to get wrong, so this makes it reproducible.

Everything is written under OUT. `data/` is never touched.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from nfl_predict import grade as G
from nfl_predict import site_build as S
from nfl_predict.data.games import load_games
from nfl_predict.data.schedule import GAME_DURATION
from nfl_predict.model.backtest import DEFAULT_SPEC
from nfl_predict.model.train import fit_final
from nfl_predict.odds.fetch import american_to_prob, devig

#: When each pass runs, matching .github/workflows/. Early: Thursday 21:00 UTC, but for a week
#: that opens earlier we use three hours before its first kickoff (what the Tuesday safety net
#: achieves). Late: Sunday 14:00 UTC.
LATE_PASS_HOUR = 14


def real_odds(row: dict) -> dict | None:
    """The historical closing line, in the shape a live odds snapshot uses."""
    if row["spread_line"] is None:
        return None
    hm, am = row.get("home_moneyline"), row.get("away_moneyline")
    p = devig(american_to_prob(hm), american_to_prob(am)) if hm and am else None
    return {"spread_line": float(row["spread_line"]), "total_line": row["total_line"],
            "p_home_moneyline": p, "p_home_from_spread": p, "n_books": 3}


def pass_times(week_games: pl.DataFrame) -> dict[str, datetime]:
    """When the early and late passes would have run for this week."""
    first = week_games["kickoff_utc"].min()
    early = first - timedelta(hours=3)
    sunday = None
    for k in week_games["kickoff_utc"].sort():
        if k.weekday() == 6:  # Sunday in UTC covers the 1pm ET slate onwards
            sunday = k.replace(hour=LATE_PASS_HOUR, minute=0, second=0, microsecond=0)
            break
    return {"early": early, **({"late": sunday} if sunday and sunday > early else {})}


def build_pass(frame: pl.DataFrame, money: pl.DataFrame, season: int, week: int,
               pass_name: str, generated: datetime) -> dict | None:
    """One prediction record, trained only on games finished before `generated`."""
    train = frame.filter(pl.col("is_played") & (pl.col("kickoff_utc") + GAME_DURATION < generated))
    rows = (frame.filter((pl.col("season") == season) & (pl.col("week") == week)
                         & (pl.col("kickoff_utc") > generated))
            .sort("kickoff_utc", "game_id").join(money, on="game_id", how="left"))
    if rows.is_empty():
        return None
    bundle = fit_final(train, DEFAULT_SPEC)
    X = rows.select(bundle["features"]).to_numpy().astype(float)
    p_home = bundle["classifier"].predict_proba(X)[:, 1]
    margin = bundle["regressor"].predict(X)
    preds = [{
        "game_id": g["game_id"], "kickoff_utc": g["kickoff_utc"].isoformat(),
        "home_team": g["home_team"], "away_team": g["away_team"],
        "p_home": round(float(p_home[i]), 4), "pred_margin": round(float(margin[i]), 2),
        "pick": g["home_team"] if p_home[i] >= 0.5 else g["away_team"],
        "vegas": real_odds(g),
        "feature_as_of_utc": g["as_of_utc"].isoformat() if g["as_of_utc"] else None,
    } for i, g in enumerate(rows.iter_rows(named=True))]
    first_kick = rows["kickoff_utc"].min()
    return {
        "season": season, "week": week, "pass": pass_name,
        "generated_at_utc": generated.isoformat(),
        "gate": {"earliest_kickoff_utc": first_kick.isoformat(),
                 "seconds_before_first_kickoff": int((first_kick - generated).total_seconds())},
        "model": {"learner": bundle["spec"]["learner"], "params": bundle["spec"]["params"],
                  "n_features": len(bundle["features"]), "feature_hash": "preview",
                  "n_train": bundle["n_train"], "trained_through_game": bundle["trained_through_game"],
                  "trained_through_utc": bundle["trained_through_utc"],
                  "trained_at_utc": bundle["trained_at_utc"], "code_version": bundle["code_version"]},
        "n_games": len(preds), "predictions": preds,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--through-week", type=int, default=6)
    ap.add_argument("--as-of", help="freeze time here (ISO, UTC); default = end of the last week")
    args = ap.parse_args()

    frame = pl.read_parquet("data/processed/games.parquet")
    money = load_games().select("game_id", "home_moneyline", "away_moneyline")
    weeks = list(range(1, args.through_week + 1))
    season_games = frame.filter((pl.col("season") == args.season) & pl.col("week").is_in(weeks))

    if args.as_of:
        as_of = datetime.fromisoformat(args.as_of)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
    else:
        as_of = season_games["kickoff_utc"].max() + GAME_DURATION + timedelta(hours=1)
    print(f"simulating {args.season} weeks {weeks[0]}-{weeks[-1]} as of {as_of:%a %b %d %H:%M} UTC\n")

    out = args.out
    shutil.rmtree(out, ignore_errors=True)
    (out / "predictions").mkdir(parents=True)
    (out / "results").mkdir(parents=True)

    for week in weeks:
        wg = frame.filter((pl.col("season") == args.season) & (pl.col("week") == week))
        for pass_name, when in pass_times(wg).items():
            if when >= as_of:
                continue  # that pass has not run yet
            rec = build_pass(frame, money, args.season, week, pass_name, when)
            if rec is None:
                continue
            (out / "predictions" / f"{args.season}_{week:02d}_{pass_name}.json").write_text(json.dumps(rec, indent=2))
            print(f"  wk{week:2} {pass_name:5} pass: {rec['n_games']:2} games, "
                  f"{rec['gate']['seconds_before_first_kickoff'] / 3600:5.1f}h before first kickoff, "
                  f"trained on {rec['model']['n_train']}")

    # Only games finished by `as_of` have results.
    finished = frame.with_columns(
        (pl.col("is_played") & (pl.col("kickoff_utc") + GAME_DURATION < as_of)).alias("is_played")
    ).join(load_games().select("game_id", "home_score", "away_score"), on="game_id", how="left")

    G.PREDICTIONS_DIR = S.PREDICTIONS_DIR = out / "predictions"
    G.RESULTS_DIR = S.RESULTS_DIR = out / "results"
    G.HISTORY_PATH = S.HISTORY_PATH = out / "results" / "history.json"
    S.SITE_DIR = out / "site"
    G.load_games = lambda: finished

    print("\ngrading:")
    G.run(args.season)
    print("\nbuilding:")
    S.main()

    index = (out / "site" / "index.html").read_text(encoding="utf-8")
    print(f"\n  index.html shows: {re.search(r'<h2>(.*?)</h2>', index).group(1)}")
    print(f"  week strip: {' | '.join(re.findall(r'>(Week \\d+|Season|This week)</a>', index))}")
    print(f"  open {out / 'site' / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
