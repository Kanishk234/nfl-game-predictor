"""Dress rehearsal: replay two real, completed weeks through the whole pipeline.

Predict them as if they had not happened (walk-forward, so the model only sees earlier games),
freeze the real historical Vegas lines beside the predictions, grade them against the real final
scores, and render the site. Everything is written to a scratch directory: data/ is untouched.

This is what the live site will look like once Week 1 is graded.
"""
import json
import shutil
import sys
from datetime import timedelta
from pathlib import Path

import polars as pl

from nfl_predict import grade as G
from nfl_predict import predict as P
from nfl_predict import site_build as S
from nfl_predict.data.games import load_games
from nfl_predict.model.backtest import DEFAULT_SPEC, weekly_folds
from nfl_predict.model.train import fit_final

SEASON, WEEKS = 2025, (1, 2)
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "preview")
shutil.rmtree(OUT, ignore_errors=True)
(OUT / "predictions").mkdir(parents=True)
(OUT / "results").mkdir(parents=True)
(OUT / "site").mkdir(parents=True)

frame = pl.read_parquet("data/processed/games.parquet")
games = load_games()
money = games.select("game_id", "home_moneyline", "away_moneyline")

from nfl_predict.odds.fetch import american_to_prob, devig


def odds_for(row: dict) -> dict | None:
    """The real historical line, in the shape the live odds snapshot uses."""
    if row["spread_line"] is None:
        return None
    hm, am = row["home_moneyline"], row["away_moneyline"]
    p = devig(american_to_prob(hm), american_to_prob(am)) if hm and am else None
    return {"spread_line": float(row["spread_line"]), "total_line": row["total_line"],
            "p_home_moneyline": p, "p_home_from_spread": p, "n_books": 3}


folds = {int(f.name.split("w")[1]): f for f in weekly_folds(frame, (SEASON,))}
for week in WEEKS:
    fold = folds[week]
    bundle = fit_final(fold.train, DEFAULT_SPEC)          # trained only on earlier games
    rows = fold.test.sort("kickoff_utc", "game_id").join(money, on="game_id", how="left")
    X = rows.select(bundle["features"]).to_numpy().astype(float)
    p_home = bundle["classifier"].predict_proba(X)[:, 1]
    margin = bundle["regressor"].predict(X)

    first_kick = rows["kickoff_utc"].min()
    generated = first_kick - timedelta(hours=3)           # a realistic Thursday-pass lead time
    preds = []
    for i, g in enumerate(rows.iter_rows(named=True)):
        preds.append({
            "game_id": g["game_id"], "kickoff_utc": g["kickoff_utc"].isoformat(),
            "home_team": g["home_team"], "away_team": g["away_team"],
            "p_home": round(float(p_home[i]), 4), "pred_margin": round(float(margin[i]), 2),
            "pick": g["home_team"] if p_home[i] >= 0.5 else g["away_team"],
            "vegas": odds_for(g),
            "feature_as_of_utc": g["as_of_utc"].isoformat() if g["as_of_utc"] else None,
        })
    record = {
        "season": SEASON, "week": week, "pass": "early",
        "generated_at_utc": generated.isoformat(),
        "gate": {"earliest_kickoff_utc": first_kick.isoformat(),
                 "seconds_before_first_kickoff": int((first_kick - generated).total_seconds())},
        "model": {k: bundle[k] for k in ("n_train", "trained_through_game", "trained_through_utc",
                                         "trained_at_utc", "code_version")}
                 | {"learner": bundle["spec"]["learner"], "params": bundle["spec"]["params"],
                    "n_features": len(bundle["features"]), "feature_hash": "preview"},
        "n_games": len(preds), "predictions": preds,
    }
    (OUT / "predictions" / f"{SEASON}_{week:02d}_early.json").write_text(json.dumps(record, indent=2))
    print(f"  wrote a {SEASON} week {week} prediction: {len(preds)} games, "
          f"model trained on {bundle['n_train']} games through {bundle['trained_through_game']}")

# --- grade them against the real results, then render ---------------------------------------
P.PREDICTIONS_DIR = G.PREDICTIONS_DIR = S.PREDICTIONS_DIR = OUT / "predictions"
G.RESULTS_DIR = S.RESULTS_DIR = OUT / "results"
G.HISTORY_PATH = S.HISTORY_PATH = OUT / "results" / "history.json"
S.SITE_DIR = OUT / "site"

print("\ngrading:")
G.run(SEASON)
print("\nbuilding the site:")
S.main()

hist = json.loads((OUT / "results" / "history.json").read_text())["seasons"][str(SEASON)]["summary"]
m, v = hist["model"], hist["vegas"]
print(f"\n  {SEASON} weeks {WEEKS[0]}-{WEEKS[-1]}: we picked {m['accuracy']*100:.1f}% right, "
      f"Vegas {v['accuracy']*100:.1f}%.  spread error {m['spread_mae']:.1f} vs {v['spread_mae']:.1f} pts.  "
      f"ATS {m['ats']['ats_w']}-{m['ats']['ats_l']}-{m['ats']['ats_push']}")
print(f"  preview written to {OUT / 'site'}")
