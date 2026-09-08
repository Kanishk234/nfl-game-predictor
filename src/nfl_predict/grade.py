"""Grade published predictions against final scores and the frozen baseline.

    python -m nfl_predict.grade            # grade every week with a prediction file
    python -m nfl_predict.grade --week 3   # one week (current season)

Writes `data/results/<season>_<week>.json` per week and regenerates `data/results/history.json`
from those files. Results are facts, so a week's file is rewritten as more of its games finish
(`complete` says whether all of them have). The history is rebuilt from the per-week files on
every run rather than appended to, which is what makes a second run idempotent by construction.

Which prediction counts. A game can appear in both passes. Its **official** prediction is the
latest pass generated before that game's kickoff: the early pass for a Wednesday/Thursday game,
the late pass for the Sunday/Monday slate. Every pass is graded and reported on its own too.

Predictions are never modified here. This module only reads them.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from nfl_predict.data.games import load_games
from nfl_predict.model.metrics import (
    ATS_BREAKEVEN,
    calibration_table,
    spread_metrics,
    win_prob_metrics,
)
from nfl_predict.predict import PREDICTIONS_DIR

RESULTS_DIR = Path("data/results")
HISTORY_PATH = RESULTS_DIR / "history.json"


def load_predictions(season: int, week: int) -> dict[str, dict]:
    """{pass_name: record} for every published pass of the week."""
    out = {}
    for path in sorted(PREDICTIONS_DIR.glob(f"{season}_{week:02d}_*.json")):
        rec = json.loads(path.read_text())
        out[rec["pass"]] = rec
    return out


def official_predictions(passes: dict[str, dict]) -> dict[str, tuple[str, dict]]:
    """game_id -> (pass_name, prediction row): the latest pass generated before that kickoff."""
    chosen: dict[str, tuple[str, dict, datetime]] = {}
    for pass_name, rec in passes.items():
        generated = datetime.fromisoformat(rec["generated_at_utc"])
        for row in rec["predictions"]:
            kickoff = datetime.fromisoformat(row["kickoff_utc"])
            if generated >= kickoff:
                continue  # a late prediction is not a prediction
            prev = chosen.get(row["game_id"])
            if prev is None or generated > prev[2]:
                chosen[row["game_id"]] = (pass_name, row, generated)
    return {g: (p, r) for g, (p, r, _) in chosen.items()}


def grade_rows(rows: list[tuple[str, dict]], finals: pl.DataFrame) -> list[dict]:
    """Per-game grades for games that have finished."""
    by_id = {r["game_id"]: r for r in finals.iter_rows(named=True)}
    graded = []
    for pass_name, p in rows:
        g = by_id.get(p["game_id"])
        if g is None or not g["is_played"]:
            continue
        margin = float(g["home_score"] - g["away_score"])
        home_win = int(margin > 0)
        p_home = p["p_home"]
        v = p.get("vegas")
        entry = {
            "game_id": p["game_id"], "pass": pass_name, "kickoff_utc": p["kickoff_utc"],
            "home_team": p["home_team"], "away_team": p["away_team"],
            "home_score": int(g["home_score"]), "away_score": int(g["away_score"]),
            "margin": margin, "home_win": home_win,
            # Plain-English verdict beside the numbers: who we said, who actually won.
            "winner": p["home_team"] if margin > 0 else (p["away_team"] if margin < 0 else "tie"),
            "pick": p["pick"],
            "model": {
                "p_home": p_home, "pred_margin": p["pred_margin"], "pick": p["pick"],
                "correct": int((p_home >= 0.5) == (home_win == 1)) if margin != 0 else None,
                "brier": (p_home - home_win) ** 2,
                "abs_error": abs(p["pred_margin"] - margin),
            },
            "vegas": None,
        }
        if v:
            pv = v["p_home_moneyline"] if v.get("p_home_moneyline") is not None else v["p_home_from_spread"]
            line = v["spread_line"]
            entry["vegas"] = {
                "p_home": pv, "spread_line": line,
                "correct": int((pv >= 0.5) == (home_win == 1)) if margin != 0 else None,
                "brier": (pv - home_win) ** 2,
                "abs_error": abs(line - margin),
            }
            # Our side against their line: home if we say more than the line, else away.
            take_home = p["pred_margin"] > line
            if margin == line:
                ats = "push"
            else:
                ats = "win" if take_home == (margin > line) else "loss"
            entry["model"]["ats"] = ats
        graded.append(entry)
    return graded


def summarise(graded: list[dict]) -> dict:
    """Aggregate metrics over graded games, for the model and (where lines exist) Vegas."""
    if not graded:
        return {"n": 0}
    y = np.array([g["home_win"] for g in graded], float)
    margin = np.array([g["margin"] for g in graded], float)
    pm = np.array([g["model"]["p_home"] for g in graded], float)
    mm = np.array([g["model"]["pred_margin"] for g in graded], float)
    out = {"n": len(graded), "model": win_prob_metrics(pm, y).row()}
    out["model"]["spread_mae"] = float(np.abs(mm - margin).mean())
    with_v = [g for g in graded if g["vegas"]]
    if with_v:
        yv = np.array([g["home_win"] for g in with_v], float)
        pv = np.array([g["vegas"]["p_home"] for g in with_v], float)
        line = np.array([g["vegas"]["spread_line"] for g in with_v], float)
        mv = np.array([g["margin"] for g in with_v], float)
        mmv = np.array([g["model"]["pred_margin"] for g in with_v], float)
        out["vegas"] = win_prob_metrics(pv, yv).row()
        out["vegas"]["spread_mae"] = float(np.abs(line - mv).mean())
        out["model"]["ats"] = spread_metrics(mmv, mv, line).row()
        out["model"]["ats"]["breakeven"] = ATS_BREAKEVEN
        out["n_with_line"] = len(with_v)
    return out


def grade_week(season: int, week: int, games: pl.DataFrame) -> dict | None:
    passes = load_predictions(season, week)
    if not passes:
        return None
    week_games = games.filter((pl.col("season") == season) & (pl.col("week") == week))
    official = official_predictions(passes)
    graded = grade_rows(list(official.values()), week_games)
    per_pass = {
        name: summarise(grade_rows([(name, r) for r in rec["predictions"]], week_games))
        for name, rec in passes.items()
    }
    n_week = week_games.height
    return {
        "season": season, "week": week,
        "graded_at_utc": datetime.now().astimezone().isoformat(),
        "n_games": n_week,
        "n_predicted": len(official),
        "n_graded": len(graded),
        "complete": len(graded) == n_week and n_week > 0,
        "passes": sorted(passes),
        "official_rule": "latest pass generated before each game's kickoff",
        "summary": summarise(graded),
        "by_pass": per_pass,
        "games": graded,
    }


def result_path(season: int, week: int) -> Path:
    return RESULTS_DIR / f"{season}_{week:02d}.json"


def rebuild_history() -> dict:
    """Season-to-date record from the per-week files. Regenerated, never appended."""
    weeks = []
    all_games: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("*_*.json")):
        if path.name == HISTORY_PATH.name:
            continue
        r = json.loads(path.read_text())
        weeks.append({k: r[k] for k in ("season", "week", "n_games", "n_graded", "complete", "summary")})
        all_games.extend(r["games"])
    by_season = {}
    for s in sorted({w["season"] for w in weeks}):
        games_s = [g for g in all_games if g["game_id"].startswith(f"{s}_")]
        entry = {"summary": summarise(games_s), "weeks": [w for w in weeks if w["season"] == s]}
        if games_s:
            y = np.array([g["home_win"] for g in games_s], float)
            entry["calibration"] = {
                "model": calibration_table(np.array([g["model"]["p_home"] for g in games_s]), y),
            }
            with_v = [g for g in games_s if g["vegas"]]
            if with_v:
                entry["calibration"]["vegas"] = calibration_table(
                    np.array([g["vegas"]["p_home"] for g in with_v]),
                    np.array([g["home_win"] for g in with_v], float),
                )
        by_season[str(s)] = entry
    return {"rebuilt_at_utc": datetime.now().astimezone().isoformat(), "seasons": by_season}


def run(season: int | None = None, week: int | None = None) -> list[Path]:
    games = load_games()
    if season is None:
        season = int(games.filter(pl.col("is_played"))["season"].max())
    targets = (
        [(season, week)] if week is not None
        else sorted({(int(p.name[:4]), int(p.name[5:7])) for p in PREDICTIONS_DIR.glob("*_*_*.json")})
    )
    written = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for s, w in targets:
        result = grade_week(s, w, games)
        if result is None:
            continue
        if result["n_graded"] == 0:
            print(f"{s} week {w:02d}: no completed games yet")
            continue
        path = result_path(s, w)
        path.write_text(json.dumps(result, indent=2))
        written.append(path)
        m = result["summary"]
        status = "complete" if result["complete"] else f"{result['n_graded']}/{result['n_games']} played"
        if m["n"]:
            v = m.get("vegas", {})
            print(f"{s} week {w:02d} ({status}): model {m['model']['accuracy']:.3f} acc, "
                  f"brier {m['model']['brier']:.3f}, mae {m['model']['spread_mae']:.2f}"
                  + (f" | vegas {v['accuracy']:.3f} acc, brier {v['brier']:.3f}, mae {v['spread_mae']:.2f}" if v else "")
                  + (f" | ats {m['model']['ats']['ats_w']}-{m['model']['ats']['ats_l']}-{m['model']['ats']['ats_push']}" if "ats" in m["model"] else ""))
    HISTORY_PATH.write_text(json.dumps(rebuild_history(), indent=2))
    written.append(HISTORY_PATH)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    args = ap.parse_args(argv)
    for p in run(args.season, args.week):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
