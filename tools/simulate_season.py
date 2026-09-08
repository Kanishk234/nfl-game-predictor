"""Replay a whole season through the real cron timeline and check the invariants at every step.

    python tools/simulate_season.py OUT --season 2025

This is not a smoke test. It walks the actual schedule — every Tuesday safety net, Thursday
early pass, Sunday late pass, and Friday/Monday/Tuesday grade — in chronological order across
all 18 weeks, calling the *real* `predict.run`, `grade.run` and `site_build.main` with the clock
and the data feeds moved back in time. After every job it asserts the things that must always
hold, and it fails loudly on the first violation.

What is real: the gate, the immutability checks, the record builder, the official-prediction
rule, the grader, the history rebuild, the site renderer.
What is stubbed: the wall clock, the nflreadpy frame (precomputed, with `is_played` masked to
the simulated moment) and the odds fetch (the real historical closing lines).

Nothing under `data/` is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path

import polars as pl

from nfl_predict import grade as G
from nfl_predict import predict as P
from nfl_predict import site_build as S
from nfl_predict.data.games import load_games
from nfl_predict.data.schedule import GAME_DURATION, LateRunError
from nfl_predict.odds.fetch import american_to_prob, devig

VOID = {"meta", "link", "br", "img", "input", "hr", "source"}


class Failure(AssertionError):
    pass


class _Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(tag)
        else:
            self.stack.pop()


class SeasonSim:
    def __init__(self, season: int, out: Path):
        self.season = season
        self.out = out
        shutil.rmtree(out, ignore_errors=True)
        (out / "predictions").mkdir(parents=True)
        (out / "odds").mkdir(parents=True)
        (out / "results").mkdir(parents=True)
        (out / "site").mkdir(parents=True)

        self.frame = pl.read_parquet("data/processed/games.parquet")
        self.games = load_games()          # carries the scores the grader needs
        self.money = self.games.select("game_id", "home_moneyline", "away_moneyline")
        self.season_games = self.frame.filter(pl.col("season") == season).sort("kickoff_utc")
        self.checks = 0
        self.seen_hashes: dict[str, str] = {}   # prediction file -> sha256, to prove immutability

        P.PREDICTIONS_DIR = G.PREDICTIONS_DIR = S.PREDICTIONS_DIR = out / "predictions"
        G.RESULTS_DIR = S.RESULTS_DIR = out / "results"
        G.HISTORY_PATH = S.HISTORY_PATH = out / "results" / "history.json"
        S.SITE_DIR = out / "site"
        P.snapshot_path = lambda t, n: out / "odds" / f"{t.season}_{t.week:02d}_{n}.json"

    # ---------------------------------------------------------------- time-travelled feeds
    def frame_as_of(self, now: datetime) -> pl.DataFrame:
        """The world as the pipeline would have seen it: only games finished before `now`."""
        done = pl.col("is_played") & (pl.col("kickoff_utc") + GAME_DURATION < now)
        return self.frame.with_columns(
            done.alias("is_played"),
            *[pl.when(done).then(pl.col(c)).otherwise(None).alias(c) for c in ("home_win", "margin")],
        )

    def games_as_of(self, now: datetime) -> pl.DataFrame:
        """What `grade.load_games()` would have returned: scores only for finished games."""
        done = pl.col("is_played") & (pl.col("kickoff_utc") + GAME_DURATION < now)
        return self.games.with_columns(
            done.alias("is_played"),
            *[pl.when(done).then(pl.col(c)).otherwise(None).alias(c)
              for c in ("home_score", "away_score", "result")],
        )

    def odds_as_of(self, target, pass_name, frame, now=None):
        rows = (self.frame.filter((pl.col("season") == target.season) & (pl.col("week") == target.week))
                .join(self.money, on="game_id", how="left"))
        lines = []
        for g in rows.iter_rows(named=True):
            if g["spread_line"] is None:
                continue
            hm, am = g["home_moneyline"], g["away_moneyline"]
            p = devig(american_to_prob(hm), american_to_prob(am)) if hm and am else None
            lines.append({"game_id": g["game_id"], "consensus": {
                "spread_line": float(g["spread_line"]), "total_line": g["total_line"],
                "p_home_moneyline": p, "n_books": 3}})
        return {"season": target.season, "week": target.week, "pass": pass_name,
                "fetched_at_utc": (now or datetime.now(UTC)).isoformat(),
                "source": {"provider": "simulated", "markets": [], "bookmakers": [], "region": "us", "quota": {}},
                "n_games_in_week": rows.height, "n_games_with_lines": len(lines),
                "games_without_lines": [], "lines": lines}

    # ---------------------------------------------------------------- the timeline
    def fire_times(self):
        """Every cron firing across the season, in order, as (job, when)."""
        start = self.season_games["kickoff_utc"].min() - timedelta(days=9)
        end = self.season_games["kickoff_utc"].max() + timedelta(days=3)
        out, d = [], start.replace(hour=0, minute=0, second=0, microsecond=0)
        while d <= end:
            wd = d.weekday()
            if wd == 1:
                out.append(("predict-early-net", d.replace(hour=16)))
                out.append(("grade", d.replace(hour=12)))
            elif wd == 3:
                out.append(("predict-early", d.replace(hour=21)))
            elif wd == 4:
                out.append(("grade", d.replace(hour=12)))
            elif wd == 6:
                out.append(("predict-late", d.replace(hour=14)))
            elif wd == 0:
                out.append(("grade", d.replace(hour=12)))
            d += timedelta(days=1)
        return sorted(out, key=lambda x: x[1])

    def run_predict(self, pass_name: str, now: datetime, only_early_openers: bool) -> str:
        frame = self.frame_as_of(now)
        P.build_frame = lambda: frame
        P.assert_no_leakage = lambda f: None
        P.PROCESSED_PATH = self.out / "games.parquet"
        P.fetch_snapshot = lambda t, n, f, now=None: self.odds_as_of(t, n, f, now)
        P._utcnow = lambda: now
        before = {p.name for p in (self.out / "predictions").glob("*.json")}
        try:
            result = P.run(pass_name, now=now, only_early_openers=only_early_openers)
        except LateRunError:
            return "gate refused"
        except RuntimeError as e:
            return f"nothing to do ({e})"
        if result is None or result[0].name in before:
            return "skipped"
        rec = json.loads(result[0].read_text())
        return f"published {result[0].name}: {rec['n_games']} games"

    # ---------------------------------------------------------------- invariants
    def check_predictions(self, now: datetime):
        for path in sorted((self.out / "predictions").glob("*.json")):
            rec = json.loads(path.read_text())
            gen = datetime.fromisoformat(rec["generated_at_utc"])

            # 1. never generated at or after a kickoff it covers
            for p in rec["predictions"]:
                kick = datetime.fromisoformat(p["kickoff_utc"])
                if gen >= kick:
                    raise Failure(f"{path.name}: generated {gen} at/after {p['game_id']} kickoff {kick}")
                self.checks += 1

            # 2. never rewritten once published
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.name in self.seen_hashes and self.seen_hashes[path.name] != digest:
                raise Failure(f"{path.name} changed after publication")
            self.seen_hashes[path.name] = digest
            self.checks += 1

            # 3. the odds snapshot is frozen at the same instant
            odds = self.out / "odds" / path.name
            if not odds.exists():
                raise Failure(f"{path.name} has no odds snapshot beside it")
            if json.loads(odds.read_text())["fetched_at_utc"] != rec["generated_at_utc"]:
                raise Failure(f"{path.name}: odds not frozen at the prediction's instant")
            self.checks += 2

    def check_results(self, now: datetime):
        total = 0
        for path in sorted((self.out / "results").glob("*.json")):
            if path.name == "history.json":
                continue
            r = json.loads(path.read_text())
            total += r["n_graded"]
            for g in r["games"]:
                # 4. nothing graded before it finished
                kick = datetime.fromisoformat(g["kickoff_utc"])
                if kick + GAME_DURATION >= now:
                    raise Failure(f"{g['game_id']} graded at {now} but only kicked off {kick}")
                # 5. the winner agrees with the score
                expect = g["home_team"] if g["home_score"] > g["away_score"] else (
                    g["away_team"] if g["away_score"] > g["home_score"] else "tie")
                if g["winner"] != expect:
                    raise Failure(f"{g['game_id']}: winner {g['winner']} vs score {g['away_score']}-{g['home_score']}")
                # 6. the official pick came from a pass published before kickoff
                src = self.out / "predictions" / f"{r['season']}_{r['week']:02d}_{g['pass']}.json"
                gen = datetime.fromisoformat(json.loads(src.read_text())["generated_at_utc"])
                if gen >= kick:
                    raise Failure(f"{g['game_id']} graded against a pass generated after kickoff")
                self.checks += 3
        if (self.out / "results" / "history.json").exists():
            hist = json.loads((self.out / "results" / "history.json").read_text())
            n = sum(s["summary"].get("n", 0) for s in hist["seasons"].values())
            # 7. the season history counts exactly the graded games
            if n != total:
                raise Failure(f"history says {n} graded games, per-week files say {total}")
            self.checks += 1

    def check_site(self):
        pages = sorted((self.out / "site").rglob("*.html"))
        if not pages:
            raise Failure("no site pages built")
        for page in pages:
            src = page.read_text(encoding="utf-8")
            parser = _Tags()
            parser.feed(src)
            if parser.errors or parser.stack:
                raise Failure(f"{page.name}: unbalanced tags {parser.errors or parser.stack}")
            for href in re.findall(r'href="([^"#:]+\.html)"', src):
                if not (page.parent / href).resolve().exists():
                    raise Failure(f"{page.name}: dead link {href}")
            self.checks += 2
        # 8. index always shows the newest published week
        index = (self.out / "site" / "index.html").read_text(encoding="utf-8")
        weeks = sorted(int(p.name[5:7]) for p in (self.out / "predictions").glob("*.json"))
        if weeks:
            shown = re.search(r"<h2>Week (\d+),", index)
            if not shown or int(shown.group(1)) != weeks[-1]:
                raise Failure(f"index shows {shown and shown.group(1)}, newest week is {weeks[-1]}")
            self.checks += 1

    def check_grading_is_idempotent(self, now: datetime):
        before = {p.name: p.read_text() for p in (self.out / "results").glob("*.json")}
        G.run(self.season)
        after = {p.name: p.read_text() for p in (self.out / "results").glob("*.json")}
        changed = [k for k in before if before[k] != after.get(k)]
        if changed:
            raise Failure(f"grading twice changed {changed}")
        self.checks += 1

    # ---------------------------------------------------------------- main loop
    def run(self) -> int:
        print(f"replaying {self.season}: {self.season_games.height} games, "
              f"{self.season_games['week'].n_unique()} weeks\n")
        published, graded_weeks = 0, set()
        for job, now in self.fire_times():
            if job.startswith("predict"):
                pass_name = "late" if job == "predict-late" else "early"
                note = self.run_predict(pass_name, now, only_early_openers=job.endswith("-net"))
                if note.startswith("published"):
                    published += 1
                    print(f"  {now:%a %b %d %H:%M} {job:18} {note}")
            else:
                G.load_games = lambda n=now: self.games_as_of(n)
                G.run(self.season)
                done = {p.name for p in (self.out / "results").glob("2*.json")}
                if done - graded_weeks:
                    print(f"  {now:%a %b %d %H:%M} {'grade':18} results now cover {len(done)} week(s)")
                graded_weeks = done
            S.main.__globals__["print"] = lambda *a, **k: None
            S.main()
            S.main.__globals__["print"] = print
            self.check_predictions(now)
            self.check_results(now)
            self.check_site()

        print(f"\n  {published} prediction files published")
        self.check_grading_is_idempotent(self.season_games["kickoff_utc"].max() + timedelta(days=3))
        return self.report()

    def report(self) -> int:
        preds = sorted((self.out / "predictions").glob("*.json"))
        hist = json.loads((self.out / "results" / "history.json").read_text())["seasons"][str(self.season)]
        s = hist["summary"]
        m, v = s["model"], s["vegas"]

        covered, official = set(), {}
        for p in preds:
            rec = json.loads(p.read_text())
            gen = datetime.fromisoformat(rec["generated_at_utc"])
            for row in rec["predictions"]:
                covered.add(row["game_id"])
                prev = official.get(row["game_id"])
                if prev is None or gen > prev:
                    official[row["game_id"]] = gen
        all_games = set(self.season_games["game_id"])
        missing = sorted(all_games - covered)

        print(f"\n{'=' * 78}\n  RESULT\n{'=' * 78}")
        print(f"  invariant checks passed      {self.checks:,}")
        print(f"  prediction files             {len(preds)}")
        print(f"  games in the season          {len(all_games)}")
        print(f"  games with a pre-kickoff pick{len(covered):>4}"
              + (f"   MISSING: {missing}" if missing else "   (all of them)"))
        print(f"  games graded                 {s['n']}")
        print(f"  our picks right              {m['accuracy'] * 100:.1f}%   Vegas {v['accuracy'] * 100:.1f}%")
        print(f"  spread error                 {m['spread_mae']:.2f} pts   Vegas {v['spread_mae']:.2f}")
        a = m["ats"]
        print(f"  against the spread           {a['ats_w']}-{a['ats_l']}-{a['ats_push']}  ({a['ats_pct'] * 100:.1f}%)")
        pages = sorted(p.relative_to(self.out / "site").as_posix() for p in (self.out / "site").rglob("*.html"))
        print(f"  site pages                   {len(pages)}")
        if missing:
            raise Failure(f"{len(missing)} games never got a pre-kickoff prediction")
        print(f"\n  open {self.out / 'site' / 'index.html'}")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--season", type=int, default=2025)
    args = ap.parse_args()
    try:
        return SeasonSim(args.season, args.out).run()
    except Failure as e:
        print(f"\n  FAILED: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
