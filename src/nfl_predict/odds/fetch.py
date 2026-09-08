"""Fetch current NFL lines from The Odds API and freeze them as a snapshot.

Run as `python -m nfl_predict.odds.fetch --pass early` to write
`data/odds/<season>_<week>_<pass>.json` for the upcoming week. The prediction pass (Phase 4)
calls `fetch_snapshot` itself so the baseline is frozen at the same instant as the prediction;
the CLI exists for on-demand use and for checking the plumbing.

Secret handling, deliberately strict (see the project's security rule):

- The key is read from the `ODDS_API_KEY` environment variable only. Locally a gitignored
  `.env` is loaded if the variable is unset; in CI it arrives from an Actions secret.
- It is sent as a query parameter because that is the only form the API accepts, so every
  string that could carry it back out (exception text, response URL, log lines) goes through
  `scrub` before leaving this module.
- It is never written to the snapshot. The snapshot records request *metadata* (markets,
  bookmakers, quota remaining) and nothing else about the request.

Free-tier budget: one request per market per region. Three markets, one region, so each
snapshot costs 3 of the 500 monthly credits. Two passes a week is ~25 a month.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import UTC, datetime, timedelta
from pathlib import Path

import nflreadpy as nfl
import polars as pl
import requests

from nfl_predict.data.games import load_games
from nfl_predict.data.schedule import WeekTarget, next_week_target
from nfl_predict.retry import with_retries

ODDS_DIR = Path("data/odds")
API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
BOOKMAKERS = ("draftkings", "fanduel", "betmgm")
MARKETS = ("spreads", "h2h", "totals")
#: The API keys events by full team name; the schedule uses abbreviations.
_ENV_FILE = Path(".env")


class OddsFetchError(RuntimeError):
    """The odds could not be fetched. Not fatal to a prediction: see predict.run."""


class OddsPermanentError(OddsFetchError):
    """A failure retrying cannot fix: a bad key, an exhausted quota, a malformed request."""


class SnapshotExistsError(FileExistsError):
    """Snapshots are immutable. A second run for the same pass must not overwrite the first."""


def load_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY")
    if not key and _ENV_FILE.is_file():
        for line in _ENV_FILE.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not key:
        raise OddsFetchError(
            "ODDS_API_KEY is not set (export it, or put ODDS_API_KEY=... in a gitignored .env)"
        )
    return key


def scrub(text: str, key: str) -> str:
    """Remove the key from any string that might be logged or raised."""
    return text.replace(key, "***") if key else text


def team_name_map(valid_abbrs: set[str]) -> dict[str, str]:
    """Full name -> abbreviation, restricted to codes the schedule actually uses.

    `load_teams()` carries historical duplicates (the Rams appear as both LA and LAR), and a
    plain dict would keep whichever came last -- which is how a Week 1 game silently failed to
    match. Filtering to the schedule's own codes makes the map unambiguous.
    """
    teams = nfl.load_teams().select("team_name", "team_abbr").filter(pl.col("team_abbr").is_in(list(valid_abbrs)))
    return dict(teams.iter_rows())


def american_to_prob(price: float) -> float:
    """Implied probability of an American moneyline, vig included."""
    price = float(price)
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def devig(p_home: float, p_away: float) -> float:
    """Home win probability with the overround removed proportionally."""
    return p_home / (p_home + p_away)


def _request(key: str) -> tuple[list[dict], dict]:
    params = {
        "apiKey": key, "regions": "us", "markets": ",".join(MARKETS),
        "oddsFormat": "american", "bookmakers": ",".join(BOOKMAKERS),
    }
    try:
        resp = requests.get(API_URL, params=params, timeout=30)
    except requests.RequestException as exc:  # never let the URL with the key escape
        raise OddsFetchError(f"odds request failed: {scrub(str(exc), key)}") from None
    if resp.status_code != 200:
        # 401/403 = bad key, 422 = bad request, 429 = quota gone. Retrying cannot help.
        cls = OddsPermanentError if resp.status_code in (401, 403, 422, 429) else OddsFetchError
        raise cls(f"odds API returned {resp.status_code}: {scrub(resp.text[:300], key)}")
    quota = {
        "requests_used": resp.headers.get("x-requests-used"),
        "requests_remaining": resp.headers.get("x-requests-remaining"),
        "requests_last": resp.headers.get("x-requests-last"),
    }
    return resp.json(), quota


def _parse_event(event: dict, abbr: dict[str, str]) -> dict | None:
    """One event -> per-bookmaker lines plus a consensus, all from the home team's side."""
    home_name, away_name = event["home_team"], event["away_team"]
    if home_name not in abbr or away_name not in abbr:
        return None
    books = {}
    for bk in event.get("bookmakers", []):
        entry: dict = {"last_update": None}
        for market in bk.get("markets", []):
            entry["last_update"] = market.get("last_update") or entry["last_update"]
            by_name = {o["name"]: o for o in market["outcomes"]}
            if market["key"] == "spreads" and home_name in by_name and away_name in by_name:
                entry["spread_home"] = float(by_name[home_name]["point"])
                entry["spread_home_price"] = by_name[home_name]["price"]
                entry["spread_away_price"] = by_name[away_name]["price"]
            elif market["key"] == "h2h" and home_name in by_name and away_name in by_name:
                entry["moneyline_home"] = by_name[home_name]["price"]
                entry["moneyline_away"] = by_name[away_name]["price"]
            elif market["key"] == "totals" and "Over" in by_name:
                entry["total"] = float(by_name["Over"]["point"])
                entry["over_price"] = by_name["Over"]["price"]
                entry["under_price"] = by_name["Under"]["price"] if "Under" in by_name else None
        if len(entry) > 1:
            books[bk["key"]] = entry

    def median_of(field: str) -> float | None:
        vals = [b[field] for b in books.values() if field in b]
        return float(statistics.median(vals)) if vals else None

    probs = [
        devig(american_to_prob(b["moneyline_home"]), american_to_prob(b["moneyline_away"]))
        for b in books.values() if "moneyline_home" in b
    ]
    return {
        "event_id": event["id"],
        "commence_time": event["commence_time"],
        "home_team": abbr[home_name],
        "away_team": abbr[away_name],
        "books": books,
        "consensus": {
            # Positive = home favoured, matching nflreadpy's `spread_line` sign convention.
            "spread_line": -median_of("spread_home") if median_of("spread_home") is not None else None,
            "total_line": median_of("total"),
            "p_home_moneyline": float(statistics.median(probs)) if probs else None,
            "n_books": len(books),
        },
    }


def match_to_schedule(lines: list[dict], week_games: pl.DataFrame) -> list[dict]:
    """Attach nflverse game_ids. Match on teams, then on kickoff within a day (rescheduled
    games move; the teams do not)."""
    out = []
    by_teams = {(r["home_team"], r["away_team"]): r for r in week_games.iter_rows(named=True)}
    for line in lines:
        g = by_teams.get((line["home_team"], line["away_team"]))
        if g is None:
            continue
        commence = datetime.fromisoformat(line["commence_time"])
        if abs(commence - g["kickoff_utc"]) > timedelta(days=1):
            continue
        out.append({"game_id": g["game_id"], "kickoff_utc": g["kickoff_utc"].isoformat(), **line})
    return sorted(out, key=lambda r: (r["kickoff_utc"], r["game_id"]))


def snapshot_path(target: WeekTarget, pass_name: str) -> Path:
    return ODDS_DIR / f"{target.season}_{target.week:02d}_{pass_name}.json"


def fetch_snapshot(target: WeekTarget, pass_name: str, games: pl.DataFrame, now: datetime | None = None) -> dict:
    """Fetch, parse and match lines for the target week. Does not write."""
    key = load_api_key()
    now = now or datetime.now(UTC)
    events, quota = with_retries(
        lambda: _request(key), what="odds fetch",
        retry_on=(OddsFetchError, OSError), give_up_on=(OddsPermanentError,),
    )
    week_games = games.filter((pl.col("season") == target.season) & (pl.col("week") == target.week))
    abbr = team_name_map(set(week_games["home_team"]) | set(week_games["away_team"]))
    parsed = [p for p in (_parse_event(e, abbr) for e in events) if p is not None]
    lines = match_to_schedule(parsed, week_games)
    missing = sorted(set(week_games["game_id"]) - {r["game_id"] for r in lines})
    return {
        "season": target.season,
        "week": target.week,
        "pass": pass_name,
        "fetched_at_utc": now.isoformat(),
        "source": {"provider": "the-odds-api.com", "markets": list(MARKETS),
                   "bookmakers": list(BOOKMAKERS), "region": "us", "quota": quota},
        "n_games_in_week": week_games.height,
        "n_games_with_lines": len(lines),
        "games_without_lines": missing,
        "lines": lines,
    }


def write_snapshot(snapshot: dict, path: Path) -> Path:
    if path.exists():
        raise SnapshotExistsError(
            f"{path} already exists; odds snapshots are immutable once written (see CLAUDE.md)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snapshot, indent=2)
    if "apiKey" in text:
        raise OddsFetchError("refusing to write a snapshot that contains request parameters")
    path.write_text(text)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch and freeze this week's lines.")
    ap.add_argument("--pass", dest="pass_name", choices=["early", "late"], required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="write here instead of data/odds/ (for checking the plumbing)")
    args = ap.parse_args(argv)

    games = load_games()
    target = next_week_target(games)
    snap = fetch_snapshot(target, args.pass_name, games)
    path = write_snapshot(snap, args.out or snapshot_path(target, args.pass_name))
    q = snap["source"]["quota"]
    print(f"wrote {path}: {snap['n_games_with_lines']}/{snap['n_games_in_week']} games for "
          f"{target.season} week {target.week} ({args.pass_name}); "
          f"quota remaining {q['requests_remaining']}")
    if snap["games_without_lines"]:
        print(f"  no lines for: {snap['games_without_lines']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
