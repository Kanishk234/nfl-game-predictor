"""Did the scheduled runs actually fire?

    python tools/schedule_report.py
    python tools/schedule_report.py --since 2026-09-01 --workflow predict-early

GitHub delays scheduled runs under load and drops them outright rather than queueing them. That
is not visible in the Actions tab, which only shows what *did* run — a dropped slot leaves no
trace at all, which is exactly how the 2026-09-13 late pass and the 2026-09-14 grade job went
missing without anything going red.

This reconstructs the slots from the cron expressions in `.github/workflows/`, matches them
against the runs the public API reports, and names what is missing. Read-only, no auth, no
dependencies beyond the standard library: it only ever GETs the public runs endpoint.

The window defaults to the last commit that touched `.github/workflows/`, because comparing
today's cron expressions against runs from before they existed reports the whole history as
dropped. Pass --since to override.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

#: How long after its slot a run may start and still count as that slot rather than a new one.
#: Observed delays on this repo reach 3h18m; beyond six hours "late" stops being a useful word.
MAX_DELAY = timedelta(hours=6)


def started_at(run: dict) -> datetime:
    """When the run was created. `fromisoformat` handles the API's trailing Z directly."""
    return datetime.fromisoformat(run["created_at"])


def repo_slug() -> str:
    url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()
    m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    if not m:
        raise SystemExit(f"cannot read a github repo out of {url!r}")
    return m.group(1)


def field(spec: str, lo: int, hi: int) -> list[int]:
    """One cron field -> the values it matches. Handles `*`, `a,b`, `a-b` and `*/n`."""
    out: set[int] = set()
    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, raw = part.split("/")
            step = int(raw)
        if part == "*":
            lo_, hi_ = lo, hi
        elif "-" in part:
            lo_, hi_ = (int(x) for x in part.split("-"))
        else:
            lo_ = hi_ = int(part)
        out.update(range(lo_, hi_ + 1, step))
    return sorted(out)


def crons(path: Path) -> list[str]:
    """Cron expressions of one workflow, read without a YAML parser.

    Deliberately regex rather than pyyaml: this is a diagnostic you want to be able to run in a
    bare checkout when something is already wrong, without installing the package first.
    """
    text = path.read_text(encoding="utf-8")
    return re.findall(r'^\s*-\s*cron:\s*["\']([^"\']+)["\']', text, re.MULTILINE)


def slots(expr: str, start: datetime, end: datetime) -> list[datetime]:
    """Every firing time of one cron expression in [start, end)."""
    minute, hour, dom, _month, dow = expr.split()
    minutes, hours = field(minute, 0, 59), field(hour, 0, 23)
    days = {d % 7 for d in field(dow, 0, 7)} if dow != "*" else set(range(7))
    any_day = dom == "*" and dow == "*"
    out = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < end:
        if any_day or (day.weekday() + 1) % 7 in days:
            for h in hours:
                for m in minutes:
                    t = day.replace(hour=h, minute=m)
                    if start <= t < end:
                        out.append(t)
        day += timedelta(days=1)
    return sorted(out)


def fetch_runs(slug: str, since: datetime) -> list[dict]:
    runs, page = [], 1
    while page <= 10:
        url = (f"https://api.github.com/repos/{slug}/actions/runs"
               f"?per_page=100&page={page}&event=schedule")
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "nfl-predict-schedule-report"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                batch = json.load(resp).get("workflow_runs", [])
        except urllib.error.HTTPError as exc:
            raise SystemExit(f"GitHub API returned {exc.code}: {exc.reason}") from exc
        if not batch:
            break
        runs += batch
        if started_at(batch[-1]) < since:
            break
        page += 1
    return [r for r in runs
            if started_at(r) >= since - MAX_DELAY]


def workflows_last_changed() -> datetime | None:
    out = subprocess.run(["git", "log", "-1", "--format=%cI", "--", ".github/workflows"],
                         cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    return datetime.fromisoformat(out).astimezone(UTC) if out else None


def match(expected: list[datetime], runs: list[dict]) -> tuple[dict, list[datetime]]:
    """Pair each run with the latest unclaimed slot it could have come from."""
    claimed: dict[datetime, dict] = {}
    for run in sorted(runs, key=lambda r: r["created_at"]):
        started = started_at(run)
        candidates = [s for s in expected
                      if s not in claimed and s <= started <= s + MAX_DELAY]
        if candidates:
            claimed[max(candidates)] = run
    return claimed, [s for s in expected if s not in claimed]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", help="ISO date; defaults to the last workflow change")
    ap.add_argument("--workflow", help="only this workflow (file stem, e.g. predict-early)")
    args = ap.parse_args(argv)

    now = datetime.now(UTC)
    if args.since:
        since = datetime.fromisoformat(args.since).replace(tzinfo=UTC)
        why = "as given"
    else:
        changed = workflows_last_changed()
        since = changed or now - timedelta(days=14)
        why = "last workflow change" if changed else "no workflow history; last 14 days"

    slug = repo_slug()
    print(f"{slug}   since {since:%Y-%m-%d %H:%M} UTC ({why})   now {now:%Y-%m-%d %H:%M} UTC\n")

    runs = fetch_runs(slug, since)
    by_workflow: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_workflow[Path(r["path"]).stem].append(r)

    paths = sorted(WORKFLOWS.glob("*.yml"))
    if args.workflow:
        paths = [p for p in paths if p.stem == args.workflow]
    if not paths:
        raise SystemExit("no matching workflow files")

    delays: list[timedelta] = []
    settled: list[tuple[bool, bool]] = []   # (slot was at :00, slot fired) for decided slots
    total_fired = total_dropped = total_pending = 0

    for path in paths:
        expr_list = crons(path)
        if not expr_list:
            continue
        expected = sorted(s for e in expr_list for s in slots(e, since, now))
        top_of_hour = {s for e in expr_list if e.split()[0] == "0" for s in slots(e, since, now)}
        claimed, missing = match(expected, by_workflow.get(path.stem, []))
        # a slot inside MAX_DELAY of now may still be coming, not dropped
        pending = [s for s in missing if now - s < MAX_DELAY]
        dropped = [s for s in missing if now - s >= MAX_DELAY]
        total_fired += len(claimed)
        total_dropped += len(dropped)
        total_pending += len(pending)

        print(f"{path.stem}  ({len(expected)} slot(s) expected)")
        for slot in expected:
            if now - slot >= MAX_DELAY:
                settled.append((slot in top_of_hour, slot in claimed))
            if slot in claimed:
                run = claimed[slot]
                started = started_at(run)
                late = started - slot
                delays.append(late)
                mins = int(late.total_seconds() // 60)
                flag = "" if run["conclusion"] == "success" else f"  [{run['conclusion']}]"
                print(f"  {slot:%a %m-%d %H:%M}  fired +{mins // 60}h{mins % 60:02d}m{flag}")
            elif slot in pending:
                print(f"  {slot:%a %m-%d %H:%M}  pending (still inside the {MAX_DELAY} window)")
            else:
                print(f"  {slot:%a %m-%d %H:%M}  DROPPED")
        print()

    seen = total_fired + total_dropped
    print("=" * 62)
    if not seen:
        print("no slots have come due yet in this window — nothing to conclude")
        return 0
    print(f"fired {total_fired}/{seen} ({100 * total_fired / seen:.0f}%), "
          f"dropped {total_dropped}, pending {total_pending}")
    if delays:
        worst = max(delays)
        median = sorted(delays)[len(delays) // 2]
        print(f"delay: median {int(median.total_seconds() // 60)}m, "
              f"worst {int(worst.total_seconds() // 60)}m")

    # The premise the whole stagger rests on: is :00 actually worse than the rest of the hour?
    # If both lines end up healthy, the drops were never about the minute and the next thing to
    # try is an external trigger, not more slots.
    for is_top in (True, False):
        oks = [ok for top, ok in settled if top is is_top]
        if oks:
            label = "on the hour (:00)" if is_top else "off the hour"
            print(f"{label:<18} fired {sum(oks)}/{len(oks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
