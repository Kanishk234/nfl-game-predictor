"""Cron-expression math shared by every test that checks *when* something is scheduled to run.

Not a test module itself (no `test_` prefix, so pytest does not collect it) — this exists so
tests/test_workflows.py and tests/test_cloudflare_worker.py can check GitHub's YAML crons and
the Cloudflare Worker's TOML crons against the same kickoff deadlines without two copies of the
arithmetic drifting apart.
"""

from datetime import time, timedelta

#: Kickoffs the unattended season has to beat, as (cron weekday, UTC time). Cron weekdays are
#: 0=Sunday. Thursday night football is a *Friday* kickoff in UTC, which is exactly the kind of
#: conversion this table exists to stop anyone re-deriving by hand.
DEADLINES = {
    "TNF, EDT": (5, time(0, 15)),
    "TNF, EST": (5, time(1, 15)),
    "international Sunday, EDT": (0, time(13, 30)),
    "international Sunday, EST": (0, time(14, 30)),
    "Sunday slate, EDT": (0, time(17, 0)),
    "Sunday slate, EST": (0, time(18, 0)),
    "MNF, EDT": (2, time(0, 15)),
    "MNF, EST": (2, time(1, 15)),
}

SUNDAY_DEADLINES = {k: v for k, v in DEADLINES.items() if "Sunday" in k}
TNF_DEADLINES = {k: v for k, v in DEADLINES.items() if k.startswith("TNF")}


def cron_slots(expr: str) -> list[tuple[int, time]]:
    """(cron weekday, time) pairs one 5-field cron expression fires at.

    Expands `H-H` hour ranges and `a,b,c` day lists, which is as much of the cron grammar as any
    schedule in this repo actually uses.
    """
    minute, hour, _, _, dow = expr.split()
    if "-" in hour:
        lo, hi = (int(x) for x in hour.split("-"))
        hours = range(lo, hi + 1)
    else:
        hours = [int(hour)]
    days = range(7) if dow == "*" else [int(d) for d in dow.split(",")]
    return [(d, time(h, int(minute))) for d in days for h in hours]


def week_minutes(dow: int, t: time) -> int:
    """Minutes since Sunday 00:00, so a Thursday cron and a Friday kickoff are comparable."""
    return dow * 24 * 60 + t.hour * 60 + t.minute


def slots_in_window(exprs: list[str], deadline: tuple[int, time], lead: timedelta) -> list[str]:
    """Cron expressions of `exprs` that fire within `lead` before `deadline`."""
    end = week_minutes(*deadline)
    start = end - lead.total_seconds() / 60
    return [expr for expr in exprs for d, t in cron_slots(expr)
            if start <= week_minutes(d, t) < end]


def cloudflare_dow_to_posix(expr: str) -> str:
    """Translate a Cloudflare Cron Trigger's weekday field to the POSIX convention every other
    cron in this repo uses (GitHub Actions, and `DEADLINES` above).

    Cloudflare numbers the weekday field 1=Sunday..7=Saturday; POSIX/GitHub Actions numbers it
    0=Sunday..6=Saturday. Off by exactly one. `tools/cloudflare-worker/wrangler.toml` learned
    this the hard way — Cloudflare's API rejects `0` outright ("invalid cron string") rather than
    silently misinterpreting it, so the failure is loud at deploy time, but nothing catches the
    same mismatch inside this test suite's own arithmetic without this conversion. Only the
    weekday field changes; minute/hour/day-of-month/month are identical between the two systems.
    """
    minute, hour, dom, month, dow = expr.split()
    if dow != "*":
        dow = ",".join(str(int(v) - 1) for v in dow.split(","))
    return f"{minute} {hour} {dom} {month} {dow}"
