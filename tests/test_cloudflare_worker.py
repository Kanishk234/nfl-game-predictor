"""The Cloudflare Worker that backstops GitHub's own scheduler, checked from the outside.

This is Python-only tooling checking a JS/TOML deployable that nothing in this repo's CI ever
runs or deploys (see tools/cloudflare-worker/README.md — it is deployed by hand, on purpose, so
that a broken pipeline here cannot also take out the backstop). That makes it easy for the cron
schedule or the workflow list to drift from what worker.js and wrangler.toml actually say without
anyone noticing until a week silently misses both systems at once. These tests are the seam.
"""

import re
from datetime import time, timedelta
from pathlib import Path

import pytest

from _cron_helpers import (
    DEADLINES,
    SUNDAY_DEADLINES,
    TNF_DEADLINES,
    cron_slots,
    slots_in_window,
    week_minutes,
)
from nfl_predict.predict import LATE_PASS_LEAD_LIMIT

ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = ROOT / "tools" / "cloudflare-worker"
WORKER_JS = WORKER_DIR / "worker.js"
WRANGLER_TOML = WORKER_DIR / "wrangler.toml"

#: Cloudflare's free-plan cap. Not a guess — see tools/cloudflare-worker/README.md and the
#: WebFetch-verified figure recorded in docs/LOG.md: 5 Cron Triggers per *account*, not per
#: Worker, so this repo's Worker cannot use more even if it wanted to.
CLOUDFLARE_FREE_CRON_CAP = 5


def worker_source() -> str:
    return WORKER_JS.read_text(encoding="utf-8")


def wrangler_crons() -> list[str]:
    """The cron expressions inside wrangler.toml's `[triggers]` block.

    Regex rather than a TOML parser: this is a two-file diagnostic, not worth a new dependency
    for. The pattern only has to survive this one file, which is short and reviewed by hand.
    """
    text = WRANGLER_TOML.read_text(encoding="utf-8")
    m = re.search(r"crons\s*=\s*\[(.*?)\]", text, re.DOTALL)
    assert m, "no `crons = [...]` array found in wrangler.toml"
    return re.findall(r'"([^"]+)"', m.group(1))


def worker_workflow_list() -> list[str]:
    m = re.search(r"const WORKFLOWS\s*=\s*\[(.*?)\];", worker_source(), re.DOTALL)
    assert m, "no `const WORKFLOWS = [...]` found in worker.js"
    return re.findall(r'"([^"]+)"', m.group(1))


def test_the_worker_files_exist():
    assert WORKER_JS.exists(), "tools/cloudflare-worker/worker.js is missing"
    assert WRANGLER_TOML.exists(), "tools/cloudflare-worker/wrangler.toml is missing"


def test_the_worker_dispatches_the_real_publishing_workflows():
    """Every name in worker.js's WORKFLOWS list must be a workflow that actually exists.

    A rename in .github/workflows/ silently breaks the Worker's dispatch calls — GitHub's API
    returns 404 for an unknown workflow file, which the Worker only surfaces as a log line nobody
    is watching. This is the check that would have caught it instead.
    """
    listed = worker_workflow_list()
    assert listed, "worker.js lists no workflows to dispatch"
    for name in listed:
        assert (ROOT / ".github" / "workflows" / name).exists(), (
            f"worker.js dispatches {name}, which does not exist in .github/workflows/"
        )


def test_the_worker_covers_every_publishing_workflow():
    """The reverse of the check above: nothing that publishes is missing from the Worker."""
    listed = set(worker_workflow_list())
    assert listed == {"predict-early.yml", "predict-late.yml", "grade.yml"}, (
        f"worker.js dispatches {sorted(listed)}; expected exactly the three publishing workflows"
    )


def test_the_cron_count_is_within_the_cloudflare_free_plan_cap():
    crons = wrangler_crons()
    assert crons, "wrangler.toml declares no cron triggers"
    assert len(crons) <= CLOUDFLARE_FREE_CRON_CAP, (
        f"{len(crons)} cron triggers declared; the free plan allows "
        f"{CLOUDFLARE_FREE_CRON_CAP} per Cloudflare account"
    )


def test_every_cron_expression_is_well_formed():
    for expr in wrangler_crons():
        assert len(expr.split()) == 5, f"not a 5-field cron expression: {expr!r}"
        cron_slots(expr)  # raises if a field cannot be parsed


@pytest.mark.parametrize("label", TNF_DEADLINES, ids=list(TNF_DEADLINES))
def test_at_least_one_worker_tick_precedes_thursday_night(label: str):
    """The Worker exists to backstop the highest-stakes deadline in the repo.

    Reuses the same 12h lookback as the GitHub-side test for the same claim
    (test_workflows.py::test_the_early_pass_has_spare_attempts_before_thursday_night), so
    tightening one without the other is caught the same way.
    """
    usable = slots_in_window(wrangler_crons(), TNF_DEADLINES[label], timedelta(hours=12))
    assert usable, f"{label}: no Worker cron tick falls in the 12h before TNF: {wrangler_crons()}"


@pytest.mark.parametrize("label", SUNDAY_DEADLINES, ids=list(SUNDAY_DEADLINES))
def test_at_least_one_worker_tick_precedes_every_shape_of_sunday(label: str):
    """Both the international and the normal Sunday case need their own covering tick — a
    Worker schedule that only ever fires after 13:30 UTC would silently stop backstopping the
    six-to-eleven international games a season that predict-late.yml's own comments describe."""
    usable = slots_in_window(wrangler_crons(), SUNDAY_DEADLINES[label], LATE_PASS_LEAD_LIMIT)
    assert usable, (
        f"{label}: no Worker cron tick falls within LATE_PASS_LEAD_LIMIT "
        f"({LATE_PASS_LEAD_LIMIT}) of kickoff: {wrangler_crons()}"
    )


#: Cron expressions in wrangler.toml that exist to precede a kickoff. The grade tick is
#: deliberately not one of these — grading has no kickoff deadline, only "after MNF ends" — so
#: it is checked on its own terms below rather than forced through the kickoff-margin test.
KICKOFF_TICKS = ["11 20 * * 4", "37 22 * * 4", "35 9 * * 0", "51 14 * * 0"]


def test_every_declared_tick_is_accounted_for():
    """KICKOFF_TICKS plus the grade tick must be the whole schedule, or a newly added tick
    would silently skip every check below it instead of failing one."""
    assert set(wrangler_crons()) - set(KICKOFF_TICKS) == {"13 16 * * 2"}, (
        f"wrangler.toml has a tick this test suite does not know how to classify: "
        f"{wrangler_crons()}. Add it to KICKOFF_TICKS or extend this test."
    )


def test_no_kickoff_tick_is_absurdly_early():
    """A tick more than a day ahead of every deadline in the table is not backstopping anything
    kickoff-shaped — it is either a mistake or dead weight against the 5-trigger cap."""
    for expr in KICKOFF_TICKS:
        d, t = cron_slots(expr)[0]
        wm = week_minutes(d, t)
        margins = [week_minutes(*dl) - wm for dl in DEADLINES.values()]
        soonest = min(((m if m > 0 else m + 7 * 24 * 60) for m in margins), default=None)
        assert soonest is not None and soonest <= 24 * 60, (
            f"{expr}: no deadline in DEADLINES falls within 24h after this tick"
        )


def test_the_grade_tick_lands_after_the_github_grade_slots():
    """The grade tick's whole point is to close the gap *after* GitHub's own nine grade slots —
    a copy of one of those slots would be redundant in time, not extra coverage."""
    grade_tick = next(c for c in wrangler_crons() if c not in KICKOFF_TICKS)
    d, t = cron_slots(grade_tick)[0]
    assert d == 2, f"expected the grade tick on Tuesday (2), got weekday {d}"
    # The last GitHub grade slot this repo schedules, Tuesday 15:07 UTC (see grade.yml).
    last_github_slot = week_minutes(2, time(15, 7))
    assert week_minutes(d, t) > last_github_slot, (
        f"{grade_tick} does not land after grade.yml's last Tuesday slot (15:07 UTC)"
    )


def test_the_token_is_a_secret_not_a_committed_value():
    """GITHUB_TOKEN must come from `wrangler secret put`, never from a file in this repo.

    Same rule as ODDS_API_KEY (see the project's secret-hygiene practice): a token committed to
    wrangler.toml or worker.js is a token leaked to every clone and every fork, permanently, the
    moment it is pushed — `git log` does not forget.
    """
    toml_text = WRANGLER_TOML.read_text(encoding="utf-8")
    # A real TOML section header, not the string appearing inside a `#` comment explaining why
    # there is no such section (which this file deliberately has, and which would otherwise trip
    # this check on itself).
    has_vars_section = re.search(r"^\s*\[vars\]\s*$", toml_text, re.MULTILINE)
    assert not has_vars_section, (
        "wrangler.toml has a [vars] block — GITHUB_TOKEN must be a `wrangler secret put` "
        "secret, never a plaintext var"
    )
    token_shapes = re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")
    for path in (WORKER_JS, WRANGLER_TOML):
        text = path.read_text(encoding="utf-8")
        assert not token_shapes.search(text), f"a real-looking GitHub token is committed in {path}"


def test_the_worker_reads_the_token_from_env_not_a_literal():
    assert "env.GITHUB_TOKEN" in worker_source(), (
        "worker.js should read the token from env.GITHUB_TOKEN (a Wrangler secret binding)"
    )


def test_the_worker_has_a_season_guard():
    """Without this, every tick outside Sep-Feb hits predict.py's `next_week_target`, which
    raises when there is no game left in the loaded schedule — a red run, every tick, for seven
    months a year. See worker.js's own comment for why this lives here and not in Python."""
    assert "inSeason" in worker_source(), "worker.js has lost its off-season guard"


def test_the_worker_targets_this_repo():
    src = worker_source()
    assert 'OWNER = "Kanishk234"' in src
    assert 'REPO = "nfl-game-predictor"' in src
