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
    cloudflare_dow_to_posix,
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
    """The cron expressions inside wrangler.toml's `[triggers]` block, exactly as declared.

    In Cloudflare's own weekday convention (1=Sunday..7=Saturday) — Cloudflare's API rejects
    anything else at deploy time (see wrangler.toml's own comment; this repo found out the hard
    way, deploying `* * 0` for Sunday: "invalid cron string"). Regex rather than a TOML parser:
    this is a two-file diagnostic, not worth a new dependency for. The pattern only has to
    survive this one file, which is short and reviewed by hand.
    """
    text = WRANGLER_TOML.read_text(encoding="utf-8")
    m = re.search(r"crons\s*=\s*\[(.*?)\]", text, re.DOTALL)
    assert m, "no `crons = [...]` array found in wrangler.toml"
    return re.findall(r'"([^"]+)"', m.group(1))


def wrangler_crons_posix() -> list[str]:
    """The same expressions, translated to the POSIX weekday convention `DEADLINES` and every
    other cron helper in this repo uses. Use this, not `wrangler_crons()`, for anything that
    compares a tick's weekday against a real calendar day."""
    return [cloudflare_dow_to_posix(e) for e in wrangler_crons()]


def cron_targets() -> dict[str, str]:
    """worker.js's `CRON_TARGETS` map: cron expression -> the one workflow it dispatches."""
    m = re.search(r"const CRON_TARGETS\s*=\s*\{(.*?)\};", worker_source(), re.DOTALL)
    assert m, "no `const CRON_TARGETS = {...}` found in worker.js"
    return dict(re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', m.group(1)))


def test_the_worker_files_exist():
    assert WORKER_JS.exists(), "tools/cloudflare-worker/worker.js is missing"
    assert WRANGLER_TOML.exists(), "tools/cloudflare-worker/wrangler.toml is missing"


def test_the_worker_dispatches_the_real_publishing_workflows():
    """Every value in worker.js's CRON_TARGETS map must be a workflow that actually exists.

    A rename in .github/workflows/ silently breaks the Worker's dispatch calls — GitHub's API
    returns 404 for an unknown workflow file, which the Worker only surfaces as a log line nobody
    is watching. This is the check that would have caught it instead.
    """
    listed = set(cron_targets().values())
    assert listed, "worker.js's CRON_TARGETS maps no cron to any workflow"
    for name in listed:
        assert (ROOT / ".github" / "workflows" / name).exists(), (
            f"worker.js dispatches {name}, which does not exist in .github/workflows/"
        )


def test_the_worker_covers_every_publishing_workflow():
    """The reverse of the check above: nothing that publishes is missing from the Worker."""
    listed = set(cron_targets().values())
    assert listed == {"predict-early.yml", "predict-late.yml", "grade.yml"}, (
        f"worker.js dispatches {sorted(listed)}; expected exactly the three publishing workflows"
    )


def test_every_wrangler_cron_has_exactly_one_dispatch_target():
    """wrangler.toml's declared crons and worker.js's CRON_TARGETS keys must be the same set.

    A cron declared in wrangler.toml with no entry in CRON_TARGETS dispatches nothing on that
    tick — the Worker logs an error nobody is watching and silently does not back up whatever
    that tick existed to cover. An entry in CRON_TARGETS with no matching cron in wrangler.toml
    is dead code that nothing ever triggers. Both are silent by construction; this is the test
    that isn't.
    """
    assert set(cron_targets()) == set(wrangler_crons()), (
        f"CRON_TARGETS keys {sorted(cron_targets())} != wrangler.toml crons "
        f"{sorted(wrangler_crons())}"
    )


def test_each_real_tick_dispatches_exactly_one_workflow():
    """A cron mapping to one workflow (CRON_TARGETS) is one part of this; the other is that
    `scheduled()` actually looks up only its own tick's entry rather than iterating every target.

    This is the whole point of CRON_TARGETS over the earlier "dispatch all three every tick"
    design: all three publishing workflows share the `data-writes` concurrency group (so a
    predict pass and a grade run never race on `git push`), and GitHub Actions concurrency groups
    hold at most one running run plus one queued run — a third simultaneous dispatch in the same
    group is cancelled outright. Verified live on this repo's own account: the first
    `/dispatch-now` call after deploying fired all three at once, and `grade` came back
    `"conclusion": "cancelled"` with zero jobs ever created. `test_every_wrangler_cron_has_
    exactly_one_dispatch_target` covers the data side (CRON_TARGETS' shape); this covers the
    behavioral side — a future edit could restore CRON_TARGETS' one-key-one-value shape and still
    reintroduce the incident by having `scheduled()` iterate every target regardless of
    `event.cron`, which no purely data-shaped check would catch.
    """
    src = worker_source()
    assert "dispatchMany(env, ALL_WORKFLOWS)" in src, (
        "expected dispatchMany(env, ALL_WORKFLOWS) inside fetch() (the manual /dispatch-now "
        "check) and nowhere else"
    )
    assert re.search(r"scheduled\(event, env, ctx\)\s*\{.*?CRON_TARGETS\[event\.cron\]", src, re.DOTALL), (
        "scheduled() should look up CRON_TARGETS[event.cron] and dispatch only that one "
        "workflow — not iterate every target on every tick"
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


def test_every_weekday_field_is_in_cloudflares_range():
    """Cheap, direct, and deliberately independent of `cloudflare_dow_to_posix`.

    The first deploy of this file used POSIX weekday numbers (0=Sunday) instead of Cloudflare's
    (1=Sunday). Only the Sunday entries (`0`) got caught — Cloudflare's API rejects an
    out-of-range value outright ("invalid cron string"). The Thursday and Tuesday entries (`4`
    and `2`) were silently *accepted*, because both are valid values in Cloudflare's own 1-7
    range — they just meant Wednesday and Monday instead. Three of five ticks would have fired
    on the wrong day with no error at all; only bad luck on the other two surfaced anything. This
    test would have caught all five before the first `wrangler deploy`, without needing a real
    deploy to find out: any weekday field outside 1-7 fails loudly, right here.
    """
    for expr in wrangler_crons():
        dow = expr.split()[-1]
        if dow == "*":
            continue
        for value in dow.split(","):
            assert 1 <= int(value) <= 7, (
                f"{expr!r}: weekday {value} is outside Cloudflare's 1=Sunday..7=Saturday range "
                f"— did this get written in the POSIX convention (0=Sunday) by mistake?"
            )


@pytest.mark.parametrize("label", TNF_DEADLINES, ids=list(TNF_DEADLINES))
def test_at_least_one_worker_tick_precedes_thursday_night(label: str):
    """The Worker exists to backstop the highest-stakes deadline in the repo.

    Reuses the same 12h lookback as the GitHub-side test for the same claim
    (test_workflows.py::test_the_early_pass_has_spare_attempts_before_thursday_night), so
    tightening one without the other is caught the same way.
    """
    usable = slots_in_window(wrangler_crons_posix(), TNF_DEADLINES[label], timedelta(hours=12))
    assert usable, (
        f"{label}: no Worker cron tick falls in the 12h before TNF: {wrangler_crons_posix()}"
    )


@pytest.mark.parametrize("label", SUNDAY_DEADLINES, ids=list(SUNDAY_DEADLINES))
def test_at_least_one_worker_tick_precedes_every_shape_of_sunday(label: str):
    """Both the international and the normal Sunday case need their own covering tick — a
    Worker schedule that only ever fires after 13:30 UTC would silently stop backstopping the
    six-to-eleven international games a season that predict-late.yml's own comments describe."""
    usable = slots_in_window(wrangler_crons_posix(), SUNDAY_DEADLINES[label], LATE_PASS_LEAD_LIMIT)
    assert usable, (
        f"{label}: no Worker cron tick falls within LATE_PASS_LEAD_LIMIT "
        f"({LATE_PASS_LEAD_LIMIT}) of kickoff: {wrangler_crons_posix()}"
    )


#: Cron expressions in wrangler.toml (Cloudflare's own weekday convention, 1=Sunday) that exist
#: to precede a *fixed weekly* kickoff (TNF, Sunday, MNF) and so are checked against DEADLINES
#: below. Two ticks are deliberately not in this list, each checked on its own terms instead:
#: the grade tick (no kickoff deadline, only "after MNF ends") and the Tuesday early-opener tick
#: (its deadline is an irregular, occasional kickoff — Thanksgiving/Christmas/a Wednesday
#: opener — not a fixed weekly one DEADLINES can model).
KICKOFF_TICKS = ["11 20 * * 5", "35 9 * * 1", "51 14 * * 1"]

#: The two Tuesday ticks, by what they dispatch rather than by position — see
#: test_the_grade_tick_lands_after_the_github_grade_slots for why position isn't safe to assume.
GRADE_TICK = "13 16 * * 3"
EARLY_OPENER_TICK = "35 18 * * 3"


def test_every_declared_tick_is_accounted_for():
    """KICKOFF_TICKS plus the grade and early-opener ticks must be the whole schedule, or a
    newly added tick would silently skip every check below it instead of failing one."""
    assert set(wrangler_crons()) - set(KICKOFF_TICKS) == {GRADE_TICK, EARLY_OPENER_TICK}, (
        f"wrangler.toml has a tick this test suite does not know how to classify: "
        f"{wrangler_crons()}. Add it to KICKOFF_TICKS or extend this test."
    )


def test_no_kickoff_tick_is_absurdly_early():
    """A tick more than a day ahead of every deadline in the table is not backstopping anything
    kickoff-shaped — it is either a mistake or dead weight against the 5-trigger cap."""
    for expr in KICKOFF_TICKS:
        d, t = cron_slots(cloudflare_dow_to_posix(expr))[0]
        wm = week_minutes(d, t)
        margins = [week_minutes(*dl) - wm for dl in DEADLINES.values()]
        soonest = min(((m if m > 0 else m + 7 * 24 * 60) for m in margins), default=None)
        assert soonest is not None and soonest <= 24 * 60, (
            f"{expr}: no deadline in DEADLINES falls within 24h after this tick"
        )


def test_the_grade_tick_lands_after_the_github_grade_slots():
    """The grade tick's whole point is to close the gap *after* GitHub's own nine grade slots —
    a copy of one of those slots would be redundant in time, not extra coverage."""
    assert GRADE_TICK in wrangler_crons()
    d, t = cron_slots(cloudflare_dow_to_posix(GRADE_TICK))[0]
    assert d == 2, f"expected the grade tick on Tuesday (POSIX weekday 2), got {d}"
    # The last GitHub grade slot this repo schedules, Tuesday 15:07 UTC (see grade.yml).
    last_github_slot = week_minutes(2, time(15, 7))
    assert week_minutes(d, t) > last_github_slot, (
        f"{GRADE_TICK} does not land after grade.yml's last Tuesday slot (15:07 UTC)"
    )


def test_the_early_opener_tick_lands_after_githubs_own_tuesday_attempts():
    """Same reasoning as the grade tick: this exists to catch the case where all four of
    GitHub's native Tuesday early-opener attempts (predict-early.yml) miss, so it should be
    positioned after the last of them, not duplicate one in time."""
    assert EARLY_OPENER_TICK in wrangler_crons()
    d, t = cron_slots(cloudflare_dow_to_posix(EARLY_OPENER_TICK))[0]
    assert d == 2, f"expected the early-opener tick on Tuesday (POSIX weekday 2), got {d}"
    # The last GitHub early-opener slot this repo schedules, Tuesday 18:07 UTC (predict-early.yml).
    last_github_slot = week_minutes(2, time(18, 7))
    assert week_minutes(d, t) > last_github_slot, (
        f"{EARLY_OPENER_TICK} does not land after predict-early.yml's last Tuesday slot (18:07 UTC)"
    )


def test_the_early_opener_tick_dispatches_predict_early():
    assert cron_targets()[EARLY_OPENER_TICK] == "predict-early.yml"


def test_the_early_opener_tick_passes_the_only_early_openers_input():
    """Without this input, a plain workflow_dispatch call has no `github.event.schedule` to
    match predict-early.yml's Tuesday-detection logic against, so it would fall through to the
    default branch and publish the full week unconditionally — every Tuesday, not just
    early-opener ones. See worker.js's CRON_INPUTS comment."""
    m = re.search(r"const CRON_INPUTS\s*=\s*\{(.*?)\};", worker_source(), re.DOTALL)
    assert m, "no `const CRON_INPUTS = {...}` found in worker.js"
    assert EARLY_OPENER_TICK in m.group(1) and "only_early_openers" in m.group(1) and '"true"' in m.group(1), (
        f"CRON_INPUTS does not appear to map {EARLY_OPENER_TICK!r} to only_early_openers: true"
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
