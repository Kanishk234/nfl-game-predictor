"""The workflow files themselves, checked from the outside.

These jobs run unattended, on Linux, once a week. A mistake here does not show up as a failing
test on a laptop — it shows up as a red run at 5pm on a Thursday, after the prediction has
already been made and with no way to back-date it.
"""

import re
import subprocess
from datetime import timedelta
from pathlib import Path

import pytest
import yaml

from _cron_helpers import (
    DEADLINES,
    SUNDAY_DEADLINES,
    TNF_DEADLINES,
    cron_slots,
    slots_in_window,
    week_minutes,
)
from nfl_predict.health import PREDICTION_DUE_WITHIN
from nfl_predict.predict import LATE_PASS_LEAD_LIMIT

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


def run_steps(path: Path):
    """Every shell command in every job of one workflow."""
    doc = yaml.safe_load(path.read_text())
    for job in doc.get("jobs", {}).values():
        for step in job.get("steps", []) or []:
            if isinstance(step, dict) and "run" in step:
                yield step["run"]


def test_there_are_workflows_to_check():
    assert WORKFLOWS, "no workflow files found; this suite would pass vacuously"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_repo_scripts_are_runnable_on_the_runner(path: Path):
    """A repo script invoked directly must carry the executable bit *in git*.

    Files committed from Windows arrive as mode 100644. `run: tools/publish.sh` then fails on
    ubuntu with exit 126, Permission denied — which is exactly what happened on 2026-09-10.
    Prefixing with `bash` sidesteps the mode entirely and is the form we use; this test accepts
    either, and fails only when a script is invoked in a way the runner cannot execute.
    """
    modes = {}
    for line in subprocess.run(["git", "ls-files", "-s"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.splitlines():
        mode, _, rest = line.partition(" ")
        modes[rest.split("\t", 1)[1]] = mode

    for command in run_steps(path):
        for line in command.splitlines():
            m = re.match(r"\s*(?:\./)?((?:tools|scripts)/[\w./-]+\.sh)\b", line)
            if not m:
                continue
            script = m.group(1)
            assert modes.get(script) == "100755", (
                f"{path.name} runs `{script}` directly but it is committed as "
                f"{modes.get(script)}, not 100755. Either `git update-index --chmod=+x "
                f"{script}` or invoke it as `bash {script}`."
            )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_referenced_script_exists(path: Path):
    for command in run_steps(path):
        for script in re.findall(r"(?:bash |sh |\./)((?:tools|scripts)/[\w./-]+\.sh)", command):
            assert (ROOT / script).exists(), f"{path.name} runs {script}, which does not exist"


def test_the_publisher_is_invoked_the_same_way_everywhere():
    """Three workflows publish. They must not drift apart in how they call the publisher."""
    calls = {p.name: line.strip() for p in WORKFLOWS for command in run_steps(p)
             for line in command.splitlines() if "publish.sh" in line}
    assert len(calls) == 3, f"expected three publishing workflows, found {sorted(calls)}"
    prefixes = {c.split("publish.sh")[0] for c in calls.values()}
    assert len(prefixes) == 1, f"publisher invoked inconsistently: {calls}"


def schedule_crons(path: Path) -> list[str]:
    """The cron expressions of one workflow.

    Note the `doc.get(True)`: YAML 1.1 reads a bare `on:` key as the boolean True, so
    `doc["on"]` is a KeyError on every workflow in this repo.
    """
    doc = yaml.safe_load(path.read_text())
    trigger = doc.get("on", doc.get(True)) or {}
    return [entry["cron"] for entry in (trigger.get("schedule") or [])]


EARLY = ROOT / ".github" / "workflows" / "predict-early.yml"
LATE = ROOT / ".github" / "workflows" / "predict-late.yml"
GRADE = ROOT / ".github" / "workflows" / "grade.yml"
HEALTH = ROOT / ".github" / "workflows" / "health.yml"


def test_no_scheduled_run_is_on_the_hour():
    """:00 is the most contended minute on the shared scheduler.

    Every slot this repo lost was at :00 — Sun 14:00 predict-late and Mon 12:00 grade dropped
    outright, Thu 21:00 and Fri 12:00 delayed by 1h50m and 3h18m.
    """
    offenders = {p.name: [c for c in schedule_crons(p)
                          if any(t.minute == 0 for _, t in cron_slots(c))]
                 for p in WORKFLOWS}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"crons on the hour: {offenders}"


@pytest.mark.parametrize("label", SUNDAY_DEADLINES, ids=list(SUNDAY_DEADLINES))
def test_the_late_pass_has_spare_attempts_in_every_kickoff_window(label: str):
    """Redundancy that survives GitHub dropping runs, for each shape of Sunday.

    A cron only publishes inside its window: earlier than `LATE_PASS_LEAD_LIMIT` before the first
    kickoff it no-ops by design. So the count that matters is not "how many crons" but "how many
    crons land between the window opening and kickoff" — which couples the schedule to the
    constant. Tightening one without widening the other silently thins the redundancy out, and
    that is what this test is here to catch.
    """
    usable = slots_in_window(schedule_crons(LATE), SUNDAY_DEADLINES[label], LATE_PASS_LEAD_LIMIT)
    assert len(usable) >= 3, (
        f"{label}: only {len(usable)} late-pass slot(s) inside the window: {usable}. Observed "
        f"delays on this repo run to 3h18m and runs get dropped entirely."
    )


@pytest.mark.parametrize("label", TNF_DEADLINES, ids=list(TNF_DEADLINES))
def test_the_early_pass_has_spare_attempts_before_thursday_night(label: str):
    """The early pass has no lead-limit gate, so every Thursday slot before kickoff counts.

    This is the highest-stakes schedule in the repo. A dropped late pass costs freshness on games
    that already have an early-pass prediction; a dropped grade is picked up by the next slot.
    A dropped early pass means Thursday's game has no prediction at all, ever.
    """
    usable = slots_in_window(schedule_crons(EARLY), TNF_DEADLINES[label], timedelta(hours=12))
    assert len(usable) >= 3, (
        f"{label}: only {len(usable)} early-pass slot(s) before kickoff: {usable}"
    )


def test_the_early_pass_does_not_publish_absurdly_early():
    """The first Thursday slot is the one that wins, so it sets the baseline's freshness.

    Too late and there is no margin; too early and the Vegas line frozen alongside the prediction
    is a staler comparison. The floor is the point of the stagger; the ceiling is what the
    stagger quietly costs, and it should not drift without someone choosing it.
    """
    thursday = sorted(week_minutes(d, t) for c in schedule_crons(EARLY)
                      for d, t in cron_slots(c) if d == 4)
    margin = timedelta(minutes=week_minutes(*DEADLINES["TNF, EDT"]) - thursday[0])
    assert timedelta(hours=3) <= margin <= timedelta(hours=8), (
        f"first Thursday slot is {margin} before TNF; expected between 3h and 8h"
    )


def test_grading_does_not_hang_on_one_cron_a_day():
    """Grading is idempotent and self-healing, but it is also where a red run comes from."""
    by_day: dict[int, list[str]] = {}
    for c in schedule_crons(GRADE):
        for d, _ in cron_slots(c):
            by_day.setdefault(d, []).append(c)
    assert by_day, "grade has no scheduled runs"
    thin = {d: v for d, v in by_day.items() if len(v) < 2}
    assert not thin, f"grade days with a single attempt: {thin}"


@pytest.mark.parametrize("label", DEADLINES, ids=list(DEADLINES))
def test_health_sweeps_the_hours_before_every_kickoff(label: str):
    """The alarm has to fire while a human can still dispatch the pass by hand.

    `imminent_problems` only reports a missing prediction once the kickoff is inside
    `PREDICTION_DUE_WITHIN`, so a health run outside that window cannot see the problem at all.
    Lengthening the constant without extending these sweeps, or vice versa, leaves the check
    correct and never actually run — the same shared-failure shape as running it only inside the
    grade job. This is the test that keeps the two in step.
    """
    sweeps = slots_in_window(schedule_crons(HEALTH), DEADLINES[label], PREDICTION_DUE_WITHIN)
    assert len(sweeps) >= 2, (
        f"{label}: only {len(sweeps)} health sweep(s) in the {PREDICTION_DUE_WITHIN} before "
        f"kickoff: {sweeps}"
    )


def test_health_runs_outside_the_job_that_writes_the_data():
    """A watchdog that only runs as a step of the grade job shares its failure mode.

    When the 2026-09-14 Mon 12:00 grade cron was dropped, the health check went with it.
    """
    assert schedule_crons(HEALTH), "health has no schedule of its own"
    doc = yaml.safe_load(HEALTH.read_text())
    writes = [line for command in run_steps(HEALTH) for line in command.splitlines()
              if "publish.sh" in line or line.strip().startswith("git ")]
    assert not writes, f"the health job writes to the repo: {writes}"
    assert doc.get("concurrency") is None, (
        "health must not join the data-writes concurrency group; it has to be able to run while "
        "a stuck publish is holding it"
    )
