"""The workflow files themselves, checked from the outside.

These jobs run unattended, on Linux, once a week. A mistake here does not show up as a failing
test on a laptop — it shows up as a red run at 5pm on a Thursday, after the prediction has
already been made and with no way to back-date it.
"""

import re
import subprocess
from datetime import UTC, datetime, time
from pathlib import Path

import pytest
import yaml

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


def cron_time(expr: str) -> time:
    minute, hour = expr.split()[:2]
    return time(int(hour), int(minute))


LATE = ROOT / ".github" / "workflows" / "predict-late.yml"

#: The first Sunday kickoff the late pass has to beat, in UTC. International games are the tight
#: case and the reason the single 14:00 UTC cron was wrong: 9:30 AM ET is 13:30 UTC in EDT.
SUNDAY_FIRST_KICKOFFS = {
    "international, EDT": time(13, 30),
    "international, EST": time(14, 30),
    "normal slate, EDT": time(17, 0),
    "normal slate, EST": time(18, 0),
}


@pytest.mark.parametrize("label,kickoff", SUNDAY_FIRST_KICKOFFS.items(), ids=list(SUNDAY_FIRST_KICKOFFS))
def test_the_late_pass_has_spare_attempts_in_every_kickoff_window(label: str, kickoff: time):
    """Redundancy that survives GitHub dropping runs, for each shape of Sunday.

    A cron only publishes inside its window: earlier than `LATE_PASS_LEAD_LIMIT` before the first
    kickoff it no-ops by design. So the count that matters is not "how many crons" but "how many
    crons land between the window opening and kickoff" — which couples the schedule to the
    constant. Tightening one without widening the other silently thins the redundancy out, and
    that is what this test is here to catch.
    """
    day = datetime(2026, 9, 13, tzinfo=UTC)
    deadline = datetime.combine(day, kickoff, tzinfo=UTC)
    opens = deadline - LATE_PASS_LEAD_LIMIT
    usable = [c for c in schedule_crons(LATE)
              if opens <= datetime.combine(day, cron_time(c), tzinfo=UTC) < deadline]
    assert len(usable) >= 3, (
        f"{label}: first kickoff {kickoff}, window opens {opens.time()}, but only "
        f"{len(usable)} cron slot(s) fall inside it: {usable}. Observed delays on this repo run "
        f"to 3h18m and runs get dropped entirely, so one or two slots is not a schedule."
    )


def test_the_late_pass_avoids_the_top_of_the_hour():
    """:00 is the most contended slot on the shared scheduler, and the one that got dropped."""
    on_the_hour = [c for c in schedule_crons(LATE) if cron_time(c).minute == 0]
    assert not on_the_hour, f"predict-late crons on the hour: {on_the_hour}"


def test_the_late_pass_only_runs_on_sundays():
    days = {c.split()[4] for c in schedule_crons(LATE)}
    assert days == {"0"}, f"predict-late is scheduled off-Sunday: {days}"
