"""The workflow files themselves, checked from the outside.

These jobs run unattended, on Linux, once a week. A mistake here does not show up as a failing
test on a laptop — it shows up as a red run at 5pm on a Thursday, after the prediction has
already been made and with no way to back-date it.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

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
