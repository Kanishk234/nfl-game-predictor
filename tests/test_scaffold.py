"""Phase 0: the package imports and the layout the rest of the plan assumes exists."""

from pathlib import Path

import nfl_predict

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    assert nfl_predict.__version__


def test_data_directories_exist():
    for sub in ("predictions", "odds", "results", "processed"):
        assert (REPO_ROOT / "data" / sub).is_dir(), f"missing data/{sub}"


def test_entrypoint_modules_exist():
    for mod in (
        "data/pipeline.py",
        "model/train.py",
        "predict.py",
        "grade.py",
        "site_build.py",
    ):
        assert (REPO_ROOT / "src" / "nfl_predict" / mod).is_file(), f"missing {mod}"
