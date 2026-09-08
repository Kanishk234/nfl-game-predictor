"""The gate: no feature may carry an as-of time at or after the kickoff it predicts.

There is no feature pipeline yet (Phase 1), so there is nothing to assert against. This file
exists so `pytest -m leakage` collects and exits 0 rather than erroring on an empty selection —
the skip reason is the honest status, not a passing gate. Phase 1 replaces the skip with real
assertions over the processed dataset.
"""

import pytest

pytestmark = pytest.mark.leakage


@pytest.mark.skip(reason="no feature pipeline yet — implemented in Phase 1")
def test_no_feature_is_timestamped_at_or_after_kickoff():
    raise AssertionError("placeholder")
