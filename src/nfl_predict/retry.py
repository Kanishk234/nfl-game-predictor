"""Retry transient failures.

The scheduled jobs get one shot a week. nflverse serves its data from GitHub releases and the
odds come from a live HTTP API; either can return a 502 for a few seconds. Without a retry, a
blip that lasts less than a minute costs a whole week's prediction, and the record has a hole in
it that can never be filled — a prediction cannot be back-dated.

Retries are for *transient* failures only. A missing season file, a refused season, a bad API
key or an exhausted quota are permanent for this run and must fail fast rather than burn four
minutes proving it.
"""

from __future__ import annotations

import time
from collections.abc import Callable

#: Roughly 45 seconds of trying in total, which covers a short outage without delaying a job
#: that is genuinely broken.
DEFAULT_ATTEMPTS = 4
DEFAULT_BASE_DELAY = 3.0


def with_retries[T](
    fn: Callable[[], T],
    *,
    what: str,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    retry_on: tuple[type[BaseException], ...] = (OSError, ConnectionError, TimeoutError),
    give_up_on: tuple[type[BaseException], ...] = (),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn`, retrying transient failures with exponential backoff.

    `give_up_on` wins over `retry_on`, so a caller can mark a subclass as permanent even when
    its base class is retryable.
    """
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except give_up_on:
            raise
        except retry_on as exc:
            last = exc
            if attempt == attempts:
                break
            delay = base_delay * 2 ** (attempt - 1)
            print(f"{what} failed ({type(exc).__name__}); retrying in {delay:.0f}s "
                  f"[attempt {attempt} of {attempts}]")
            sleep(delay)
    raise RuntimeError(f"{what} failed after {attempts} attempts: {last}") from last
