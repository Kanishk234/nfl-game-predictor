# The weekly flow

What runs, when, and what it writes. All times UTC with the Eastern equivalent; GitHub Actions
cron is always UTC and has no DST awareness, so every schedule is chosen to be safe in both EDT
(UTC−4, Sep–early Nov) and EST (UTC−5, Nov–Jan).

## The week at a glance

| when (UTC) | ET | job | what happens |
|---|---|---|---|
| **Tue 16:00** | 12 PM EDT / 11 AM EST | `predict-early` (safety net) | Publishes **only** if the week opens before Thursday's pass. Normally does nothing. |
| **Thu 21:00** | 5 PM EDT / 4 PM EST | `predict-early` | Retrain → predict every game still ahead → freeze the Vegas lines → publish. |
| **Fri 12:00** | 8 AM EDT / 7 AM EST | `grade` | Scores Thursday night. |
| **Sun 14:00** | 10 AM EDT / 9 AM EST | `predict-late` | Retrain (now with Thursday's result) → re-predict the Sun/Mon games. |
| **Mon 12:00** | 8 AM EDT / 7 AM EST | `grade` | Scores the Sunday slate. |
| **Tue 12:00** | 8 AM EDT / 7 AM EST | `grade` | Scores Monday night. The week is now complete. |

Every job that writes something also rebuilds `site/` and deploys, so the page reflects the new
data within a couple of minutes.

## What each job does

### `predict-early` — Thursday 21:00 UTC

1. Rebuilds the feature frame from nflreadpy and asserts the leakage gate.
2. Refits the model on **every completed game since 2002**. This is the retrain; there is no
   separate training job.
3. Selects the target week's games whose kickoff is still ahead.
4. Fetches the Vegas lines and stamps them with **the same timestamp** as the prediction.
5. Writes `data/predictions/<season>_<week>_early.json` and `data/odds/<season>_<week>_early.json`.
6. Rebuilds the site, commits, pushes, deploys.

This is the frozen official prediction for the week's Thursday–Saturday games, and for the
Sunday/Monday games too until the late pass refreshes them.

### `predict-early` — Tuesday 16:00 UTC, the safety net

Identical, but run with `--only-early-openers`. It publishes **only** when the target week's
first kickoff falls before the next scheduled Thursday pass; otherwise it exits without writing.

It exists because a Thursday-afternoon pass would miss any game kicking off earlier that week,
which happens every season:

- **Thanksgiving** has a 12:30 PM ET game every year since 2002
- **Christmas** sometimes does (2025 week 17)
- **Wednesday openers** (2012, 2026 week 1, 2026 week 12)

Across 2002–2026 that is 31 weeks that would each have lost a game. In 2026 the net fires for
week 1 and week 12 only.

### `predict-late` — Sunday 14:00 UTC

Same steps, but Thursday's result is now in the training data and only the Sunday/Monday games
are still ahead, so only those are re-predicted. Games already played keep their early-pass
prediction, which is immutable.

### `grade` — Friday, Monday, Tuesday 12:00 UTC

1. Pulls final scores from nflreadpy.
2. For each finished game, takes the **latest prediction published before that game's kickoff**
   as official — Thursday's for a Thursday game, Sunday's for a Sunday game — and scores it
   against both the result and the frozen Vegas line.
3. Rewrites `data/results/<season>_<week>.json` and regenerates `data/results/history.json`.
4. Rebuilds the site, commits, pushes, deploys.

Running three times a week is what makes the cards fill in with scores as the week goes rather
than all at once on Tuesday. Grading is idempotent: it reads predictions and rewrites results
from facts, so running it twice changes nothing.

## What is immutable, and what is not

| file | rewritten? |
|---|---|
| `data/predictions/*.json` | **Never.** A corrected prediction is a new file with a new timestamp. |
| `data/odds/*.json` | **Never.** Frozen at the instant of its prediction. |
| `data/results/*.json` | Yes, as more of the week finishes. Results are facts. |
| `data/results/history.json` | Rebuilt from the per-week files on every run. |
| `site/` | Rebuilt on every job that writes data. |

A pass whose file already exists exits green without touching it. Nothing to do is not a failure.

## The gate

A prediction is only valid if it was generated before the game it covers kicked off. This is
enforced per game in three places:

- the pass only selects games whose kickoff is still ahead;
- it refuses to run at all once the week's last kickoff has passed;
- the grader ignores any prediction whose `generated_at_utc` is at or after its game's kickoff.

Every prediction file records the UTC time it ran, how many seconds that was before the first
kickoff it covers, and the exact model and commit that produced it.

## Running a step by hand

```
python -m nfl_predict.predict --pass early   # or --pass late
python -m nfl_predict.grade                  # safe any time; idempotent
python -m nfl_predict.site_build             # rebuild site/ from data/
```

From the Actions tab, `grade` is always safe to dispatch. Dispatching `predict-late` before
Sunday would burn that week's late slot with an earlier prediction, so only do it deliberately.

## When something looks wrong

- **A red `predict` job** usually means the runner was late enough that the whole week had
  already kicked off. That is the gate working; a late prediction is worse than none.
- **A no-op predict job** means that week and pass is already published. Expected, not an error.
- **The site looks stale** — GitHub Pages caches aggressively; hard-refresh (Ctrl+Shift+R)
  before assuming a build problem.
- **Odds missing for a game** shows on the card as "no line yet" rather than being invented.
