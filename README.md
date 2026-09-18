# nfl-predict

A model picks every NFL game **before kickoff**, writes the pick where it cannot be changed, and
gets graded afterwards against the Vegas line.

**→ [kanishk234.github.io/nfl-game-predictor](https://kanishk234.github.io/nfl-game-predictor/)**

Most prediction sites show you what they think. This one shows you what they thought *last week*,
and whether it was right.

## When the site updates

GitHub Actions' own `schedule` trigger drops runs under load — measured on this repo at a
2-in-6 fire rate on 2026-09-14/15, and again on 2026-09-18 when all three Friday grading attempts
missed outright. Each publishing job still has several staggered GitHub attempts per day (see
[`docs/WEEKLY_FLOW.md`](docs/WEEKLY_FLOW.md) for the full list), so the site often updates earlier
than the times below. But the only *guaranteed* floor is a small [Cloudflare
Worker](tools/cloudflare-worker) that calls the same jobs on a schedule GitHub does not control.
These are its times — the latest the site should ever update by, not the typical case:

| day | time (CT) | what changes |
|---|---|---|
| **Thursday** | 3:11 PM | New picks for the whole week appear, before that night's game |
| **Thursday** | 5:37 PM | Last-resort retry of the above, still ahead of a 7:15 PM CT kickoff |
| **Sunday** | 4:35 AM | Fresher picks for the Sunday and Monday games (covers early international kickoffs) |
| **Sunday** | 9:51 AM | Same, timed for a normal noon CT slate |
| **Tuesday** | 11:13 AM | Every result for the week, including Monday night — the week is complete |

Times above are Central Daylight Time; they run an hour earlier (Central Standard Time) once
clocks fall back in November. Grading (Friday/Monday's results) has no dedicated Cloudflare
backstop yet — it relies on GitHub's own nine staggered attempts across those two days, with the
Tuesday slot above as the final catch-all if every one of those misses. A few weeks open earlier
than Thursday — Thanksgiving, Christmas, the odd Wednesday opener — and those get their picks on
the preceding Tuesday instead.

Games already played keep the pick they were given. The Sunday pass only re-predicts games that
have not started.

## What you see per game

- **Our pick** and how confident the model is
- **The Vegas favorite** beside it, and a note when the two disagree
- **The point spread**, ours and the market's, on the same scale
- Once played: the score, who won, whether we were right, and whether our side covered

## How good is it?

Honestly: **not as good as Vegas.** Replayed over the five seasons it was never tuned on
(2021–2025, 1,424 games, retrained every week):

| | picks right | Brier score | spread error |
|---|---|---|---|
| This model | 65.0% | 0.221 | 10.07 pts |
| Vegas closing line | 66.5% | 0.211 | 9.76 pts |

Beating the closing line with public data is a high bar and we do not clear it. The site says so
on its own [track record page](https://kanishk234.github.io/nfl-game-predictor/season.html). The
point is the record, not the bragging.

## Why the record is trustworthy

Every pick is a JSON file committed to this repository **before** the games it covers, stamped
with the time it was generated and the exact model and commit that produced it. The Vegas line is
fetched and frozen in the same instant, into a file beside it. After the games, a separate step
reads those files and writes the results — it cannot change a prediction.

A prediction is never edited. If one is ever wrong or late, the fix is a new file with a new
timestamp; the old one stays.

- [`data/predictions/`](data/predictions) — what we said, and when
- [`data/odds/`](data/odds) — the Vegas line at that same moment
- [`data/results/`](data/results) — what actually happened
- [Actions](../../actions) — the jobs that wrote them

A test suite asserts on every commit that no feature uses information from after a game's
kickoff. That check runs on the whole dataset back to 2002.

## The model

Logistic regression for win probability, ridge regression for the spread, on 30 features: Elo
ratings, rolling EPA form from play-by-play, quarterback ratings and draft position, rest days,
and schedule facts. It retrains before every set of picks on all 6,499 completed games since
2002.

A gradient-boosted model was tried and lost — the [full report](docs/reports/phase2_model_backtest.md)
records everything tried and rejected, including the ideas that did not work.

## Running it yourself

```bash
pip install -e ".[dev]"

python -m nfl_predict.data.pipeline      # refresh the feature data
python -m nfl_predict.model.train        # retrain and re-run the backtest
python -m nfl_predict.predict --pass early   # this week's picks (needs an odds API key)
python -m nfl_predict.grade              # score finished games
python -m nfl_predict.site_build         # rebuild site/ from data/

pytest                  # the full suite
pytest -m leakage       # just the gate
```

Live odds need a free [The Odds API](https://the-odds-api.com) key in `ODDS_API_KEY`
(a gitignored `.env` locally, an Actions secret in CI). Everything else runs from public
[nflverse](https://github.com/nflverse) data.

Runs entirely on free tiers: GitHub Actions, GitHub Pages, and a free odds API.

## More

- [Weekly flow](docs/WEEKLY_FLOW.md) — the full schedule and what each job writes
- [Plan](docs/PLAN.md) and [log](docs/LOG.md) — how it was built, and what broke on the way
- [Reports](docs/reports) — one per phase, with real numbers
