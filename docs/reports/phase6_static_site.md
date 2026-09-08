# Phase 6 — Static site

**Status:** built, reviewed by two people, redesigned on their feedback, live. **Date:** 2026-09-08.

## What was built

**`site_build.py`** — `python -m nfl_predict.site_build` renders three kinds of page from
`data/predictions/`, `data/results/` and `data/backtest.json`:

- `index.html` — the current week (the latest week with a prediction file)
- `weeks/<season>_<ww>.html` — every week, current one included
- `season.html` — season-to-date record and charts, the pre-live backtest, how to verify

Nothing else: no JavaScript, no fetches, no framework. Charts are inline SVG drawn in Python.
Every page is regenerated on every run, so a new week appears in the week strip by itself and
the site runs all season with no manual step. Links are relative, so it works under the Pages
sub-path. `data/backtest.json` moved out of the gitignored `data/processed/` and is committed.

## First version, and why it was replaced

The first build put everything in one page: a seven-column table with a spread gauge in every
row, past weeks stacked underneath. Two reviewers found it cluttered and hard to read, and asked
for something graphical per game with a plain table at the end. They were right — it was the
dense-ledger trap. The redesign:

1. **A card per game**, in a responsive grid. The matchup, kickoff in ET, our pick in words
   ("SEA to win, 67%"), a probability bar split between the two teams with a small triangle
   marking Vegas's number, and the spreads in team terms ("We say SEA by 5.1; the line is SEA by
   3.0") instead of signed numbers. Once played: the score, who won, ✓ right / ✗ wrong, and
   whether the pick covered.
2. **A compact table of the week** at the end, one row per game, same words.
3. **A week strip** at the top of every page instead of scrolling: This week, Week 1, Week 2, …,
   Season. The current page is highlighted.
4. Provenance (published when, how long before kickoff, model commit, links to the files)
   collapsed under "Where this came from" on each week page.

## Design

The subject is a verifiable record, so evidence sits with the data rather than in a footer,
and the hero is the ledger, not a headline number. One element carries the visual weight — the
per-game spread gauge — and everything else is quiet. Bricolage Grotesque for headings and
numbers (tabular figures), Source Sans 3 for body. Pale bone-grey ground, slate ink, model blue
and Vegas bronze; the two series colours were validated for colour-vision-deficiency separation
and contrast on both the light and dark surface with the dataviz validator, and the dark pair
was re-stepped once to pass the lightness band. Green/red are reserved for results and always
paired with a glyph. Dark mode follows the system setting. No all-caps labels, no decorative
motion.

## Verified

- Renders the real Week 1 data: 16 cards, 16 table rows, three pages, all provenance links
  resolve to the committed paths. HTML parses with balanced tags on every page. No `<script>`,
  no `http://`, no `apiKey`.
- 10 tests covering every data state: no predictions, one page per week with the index as the
  latest, an upcoming card in plain words, an away pick showing the away team's confidence, a
  graded card with score and verdict, a late pass superseding the early one, the season page
  with a hollow small-n calibration bin, the probability bar, and a real build.
- `%-d`/`%-I` in `strftime` are glibc-only and crash on Windows; dates are formatted by hand.
- `pytest` → 81 passed offline; `ruff` clean.

## Wired into automation

Each cron workflow now runs `site_build` after its data step and commits `site/` with the data,
so the page updates on every publish and every grade. The exit criterion — the page correctly
rendering a full round trip of predicted week, baseline and graded result — is met with test
fixtures; the real round trip completes when Week 1 is graded on Tuesday.

## Deferred

- The first live version was reviewed in a browser; the redesign's structure is verified and
  its layout should get the same look once it deploys.
- Per-team pages, favourite/underdog breakdowns: the per-game grades support them; not built
  until there is a season's worth of data to make them meaningful.
- Fonts load from Google Fonts with system fallbacks. Self-hosting them would remove the one
  external request; not needed for $0.
