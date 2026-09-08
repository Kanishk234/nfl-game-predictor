"""Render the static site from the committed data.

    python -m nfl_predict.site_build

Reads `data/predictions/`, `data/results/` and `data/backtest.json` and writes `site/`:

- `index.html`                the current week (the latest week with a prediction file)
- `weeks/<season>_<ww>.html`  every week, current one included, so nothing has to scroll
- `season.html`               season-to-date record, charts, the pre-live backtest, how to verify

Every page is regenerated on every run; a new week simply appears in the week strip. No
JavaScript, no fetches: the site is a view over the committed JSON and links back to it. Charts
are inline SVG drawn here. All links are relative so the site works under a sub-path.
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from nfl_predict.grade import HISTORY_PATH, RESULTS_DIR
from nfl_predict.model.train import BACKTEST_PATH
from nfl_predict.predict import PREDICTIONS_DIR

SITE_DIR = Path("site")
REPO_URL = "https://github.com/Kanishk234/nfl-game-predictor"
ET = ZoneInfo("America/New_York")

#: Chart colours, validated for colour-vision-deficiency separation and contrast on both
#: surfaces with the dataviz validator.
MODEL_LIGHT, VEGAS_LIGHT = "#1F6FB5", "#9A6A1F"
MODEL_DARK, VEGAS_DARK = "#3F7FC6", "#B8891F"


# ----------------------------------------------------------------------------- data loading

def load_predictions() -> list[dict]:
    recs = [json.loads(p.read_text()) for p in sorted(PREDICTIONS_DIR.glob("*_*_*.json"))]
    return sorted(recs, key=lambda r: (r["season"], r["week"], r["generated_at_utc"]))


def load_results() -> dict[tuple[int, int], dict]:
    out = {}
    for p in sorted(RESULTS_DIR.glob("*_*.json")):
        if p.name == HISTORY_PATH.name:
            continue
        r = json.loads(p.read_text())
        out[(r["season"], r["week"])] = r
    return out


def load_history() -> dict | None:
    return json.loads(HISTORY_PATH.read_text()) if HISTORY_PATH.exists() else None


def load_backtest() -> dict | None:
    return json.loads(BACKTEST_PATH.read_text()) if BACKTEST_PATH.exists() else None


# ----------------------------------------------------------------------------- helpers

def e(s) -> str:
    return html.escape(str(s))


def fmt_et(iso: str) -> str:
    """'Wed Sep 9, 8:20 pm ET'. Built by hand: the `%-d` / `%-I` flags are glibc-only."""
    d = datetime.fromisoformat(iso).astimezone(ET)
    hour = d.hour % 12 or 12
    return f"{d:%a %b} {d.day}, {hour}:{d:%M} {'am' if d.hour < 12 else 'pm'} ET"


def pct(x: float | None, nd: int = 0) -> str:
    return "—" if x is None else f"{x * 100:.{nd}f}%"


def by_team(margin: float, home: str, away: str, nd: int = 1) -> str:
    """A home-side margin in words: 'SEA by 5.1', 'NE by 2.0', or 'even'."""
    if abs(margin) < 0.05:
        return "even"
    return f"{home if margin > 0 else away} by {abs(margin):.{nd}f}"


def hours_before(seconds: int) -> str:
    h = seconds / 3600
    return f"{h:.0f} hours" if h >= 2 else f"{seconds // 60} minutes"


def official_rows(passes: list[dict]) -> list[tuple[str, dict]]:
    """The official prediction per game (latest pass generated before its kickoff), by kickoff."""
    chosen: dict[str, tuple[str, dict, datetime]] = {}
    for rec in passes:
        gen = datetime.fromisoformat(rec["generated_at_utc"])
        for row in rec["predictions"]:
            if gen >= datetime.fromisoformat(row["kickoff_utc"]):
                continue
            prev = chosen.get(row["game_id"])
            if prev is None or gen > prev[2]:
                chosen[row["game_id"]] = (rec["pass"], row, gen)
    return sorted(((p, r) for p, r, _ in chosen.values()), key=lambda pr: (pr[1]["kickoff_utc"], pr[1]["game_id"]))


def p_pick(p: dict) -> float:
    """Probability of the picked team, so cards and table read the same way."""
    return p["p_home"] if p["pick"] == p["home_team"] else 1 - p["p_home"]


# ----------------------------------------------------------------------------- svg pieces

def prob_bar(p_home: float, p_vegas: float | None, home: str, away: str) -> str:
    """Away share on the left, home share on the right, with a marker where Vegas puts it."""
    w, h = 300, 14
    split = w * (1 - p_home)
    parts = [f'<svg class="bar" viewBox="0 -1 {w} {h + 10}" role="img" aria-label="{e(away)} {pct(1 - p_home)}, {e(home)} {pct(p_home)}'
             + (f", Vegas has {e(home)} at {pct(p_vegas)}" if p_vegas is not None else "") + '">',
             f'<rect class="bar-away" x="0" y="0" width="{split:.1f}" height="{h}" rx="3"/>',
             f'<rect class="bar-home" x="{split + 2:.1f}" y="0" width="{w - split - 2:.1f}" height="{h}" rx="3"/>']
    if p_vegas is not None:
        vx = w * (1 - p_vegas)
        parts.append(f'<polygon class="bar-vegas" points="{vx:.1f},{h + 1} {vx - 5:.1f},{h + 8} {vx + 5:.1f},{h + 8}"/>')
    parts.append("</svg>")
    return "".join(parts)


def line_chart(series: dict[str, list[tuple[int, float]]], y_label: str, y_min: float, y_max: float, ref: float | None = None) -> str:
    w, h, ml, mr, mt, mb = 640, 240, 44, 90, 16, 32
    weeks = sorted({wk for pts in series.values() for wk, _ in pts})
    if not weeks:
        return ""
    x0, x1 = min(weeks), max(weeks)
    def X(wk): return ml + (0 if x1 == x0 else (wk - x0) / (x1 - x0)) * (w - ml - mr)
    def Y(v): return mt + (1 - (v - y_min) / (y_max - y_min)) * (h - mt - mb)
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{e(y_label)} by week">']
    for t in (y_min, (y_min + y_max) / 2, y_max):
        out.append(f'<line class="grid" x1="{ml}" y1="{Y(t):.1f}" x2="{w - mr}" y2="{Y(t):.1f}"/>'
                   f'<text class="tick" x="{ml - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t * 100:.0f}%</text>')
    if ref is not None:
        out.append(f'<line class="ref" x1="{ml}" y1="{Y(ref):.1f}" x2="{w - mr}" y2="{Y(ref):.1f}"/>')
    for wk in weeks:
        out.append(f'<text class="tick" x="{X(wk):.1f}" y="{h - 10}" text-anchor="middle">wk {wk}</text>')
    for name, pts in series.items():
        cls = "model" if name == "Model" else "vegas"
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(wk):.1f},{Y(v):.1f}" for i, (wk, v) in enumerate(pts))
        out.append(f'<path class="line {cls}" d="{d}"/>')
        for wk, v in pts:
            out.append(f'<circle class="dot {cls}" cx="{X(wk):.1f}" cy="{Y(v):.1f}" r="4"/>')
        wk, v = pts[-1]
        out.append(f'<text class="label {cls}" x="{X(wk) + 10:.1f}" y="{Y(v) + 4:.1f}">{e(name)} {v * 100:.0f}%</text>')
    out.append("</svg>")
    return "".join(out)


def calibration_chart(cal: dict[str, list[dict]]) -> str:
    """Reliability diagram. Bins with fewer than 10 games are drawn hollow."""
    w, h, ml, mr, mt, mb = 360, 320, 44, 16, 16, 40
    def X(p): return ml + p * (w - ml - mr)
    def Y(p): return mt + (1 - p) * (h - mt - mb)
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="calibration: predicted vs observed">',
           f'<line class="ref" x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(1):.1f}" y2="{Y(1):.1f}"/>']
    for t in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<text class="tick" x="{ml - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t * 100:.0f}%</text>'
                   f'<text class="tick" x="{X(t):.1f}" y="{h - 22}" text-anchor="middle">{t * 100:.0f}%</text>')
    out.append(f'<text class="tick" x="{(ml + w - mr) / 2:.1f}" y="{h - 6}" text-anchor="middle">predicted home win</text>')
    for name in ("model", "vegas"):
        for b in [b for b in cal.get(name, []) if b["n"] and b["mean_predicted"] is not None]:
            hollow = ' fill="none"' if b["n"] < 10 else ""
            out.append(f'<circle class="dot {name}" cx="{X(b["mean_predicted"]):.1f}" cy="{Y(b["observed"]):.1f}" r="{5 if b["n"] >= 10 else 4}"{hollow}>'
                       f'<title>{name}: predicted {b["mean_predicted"] * 100:.0f}%, observed {b["observed"] * 100:.0f}% (n={b["n"]})</title></circle>')
    out.append("</svg>")
    return "".join(out)


def legend_two() -> str:
    return '<p class="legend"><span class="swatch model"></span>Model <span class="swatch vegas"></span>Vegas</p>'


# ----------------------------------------------------------------------------- game cards

def game_card(pass_name: str, p: dict, g: dict | None) -> str:
    home, away = p["home_team"], p["away_team"]
    v = p.get("vegas")
    p_home = p["p_home"]
    vegas_line = f'the line is {e(by_team(v["spread_line"], home, away))}' if v else "no line yet"
    vegas_prob = f' Vegas has {e(home)} at {pct(v["p_home_moneyline"])}.' if v else ""

    if g:
        ok = g["model"]["correct"]
        verdict = "tie" if ok is None else ('<span class="hit">✓ right</span>' if ok else '<span class="miss">✗ wrong</span>')
        ats_txt = {"win": "covered the spread", "loss": "did not cover", "push": "push against the spread"}.get(g["model"].get("ats"), "")
        winner = e(g["winner"]) if g["winner"] != "tie" else "tie"
        outcome = (f'<p class="final"><span class="score">{e(away)} {g["away_score"]}, {e(home)} {g["home_score"]}</span> '
                   f'<span class="verdict">{winner} won, {verdict}</span>'
                   + (f'<br><small>{e(ats_txt)}</small>' if ats_txt else "") + '</p>')
        status = "played"
    else:
        outcome, status = "", "upcoming"

    return f'''<article class="card {status}">
  <header><h3>{e(away)} at {e(home)}</h3><time datetime="{e(p["kickoff_utc"])}">{e(fmt_et(p["kickoff_utc"]))}</time></header>
  <p class="pick"><strong>{e(p["pick"])}</strong> to win <span class="conf">{pct(p_pick(p))}</span></p>
  {prob_bar(p_home, v["p_home_moneyline"] if v else None, home, away)}
  <p class="bar-ends"><span>{e(away)} {pct(1 - p_home)}</span><span>{e(home)} {pct(p_home)}</span></p>
  <p class="spread">We say {e(by_team(p["pred_margin"], home, away))}; {vegas_line}.{vegas_prob}</p>
  {outcome}
</article>'''


def week_table(rows: list[tuple[str, dict]], graded: dict[str, dict]) -> str:
    out = ['<table class="ledger"><thead><tr><th>Kickoff</th><th>Game</th><th>Pick</th><th class="num">Win prob</th>',
           '<th>Our spread</th><th>Line</th><th class="num">Vegas prob</th><th>Pass</th><th>Result</th></tr></thead><tbody>']
    for pass_name, p in rows:
        home, away = p["home_team"], p["away_team"]
        g = graded.get(p["game_id"]); v = p.get("vegas")
        if g:
            ok = g["model"]["correct"]
            mark = "" if ok is None else (' <span class="hit">✓</span>' if ok else ' <span class="miss">✗</span>')
            res = f'{e(g["winner"])}{mark} <small>{g["away_score"]}–{g["home_score"]}</small>'
        else:
            res = '<span class="pending">—</span>'
        out.append(f'<tr><td>{e(fmt_et(p["kickoff_utc"]))}</td><td>{e(away)} at {e(home)}</td><td><strong>{e(p["pick"])}</strong></td>'
                   f'<td class="num">{pct(p_pick(p))}</td><td>{e(by_team(p["pred_margin"], home, away))}</td>'
                   f'<td>{e(by_team(v["spread_line"], home, away)) if v else "—"}</td><td class="num">{pct(v["p_home_moneyline"]) if v else "—"}</td>'
                   f'<td>{e(pass_name)}</td><td>{res}</td></tr>')
    out.append("</tbody></table>")
    return "".join(out)


def provenance(passes: list[dict]) -> str:
    items = []
    for rec in passes:
        m, gate = rec["model"], rec["gate"]
        items.append(
            f'<li><strong>{e(rec["pass"].capitalize())} pass</strong> published '
            f'<time datetime="{e(rec["generated_at_utc"])}">{e(fmt_et(rec["generated_at_utc"]))}</time>, '
            f'{e(hours_before(gate["seconds_before_first_kickoff"]))} before the first kickoff it covers, '
            f'{rec["n_games"]} games. Model trained on {m["n_train"]:,} games through {e(m["trained_through_game"])}, '
            f'code <a href="{REPO_URL}/commit/{e(m["code_version"])}">{e(m["code_version"])}</a>. '
            f'<a href="{REPO_URL}/blob/main/data/predictions/{rec["season"]}_{rec["week"]:02d}_{e(rec["pass"])}.json">prediction file</a>, '
            f'<a href="{REPO_URL}/blob/main/data/odds/{rec["season"]}_{rec["week"]:02d}_{e(rec["pass"])}.json">odds snapshot</a>.</li>')
    return f'<details class="prov"><summary>Where this came from</summary><ul>{"".join(items)}</ul></details>'


def summary_strip(s: dict | None, note: str) -> str:
    if not s or not s.get("n"):
        return ""
    m, v = s["model"], s.get("vegas")
    cells = [("Games", f'{s["n"]}'), ("Our picks right", pct(m["accuracy"])), ("Vegas right", pct(v["accuracy"]) if v else "—"),
             ("Our spread error", f'{m["spread_mae"]:.1f} pts'), ("Line's error", f'{v["spread_mae"]:.1f} pts' if v else "—")]
    if "ats" in m:
        a = m["ats"]; cells.append(("Against the spread", f'{a["ats_w"]}–{a["ats_l"]}–{a["ats_push"]}'))
    return ('<dl class="strip">' + "".join(f'<div><dt>{e(k)}</dt><dd>{val}</dd></div>' for k, val in cells)
            + f'</dl><p class="fine">{e(note)}</p>')


# ----------------------------------------------------------------------------- pages

def week_strip(weeks: list[tuple[int, int]], current: tuple[int, int] | None, root: str) -> str:
    links = "".join(
        f'<a href="{root}weeks/{s}_{w:02d}.html"{" aria-current=\"page\"" if (s, w) == current else ""}>Week {w}</a>'
        for s, w in weeks)
    return (f'<nav class="weeks"><a href="{root}index.html">This week</a>{links}'
            f'<a href="{root}season.html"{" aria-current=\"page\"" if current is None else ""}>Season</a></nav>')


def week_body(season: int, week: int, passes: list[dict], result: dict | None) -> str:
    rows = official_rows(passes)
    graded = {g["game_id"]: g for g in (result or {}).get("games", [])}
    status = "" if not result else (", complete" if result["complete"] else f', {result["n_graded"]} of {result["n_games"]} played')
    cards = "".join(game_card(pn, p, graded.get(p["game_id"])) for pn, p in rows)
    return f'''<h2>Week {week}, {season}{e(status)}</h2>
<p class="how">One card per game: who we pick, how confident we are, and where the Vegas line sits. The bar splits the win
   probability between the two teams; the small triangle under it is Vegas's number for comparison.</p>
{summary_strip(result["summary"] if result else None, "Official predictions only: the latest pass published before each game's kickoff.")}
<div class="cards">{cards}</div>
<h3 class="table-title">All games this week</h3>
{week_table(rows, graded)}
{provenance(passes)}'''


def season_body(history: dict | None, season: int | None, backtest: dict | None) -> str:
    parts = []
    s = (history or {}).get("seasons", {}).get(str(season)) if season else None
    if s and s["summary"].get("n"):
        weeks = [w for w in s["weeks"] if w["summary"].get("n")]
        series = {"Model": [(w["week"], w["summary"]["model"]["accuracy"]) for w in weeks],
                  "Vegas": [(w["week"], w["summary"]["vegas"]["accuracy"]) for w in weeks if "vegas" in w["summary"]]}
        chart = line_chart(series, "picks right", 0.3, 1.0, ref=0.5) if len(weeks) >= 2 else ""
        cal = calibration_chart(s["calibration"]) if s["summary"]["n"] >= 30 else ""
        parts.append(f'<h2>{season} season to date</h2>' + summary_strip(s["summary"], "Every graded game so far, official predictions only.")
                     + (f'<figure><figcaption>Share of picks right, by week.</figcaption>{chart}{legend_two()}</figure>' if chart else "")
                     + (f'<figure><figcaption>Calibration so far. Hollow points are bins with fewer than 10 games.</figcaption>{cal}{legend_two()}</figure>' if cal
                        else '<p class="fine">Charts appear once there are enough graded weeks to mean something.</p>'))
    else:
        parts.append(f'<h2>{season or ""} season</h2><p>No games graded yet. Results are recorded the Tuesday after each week.</p>')
    if backtest:
        parts.append(backtest_section(backtest))
    parts.append(about_section())
    return "".join(parts)


def backtest_section(bt: dict) -> str:
    o = bt["overall"]; m, v = o["model"], o["vegas"]
    seasons = ", ".join(str(s) for s in bt["holdout_seasons"])
    rows = "".join(
        f'<tr><td>{e(s)}</td><td class="num">{pct(x["model"]["win"]["accuracy"], 1)}</td><td class="num">{pct(x["vegas"]["win"]["accuracy"], 1)}</td>'
        f'<td class="num">{x["model"]["spread"]["mae"]:.2f}</td><td class="num">{x["vegas"]["spread"]["mae"]:.2f}</td></tr>'
        for s, x in bt["by_season"].items())
    return f'''<section id="backtest">
  <h2>Before going live: {seasons}</h2>
  <p>The same model, replayed week by week over five seasons it never saw during tuning: {bt["n_folds"]} retrains,
     {bt["n_games"]:,} games, each fit using only games that had finished before that week's first kickoff.
     Vegas is the closing line on the same games. <strong>The model does not beat Vegas.</strong> It trails by
     {(v["win"]["accuracy"] - m["win"]["accuracy"]) * 100:.1f} points of pick accuracy and {m["spread"]["mae"] - v["spread"]["mae"]:.2f} points of
     spread error. That is the bar the live season is measured against.</p>
  <div class="two-col">
  <table class="compact"><thead><tr><th>Season</th><th class="num">Our picks right</th><th class="num">Vegas</th><th class="num">Our spread error</th><th class="num">Vegas</th></tr></thead>
  <tbody>{rows}<tr class="total"><td>All</td><td class="num">{pct(m["win"]["accuracy"], 1)}</td><td class="num">{pct(v["win"]["accuracy"], 1)}</td>
  <td class="num">{m["spread"]["mae"]:.2f}</td><td class="num">{v["spread"]["mae"]:.2f}</td></tr></tbody></table>
  <figure><figcaption>Calibration on those {bt["n_games"]:,} games. On the diagonal, a stated probability matched how often it came true.</figcaption>
  {calibration_chart(bt["calibration"])}{legend_two()}</figure>
  </div>
  <p class="fine"><a href="{REPO_URL}/blob/main/data/backtest.json">backtest.json</a>, regenerated by every retrain, and the
     <a href="{REPO_URL}/blob/main/docs/reports/phase2_model_backtest.md">full report</a> including everything tried and rejected.</p>
</section>'''


def about_section() -> str:
    return f'''<section id="about">
  <h2>How to check any of this</h2>
  <p>Every prediction is a JSON file committed to a public repository before kickoff, stamped with the time it was
     generated and the exact code and data that produced it. Vegas lines are fetched and frozen in the same instant,
     into a second file beside it. After the games, a grader reads those files and writes the results; it cannot edit
     a prediction. If a prediction were ever wrong or late, the fix is a new file with a new timestamp; the old one stays.</p>
  <p>The model is a logistic regression and a ridge regression on 30 features: Elo, rolling EPA form from play-by-play,
     quarterback ratings and draft position, rest, and schedule facts. It retrains before every pass on every completed
     game since 2002. Nothing it uses is knowable only after a kickoff, and a test suite asserts that on every commit.</p>
  <p><a href="{REPO_URL}">Repository</a>, <a href="{REPO_URL}/tree/main/data/predictions">predictions</a>,
     <a href="{REPO_URL}/tree/main/data/odds">odds snapshots</a>, <a href="{REPO_URL}/tree/main/data/results">results</a>,
     <a href="{REPO_URL}/actions">the scheduled jobs</a>.</p>
</section>'''


CSS = f"""
:root {{ --bg: #F4F5F1; --panel: #FFFFFF; --ink: #1E2A38; --muted: #62707E; --rule: #D9DDD6;
         --model: {MODEL_LIGHT}; --vegas: {VEGAS_LIGHT}; --away: #C9D2DA; --hit: #2E7D4F; --miss: #B23A3A; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #171A1C; --panel: #1F2326; --ink: #E8EAE6; --muted: #9AA4AD; --rule: #33393E;
           --model: {MODEL_DARK}; --vegas: {VEGAS_DARK}; --away: #3A434B; --hit: #5FB37F; --miss: #E06C6C; }} }}
* {{ box-sizing: border-box; }}
html {{ color-scheme: light dark; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 17px/1.5 "Source Sans 3", "Segoe UI", system-ui, sans-serif;
        font-variant-numeric: tabular-nums; }}
main {{ max-width: 72rem; margin: 0 auto; padding: 1.5rem 1.25rem 5rem; }}
h1, h2, h3, dd, .num, .pick strong, .conf, .score {{ font-family: "Bricolage Grotesque", "Source Sans 3", system-ui, sans-serif; }}
h1 {{ font-size: 1.6rem; margin: 0; font-weight: 600; }}
h1 a {{ color: inherit; text-decoration: none; }}
h2 {{ font-size: 1.6rem; margin: 1.75rem 0 .5rem; font-weight: 600; }}
h3 {{ font-size: 1.2rem; margin: 0; font-weight: 600; }}
p {{ max-width: 66ch; }}
a {{ color: var(--model); text-underline-offset: .15em; }}
a:focus-visible, summary:focus-visible {{ outline: 2px solid var(--model); outline-offset: 3px; }}
.top {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: .5rem 1.5rem; padding-bottom: .75rem; border-bottom: 1px solid var(--rule); }}
.top .lede {{ color: var(--muted); margin: 0; }}
.weeks {{ display: flex; flex-wrap: wrap; gap: .35rem; margin: .9rem 0 0; }}
.weeks a {{ text-decoration: none; padding: .3rem .7rem; border: 1px solid var(--rule); border-radius: 999px; color: var(--ink); font-size: .95rem; }}
.weeks a[aria-current] {{ background: var(--ink); color: var(--bg); border-color: var(--ink); }}
.how, .fine, small {{ color: var(--muted); font-size: .95rem; }}
.strip {{ display: flex; flex-wrap: wrap; gap: .5rem 2rem; margin: 1rem 0 .25rem; padding: .9rem 1.1rem; background: var(--panel);
          border: 1px solid var(--rule); border-radius: 8px; }}
.strip div {{ min-width: 6rem; }} .strip dt {{ font-size: .85rem; color: var(--muted); }}
.strip dd {{ margin: 0; font-size: 1.35rem; font-weight: 600; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(21rem, 1fr)); gap: 1rem; margin: 1.25rem 0 2.5rem; }}
.card {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 1rem 1.1rem 1.1rem; }}
.card header {{ display: flex; justify-content: space-between; align-items: baseline; gap: .5rem; flex-wrap: wrap; }}
.card time {{ color: var(--muted); font-size: .9rem; }}
.pick {{ margin: .7rem 0 .5rem; font-size: 1.05rem; }}
.pick strong {{ font-size: 1.6rem; margin-right: .2rem; }} .conf {{ font-size: 1.25rem; font-weight: 600; margin-left: .25rem; }}
.bar {{ display: block; width: 100%; height: auto; overflow: visible; }}
.bar-away {{ fill: var(--away); }} .bar-home {{ fill: var(--model); }} .bar-vegas {{ fill: var(--vegas); }}
.bar-ends {{ display: flex; justify-content: space-between; margin: .3rem 0 .4rem; font-size: .9rem; color: var(--muted); }}
.spread {{ margin: .4rem 0 0; font-size: .95rem; }}
.final {{ margin: .8rem 0 0; padding-top: .7rem; border-top: 1px dashed var(--rule); }}
.score {{ font-weight: 600; margin-right: .5rem; }} .verdict {{ white-space: nowrap; }}
.hit {{ color: var(--hit); font-weight: 700; }} .miss {{ color: var(--miss); font-weight: 700; }}
.pending {{ color: var(--muted); }}
.table-title {{ margin: 2rem 0 .5rem; }}
table {{ border-collapse: collapse; width: 100%; }}
.ledger {{ display: block; overflow-x: auto; font-size: .95rem; }}
.ledger th {{ text-align: left; font-weight: 600; font-size: .85rem; color: var(--muted); border-bottom: 1px solid var(--ink); padding: .4rem .6rem; white-space: nowrap; }}
.ledger td {{ padding: .5rem .6rem; border-bottom: 1px solid var(--rule); white-space: nowrap; }}
.ledger .num, .compact .num {{ text-align: right; }}
.prov {{ margin: 1.5rem 0; color: var(--muted); font-size: .95rem; max-width: 72ch; }}
.prov summary {{ cursor: pointer; }} .prov ul {{ padding-left: 1.2rem; }} .prov li {{ margin: .4rem 0; }}
.legend {{ font-size: .9rem; color: var(--muted); margin: .5rem 0 0; }}
.swatch {{ display: inline-block; width: .8em; height: .8em; border-radius: 50%; margin: 0 .35em 0 1em; vertical-align: -.05em; }}
.swatch.model {{ background: var(--model); }} .swatch.vegas {{ background: var(--vegas); }} .legend .swatch:first-child {{ margin-left: 0; }}
.chart {{ width: 100%; max-width: 640px; height: auto; display: block; }}
.chart .grid {{ stroke: var(--rule); stroke-width: 1; }} .chart .ref {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 4; }}
.chart .tick, .chart .label {{ font-size: 12px; fill: var(--muted); font-family: "Source Sans 3", system-ui, sans-serif; }}
.chart .line {{ fill: none; stroke-width: 2; }} .chart .line.model {{ stroke: var(--model); }} .chart .line.vegas {{ stroke: var(--vegas); }}
.chart .dot {{ stroke: var(--bg); stroke-width: 2; }} .chart .dot.model {{ fill: var(--model); }} .chart .dot.vegas {{ fill: var(--vegas); }}
.chart .dot[fill="none"] {{ fill: var(--bg) !important; stroke-width: 2; }} .chart .dot[fill="none"].model {{ stroke: var(--model); }} .chart .dot[fill="none"].vegas {{ stroke: var(--vegas); }}
.chart .label.model {{ fill: var(--model); }} .chart .label.vegas {{ fill: var(--vegas); }}
figure {{ margin: 1.5rem 0; }} figcaption {{ color: var(--muted); font-size: .95rem; margin-bottom: .5rem; max-width: 62ch; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; align-items: start; }}
.compact td, .compact th {{ padding: .35rem .5rem; border-bottom: 1px solid var(--rule); font-size: .95rem; }}
.compact th {{ text-align: left; color: var(--muted); font-weight: 600; font-size: .85rem; }} .compact .total td {{ font-weight: 600; border-top: 1px solid var(--ink); }}
footer {{ margin-top: 3rem; }}
@media (max-width: 720px) {{ .two-col {{ grid-template-columns: 1fr; }} .cards {{ grid-template-columns: 1fr; }} body {{ font-size: 16px; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
"""


def page(title: str, body: str, weeks: list[tuple[int, int]], current: tuple[int, int] | None, root: str, now: datetime) -> str:
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<meta name="description" content="NFL picks, win probabilities and spreads, published before kickoff and graded against the Vegas line.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600&family=Source+Sans+3:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<main>
<div class="top"><h1><a href="{root}index.html">nfl-predict</a></h1>
<p class="lede">Every NFL game picked before kickoff, written down where it cannot be changed, graded against Vegas.</p></div>
{week_strip(weeks, current, root)}
{body}
<footer class="fine"><p>Built {e(fmt_et(now.isoformat()))} from the committed data. No JavaScript, no tracking.</p></footer>
</main>
</body>
</html>'''


def render_site(predictions: list[dict], results: dict, history: dict | None, backtest: dict | None, now: datetime) -> dict[str, str]:
    """Every page of the site, keyed by relative path."""
    by_week: dict[tuple[int, int], list[dict]] = {}
    for rec in predictions:
        by_week.setdefault((rec["season"], rec["week"]), []).append(rec)
    weeks = sorted(by_week)
    latest = weeks[-1] if weeks else None
    season = latest[0] if latest else None
    pages = {}
    for s, w in weeks:
        pages[f"weeks/{s}_{w:02d}.html"] = page(f"Week {w}, {s}", week_body(s, w, by_week[(s, w)], results.get((s, w))), weeks, (s, w), "../", now)
    if latest:
        pages["index.html"] = page("nfl-predict", week_body(*latest, by_week[latest], results.get(latest)), weeks, latest, "", now)
    else:
        pages["index.html"] = page("nfl-predict", "<h2>No predictions published yet</h2><p>The first pass runs the Tuesday before Week 1.</p>",
                                   weeks, None, "", now)
    pages["season.html"] = page("Season", season_body(history, season, backtest), weeks, None, "", now)
    return pages


def main() -> int:
    now = datetime.now(UTC)
    pages = render_site(load_predictions(), load_results(), load_history(), load_backtest(), now)
    for rel, content in pages.items():
        path = SITE_DIR / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"wrote {len(pages)} pages under {SITE_DIR}/: " + ", ".join(sorted(pages)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
