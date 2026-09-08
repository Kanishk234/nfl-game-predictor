"""Render the static site from the committed data.

    python -m nfl_predict.site_build

Reads `data/predictions/`, `data/results/` and `data/backtest.json`, writes `site/index.html`.
The site is a view over those files and nothing else: no JavaScript, no fetches, every number
on the page traceable to a committed JSON file that the page links to. Charts are inline SVG
drawn here. Paths are relative so the page works under a sub-path (GitHub Pages) or a root.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from nfl_predict.grade import HISTORY_PATH, RESULTS_DIR
from nfl_predict.model.train import BACKTEST_PATH
from nfl_predict.predict import PREDICTIONS_DIR

SITE_DIR = Path("site")
REPO_URL = "https://github.com/Kanishk234/nfl-game-predictor"
ET = ZoneInfo("America/New_York")

#: Chart colours. Validated for CVD separation and contrast on both surfaces (dataviz checks).
MODEL_LIGHT, VEGAS_LIGHT = "#1F6FB5", "#9A6A1F"
MODEL_DARK, VEGAS_DARK = "#3F7FC6", "#B8891F"
GAUGE_HALF_RANGE = 14.0  # points shown either side of pick'em on the spread gauge


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


def fmt_signed(x: float, nd: int = 1) -> str:
    return f"{x:+.{nd}f}".replace("-", "−")


def pct(x: float | None, nd: int = 0) -> str:
    return "—" if x is None else f"{x * 100:.{nd}f}%"


def hours_before(seconds: int) -> str:
    h = seconds / 3600
    return f"{h:.0f} hours" if h >= 2 else f"{seconds // 60} minutes"


def official_rows(passes: list[dict]) -> dict[str, tuple[str, dict]]:
    """Mirror of grade.official_predictions, over the site's own loaded records."""
    chosen: dict[str, tuple[str, dict, datetime]] = {}
    for rec in passes:
        gen = datetime.fromisoformat(rec["generated_at_utc"])
        for row in rec["predictions"]:
            if gen >= datetime.fromisoformat(row["kickoff_utc"]):
                continue
            prev = chosen.get(row["game_id"])
            if prev is None or gen > prev[2]:
                chosen[row["game_id"]] = (rec["pass"], row, gen)
    return {g: (p, r) for g, (p, r, _) in chosen.items()}


# ----------------------------------------------------------------------------- svg pieces

def spread_gauge(model: float, vegas: float | None, actual: float | None) -> str:
    """Two markers on one axis: our margin (circle) and the line (diamond); the result as a
    thin bar once known. Positive = home side. Values beyond the range clamp to the edge."""
    w, h, pad = 220, 26, 10
    def x(v: float) -> float:
        v = max(-GAUGE_HALF_RANGE, min(GAUGE_HALF_RANGE, v))
        return pad + (v + GAUGE_HALF_RANGE) / (2 * GAUGE_HALF_RANGE) * (w - 2 * pad)
    mid = h / 2
    parts = [f'<svg class="gauge" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
             f'aria-label="our margin {fmt_signed(model)}' + (f', line {fmt_signed(vegas)}' if vegas is not None else "")
             + (f', actual {fmt_signed(actual, 0)}' if actual is not None else "") + '">',
             f'<line class="gauge-axis" x1="{pad}" y1="{mid}" x2="{w - pad}" y2="{mid}"/>',
             f'<line class="gauge-zero" x1="{x(0):.1f}" y1="{mid - 7}" x2="{x(0):.1f}" y2="{mid + 7}"/>']
    if actual is not None:
        parts.append(f'<rect class="gauge-actual" x="{x(actual) - 1.5:.1f}" y="{mid - 10}" width="3" height="20"/>')
    if vegas is not None:
        vx = x(vegas)
        parts.append(f'<polygon class="gauge-vegas" points="{vx:.1f},{mid - 6} {vx + 6:.1f},{mid} {vx:.1f},{mid + 6} {vx - 6:.1f},{mid}"/>')
    parts.append(f'<circle class="gauge-model" cx="{x(model):.1f}" cy="{mid}" r="5"/>')
    parts.append("</svg>")
    return "".join(parts)


def line_chart(series: dict[str, list[tuple[int, float]]], y_label: str, y_min: float, y_max: float, ref: float | None = None) -> str:
    """Small multiseries line chart. series: name -> [(week, value)]. Ends direct-labelled."""
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
    """Reliability diagram: predicted vs observed home-win rate per bin, both series, with the
    diagonal as the reference. Bins with fewer than 10 games are drawn hollow."""
    w, h, ml, mr, mt, mb = 360, 320, 44, 16, 16, 40
    def X(p): return ml + p * (w - ml - mr)
    def Y(p): return mt + (1 - p) * (h - mt - mb)
    out = [f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="calibration: predicted vs observed">',
           f'<line class="ref" x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(1):.1f}" y2="{Y(1):.1f}"/>']
    for t in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<text class="tick" x="{ml - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t * 100:.0f}%</text>'
                   f'<text class="tick" x="{X(t):.1f}" y="{h - 22}" text-anchor="middle">{t * 100:.0f}%</text>')
    out.append(f'<text class="tick" x="{(ml + w - mr) / 2:.1f}" y="{h - 6}" text-anchor="middle">predicted home win</text>')
    for name, cls in (("model", "model"), ("vegas", "vegas")):
        bins = [b for b in cal.get(name, []) if b["n"] and b["mean_predicted"] is not None]
        for b in bins:
            hollow = ' fill="none"' if b["n"] < 10 else ""
            out.append(f'<circle class="dot {cls}" cx="{X(b["mean_predicted"]):.1f}" cy="{Y(b["observed"]):.1f}" r="{5 if b["n"] >= 10 else 4}"{hollow}>'
                       f'<title>{name}: predicted {b["mean_predicted"] * 100:.0f}%, observed {b["observed"] * 100:.0f}% (n={b["n"]})</title></circle>')
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------------------- html sections

def legend() -> str:
    return ('<p class="legend"><span class="swatch model"></span>Model '
            '<span class="swatch vegas"></span>Vegas consensus '
            '<span class="swatch actual"></span>Final margin</p>')


def week_table(rows: dict[str, tuple[str, dict]], graded: dict[str, dict]) -> str:
    """The ledger: one row per game, sorted by kickoff."""
    items = sorted(rows.values(), key=lambda pr: (pr[1]["kickoff_utc"], pr[1]["game_id"]))
    out = ['<table class="ledger"><thead><tr>',
           '<th>Kickoff</th><th>Game</th><th>Our pick</th><th class="num">Win prob</th>',
           '<th>Spread, ours vs the line</th><th class="num">Vegas</th><th>Result</th></tr></thead><tbody>']
    for pass_name, p in items:
        g = graded.get(p["game_id"])
        v = p.get("vegas")
        actual = g["margin"] if g else None
        gauge = spread_gauge(p["pred_margin"], v["spread_line"] if v else None, actual)
        line_txt = f'{fmt_signed(p["pred_margin"])} vs {fmt_signed(v["spread_line"])}' if v else f'{fmt_signed(p["pred_margin"])}, no line'
        vegas_txt = pct(v["p_home_moneyline"] if v else None)
        if g:
            ok_m, ok_v = g["model"]["correct"], (g["vegas"] or {}).get("correct")
            def mark(ok):
                return "" if ok is None else (' <span class="hit">✓</span>' if ok else ' <span class="miss">✗</span>')
            score = f'{g["away_team"]} {g["away_score"]}, {g["home_team"]} {g["home_score"]}' if g["winner"] != "tie" else f'tie {g["home_score"]}–{g["away_score"]}'
            ats = g["model"].get("ats")
            result = (f'<span class="winner">{e(g["winner"])}</span>{mark(ok_m)}<br><small>{e(score)}'
                      + (f', ATS {ats}' if ats else "") + '</small>')
            vegas_txt += mark(ok_v)
        else:
            result = '<span class="pending">not played</span>'
        out.append(
            f'<tr><td class="kick"><time datetime="{e(p["kickoff_utc"])}">{e(fmt_et(p["kickoff_utc"]))}</time>'
            f'<br><small>{e(pass_name)} pass</small></td>'
            f'<td class="game">{e(p["away_team"])} at <strong>{e(p["home_team"])}</strong></td>'
            f'<td class="pick"><strong>{e(p["pick"])}</strong></td>'
            f'<td class="num">{pct(p["p_home"])}<br><small>home</small></td>'
            f'<td class="gauge-cell">{gauge}<br><small>{e(line_txt)}</small></td>'
            f'<td class="num">{vegas_txt}<br><small>home</small></td>'
            f'<td class="result">{result}</td></tr>')
    out.append("</tbody></table>")
    return "".join(out)


def provenance(passes: list[dict]) -> str:
    lines = []
    for rec in passes:
        m, gate = rec["model"], rec["gate"]
        lines.append(
            f'<li><strong>{e(rec["pass"].capitalize())} pass</strong> published '
            f'<time datetime="{e(rec["generated_at_utc"])}">{e(fmt_et(rec["generated_at_utc"]))}</time>, '
            f'{e(hours_before(gate["seconds_before_first_kickoff"]))} before the first kickoff it covers. '
            f'{rec["n_games"]} games. Model trained on {m["n_train"]:,} games through {e(m["trained_through_game"])}, '
            f'code <a href="{REPO_URL}/commit/{e(m["code_version"])}">{e(m["code_version"])}</a>. '
            f'<a href="{REPO_URL}/blob/main/data/predictions/{rec["season"]}_{rec["week"]:02d}_{e(rec["pass"])}.json">prediction file</a>, '
            f'<a href="{REPO_URL}/blob/main/data/odds/{rec["season"]}_{rec["week"]:02d}_{e(rec["pass"])}.json">odds snapshot</a>.</li>')
    return f'<ul class="provenance">{"".join(lines)}</ul>'


def summary_strip(s: dict, n_label: str) -> str:
    if not s or not s.get("n"):
        return ""
    m, v = s["model"], s.get("vegas")
    cells = [("Games", f'{s["n"]}'), ("Model straight up", pct(m["accuracy"])), ("Vegas straight up", pct(v["accuracy"]) if v else "—"),
             ("Model Brier", f'{m["brier"]:.3f}'), ("Vegas Brier", f'{v["brier"]:.3f}' if v else "—"),
             ("Spread error, ours", f'{m["spread_mae"]:.1f} pts'), ("Spread error, line", f'{v["spread_mae"]:.1f} pts' if v else "—")]
    if "ats" in m:
        a = m["ats"]; cells.append(("Against the spread", f'{a["ats_w"]}–{a["ats_l"]}–{a["ats_push"]}'))
    return ('<dl class="strip">' + "".join(f'<div><dt>{e(k)}</dt><dd>{v_}</dd></div>' for k, v_ in cells)
            + f'</dl><p class="strip-note">{e(n_label)}</p>')


def backtest_section(bt: dict) -> str:
    o = bt["overall"]; m, v = o["model"], o["vegas"]
    seasons = ", ".join(str(s) for s in bt["holdout_seasons"])
    rows = "".join(
        f'<tr><td>{e(s)}</td><td class="num">{pct(x["model"]["win"]["accuracy"], 1)}</td><td class="num">{pct(x["vegas"]["win"]["accuracy"], 1)}</td>'
        f'<td class="num">{x["model"]["win"]["brier"]:.3f}</td><td class="num">{x["vegas"]["win"]["brier"]:.3f}</td>'
        f'<td class="num">{x["model"]["spread"]["mae"]:.2f}</td><td class="num">{x["vegas"]["spread"]["mae"]:.2f}</td></tr>'
        for s, x in bt["by_season"].items())
    return f'''
<section id="backtest">
  <h2>Before going live: {seasons}</h2>
  <p>The same model, replayed week by week over five seasons it never saw during tuning — {bt["n_folds"]} retrains,
     {bt["n_games"]:,} games, each fit using only games that had finished before that week's first kickoff.
     Vegas is the closing line on the same games. <strong>The model does not beat Vegas</strong>; it trails by
     {(v["win"]["accuracy"] - m["win"]["accuracy"]) * 100:.1f} points of accuracy and {m["spread"]["mae"] - v["spread"]["mae"]:.2f} points of
     spread error. Those numbers are the bar the live season is measured against.</p>
  <div class="two-col">
  <table class="compact"><thead><tr><th>Season</th><th class="num">Model</th><th class="num">Vegas</th><th class="num">Brier</th><th class="num">Vegas</th><th class="num">Spread err</th><th class="num">Vegas</th></tr></thead>
  <tbody>{rows}<tr class="total"><td>All</td><td class="num">{pct(m["win"]["accuracy"], 1)}</td><td class="num">{pct(v["win"]["accuracy"], 1)}</td>
  <td class="num">{m["win"]["brier"]:.3f}</td><td class="num">{v["win"]["brier"]:.3f}</td><td class="num">{m["spread"]["mae"]:.2f}</td><td class="num">{v["spread"]["mae"]:.2f}</td></tr></tbody></table>
  <figure><figcaption>Calibration on those {bt["n_games"]:,} games. On the diagonal, a stated probability matched the observed rate.</figcaption>
  {calibration_chart(bt["calibration"])}{legend_two()}</figure>
  </div>
  <p class="fine"><a href="{REPO_URL}/blob/main/data/backtest.json">backtest.json</a>, regenerated by every retrain, and the
     <a href="{REPO_URL}/blob/main/docs/reports/phase2_model_backtest.md">full report</a> including everything that was tried and rejected.</p>
</section>'''


def legend_two() -> str:
    return '<p class="legend"><span class="swatch model"></span>Model <span class="swatch vegas"></span>Vegas</p>'


def season_section(history: dict, season: int) -> str:
    s = history["seasons"].get(str(season))
    if not s or not s["summary"].get("n"):
        return ""
    weeks = [w for w in s["weeks"] if w["summary"].get("n")]
    series = {"Model": [(w["week"], w["summary"]["model"]["accuracy"]) for w in weeks],
              "Vegas": [(w["week"], w["summary"]["vegas"]["accuracy"]) for w in weeks if "vegas" in w["summary"]]}
    chart = line_chart(series, "straight-up accuracy", 0.3, 1.0, ref=0.5) if len(weeks) >= 2 else ""
    cal = calibration_chart(s["calibration"]) if s["summary"]["n"] >= 30 else ""
    return f'''
<section id="season">
  <h2>{season} season to date</h2>
  {summary_strip(s["summary"], "Official predictions only: the latest pass published before each game's kickoff.")}
  {"<figure><figcaption>Straight-up accuracy by week.</figcaption>" + chart + legend_two() + "</figure>" if chart else ""}
  {"<figure><figcaption>Calibration so far. Hollow points are bins with fewer than 10 games.</figcaption>" + cal + legend_two() + "</figure>" if cal else "<p class='fine'>Calibration and the weekly chart appear once there are enough graded games to mean something.</p>"}
</section>'''


def week_section(season: int, week: int, passes: list[dict], result: dict | None, open_: bool) -> str:
    rows = official_rows(passes)
    graded = {g["game_id"]: g for g in (result or {}).get("games", [])}
    status = "" if not result else (" — complete" if result["complete"] else f' — {result["n_graded"]} of {result["n_games"]} played')
    body = (summary_strip(result["summary"], "This week, official predictions.") if result else "") + legend() + week_table(rows, graded) + provenance(passes)
    if open_:
        return f'<section id="this-week"><h2>Week {week}, {season}{e(status)}</h2>{body}</section>'
    return f'<details class="past-week"><summary><h3>Week {week}, {season}{e(status)}</h3></summary>{body}</details>'


def about_section() -> str:
    return f'''
<section id="about">
  <h2>How to check any of this</h2>
  <p>Every prediction is a JSON file committed to a public repository before kickoff, stamped with the time it was
     generated and the exact code and data that produced it. Vegas lines are fetched and frozen in the same instant,
     into a second file beside it. After the games, a grader reads those files and writes the results; it cannot edit
     a prediction. If a prediction were ever wrong or late, the fix is a new file with a new timestamp — the old one
     stays.</p>
  <p>The model is a logistic regression and a ridge regression on 30 features: Elo, rolling EPA form from play-by-play,
     quarterback ratings and draft position, rest, and schedule facts. It retrains before every pass on every completed
     game since 2002. Nothing it uses is knowable only after a kickoff, and a test suite asserts that on every commit.</p>
  <p><a href="{REPO_URL}">Repository</a> · <a href="{REPO_URL}/tree/main/data/predictions">predictions</a> ·
     <a href="{REPO_URL}/tree/main/data/odds">odds snapshots</a> · <a href="{REPO_URL}/tree/main/data/results">results</a> ·
     <a href="{REPO_URL}/actions">the scheduled jobs</a></p>
</section>'''


CSS = f"""
:root {{
  --bg: #F4F5F1; --panel: #FCFCFB; --ink: #1E2A38; --muted: #62707E; --rule: #D9DDD6;
  --model: {MODEL_LIGHT}; --vegas: {VEGAS_LIGHT}; --actual: #1E2A38; --hit: #2E7D4F; --miss: #B23A3A;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #171A1C; --panel: #1F2326; --ink: #E8EAE6; --muted: #9AA4AD; --rule: #33393E;
          --model: {MODEL_DARK}; --vegas: {VEGAS_DARK}; --actual: #E8EAE6; --hit: #5FB37F; --miss: #E06C6C; }}
}}
* {{ box-sizing: border-box; }}
html {{ color-scheme: light dark; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 17px/1.55 "Source Sans 3", "Segoe UI", system-ui, sans-serif;
        font-variant-numeric: tabular-nums; }}
main {{ max-width: 68rem; margin: 0 auto; padding: 2.5rem 1.25rem 5rem; }}
h1, h2, h3, dd, .num, .pick, .winner {{ font-family: "Bricolage Grotesque", "Source Sans 3", system-ui, sans-serif; }}
h1 {{ font-size: 2.6rem; line-height: 1.05; margin: 0 0 .5rem; letter-spacing: -.01em; font-weight: 600; }}
h2 {{ font-size: 1.55rem; margin: 3rem 0 .75rem; font-weight: 600; }}
h3 {{ font-size: 1.15rem; margin: 0; display: inline; font-weight: 600; }}
p {{ max-width: 62ch; }}
a {{ color: var(--model); text-underline-offset: .15em; }}
a:focus-visible, summary:focus-visible {{ outline: 2px solid var(--model); outline-offset: 3px; }}
.lede {{ font-size: 1.15rem; color: var(--muted); max-width: 58ch; }}
.strip {{ display: flex; flex-wrap: wrap; gap: .5rem 2rem; margin: 1rem 0 .25rem; padding: .9rem 1.1rem; background: var(--panel);
          border: 1px solid var(--rule); border-radius: 6px; }}
.strip div {{ min-width: 6rem; }} .strip dt {{ font-size: .85rem; color: var(--muted); }}
.strip dd {{ margin: 0; font-size: 1.35rem; font-weight: 600; }}
.strip-note, .fine, small {{ color: var(--muted); font-size: .9rem; }}
.legend {{ font-size: .9rem; color: var(--muted); margin: .75rem 0 .25rem; }}
.swatch {{ display: inline-block; width: .8em; height: .8em; border-radius: 50%; margin: 0 .35em 0 1em; vertical-align: -.05em; }}
.swatch.model {{ background: var(--model); }} .swatch.vegas {{ background: var(--vegas); border-radius: 0; transform: rotate(45deg) scale(.85); }}
.swatch.actual {{ background: var(--actual); width: .3em; border-radius: 1px; }}
.legend .swatch:first-child {{ margin-left: 0; }}
table {{ border-collapse: collapse; width: 100%; }}
.ledger {{ display: block; overflow-x: auto; }}
.ledger thead th {{ text-align: left; font-weight: 600; font-size: .85rem; color: var(--muted); border-bottom: 1px solid var(--ink); padding: .4rem .6rem; }}
.ledger td {{ padding: .7rem .6rem; border-bottom: 1px solid var(--rule); vertical-align: top; white-space: nowrap; }}
.ledger .num, .compact .num {{ text-align: right; }}
.ledger .kick {{ color: var(--muted); font-size: .95rem; }}
.pick strong, .winner {{ font-size: 1.1rem; }}
.gauge-cell {{ text-align: center; }} .gauge {{ display: block; margin: 0 auto; }}
.gauge-axis {{ stroke: var(--rule); stroke-width: 2; }} .gauge-zero {{ stroke: var(--muted); stroke-width: 1; }}
.gauge-model {{ fill: var(--model); stroke: var(--bg); stroke-width: 2; }} .gauge-vegas {{ fill: var(--vegas); stroke: var(--bg); stroke-width: 2; }}
.gauge-actual {{ fill: var(--actual); }}
.hit {{ color: var(--hit); font-weight: 700; }} .miss {{ color: var(--miss); font-weight: 700; }}
.pending {{ color: var(--muted); }}
.provenance {{ padding-left: 1.2rem; font-size: .95rem; color: var(--muted); max-width: 70ch; }}
.provenance li {{ margin: .4rem 0; }}
.past-week {{ border-top: 1px solid var(--rule); padding: .9rem 0; }}
.past-week summary {{ cursor: pointer; list-style: none; }} .past-week summary::before {{ content: "+ "; color: var(--muted); }}
.past-week[open] summary::before {{ content: "− "; }}
.chart {{ width: 100%; max-width: 640px; height: auto; display: block; }}
.chart .grid {{ stroke: var(--rule); stroke-width: 1; }} .chart .ref {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 4; }}
.chart .tick, .chart .label {{ font-size: 12px; fill: var(--muted); font-family: "Source Sans 3", system-ui, sans-serif; }}
.chart .line {{ fill: none; stroke-width: 2; }} .chart .line.model, .chart .dot.model, .chart .label.model {{ stroke: var(--model); }}
.chart .dot.model {{ fill: var(--model); }} .chart .dot.vegas {{ fill: var(--vegas); }} .chart .label.model {{ fill: var(--model); stroke: none; }}
.chart .line.vegas, .chart .dot.vegas, .chart .label.vegas {{ stroke: var(--vegas); }} .chart .label.vegas {{ fill: var(--vegas); stroke: none; }}
.chart .dot {{ stroke: var(--bg); stroke-width: 2; }} .chart .dot[fill="none"] {{ fill: var(--bg) !important; stroke-width: 2; }}
figure {{ margin: 1.5rem 0; }} figcaption {{ color: var(--muted); font-size: .95rem; margin-bottom: .5rem; max-width: 62ch; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; align-items: start; }}
.compact td, .compact th {{ padding: .35rem .5rem; border-bottom: 1px solid var(--rule); font-size: .95rem; }}
.compact th {{ text-align: left; color: var(--muted); font-weight: 600; font-size: .85rem; }} .compact .total td {{ font-weight: 600; border-top: 1px solid var(--ink); }}
@media (max-width: 720px) {{ .two-col {{ grid-template-columns: 1fr; }} h1 {{ font-size: 2rem; }} body {{ font-size: 16px; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
"""


def render(predictions: list[dict], results: dict, history: dict | None, backtest: dict | None, now: datetime) -> str:
    by_week: dict[tuple[int, int], list[dict]] = {}
    for rec in predictions:
        by_week.setdefault((rec["season"], rec["week"]), []).append(rec)
    weeks = sorted(by_week)
    latest = weeks[-1] if weeks else None
    season = latest[0] if latest else None

    this_week = week_section(*latest, by_week[latest], results.get(latest), open_=True) if latest else \
        '<section id="this-week"><h2>No predictions published yet</h2><p>The first pass runs the Tuesday before Week 1.</p></section>'
    past = "".join(week_section(s, w, by_week[(s, w)], results.get((s, w)), open_=False) for s, w in reversed(weeks[:-1]))
    past_html = f'<section id="past"><h2>Earlier weeks</h2>{past}</section>' if past else ""
    season_html = season_section(history, season) if history and season else ""
    bt_html = backtest_section(backtest) if backtest else ""

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nfl-predict</title>
<meta name="description" content="NFL win-probability and spread predictions, published before kickoff and graded against the Vegas line.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600&family=Source+Sans+3:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style>
</head>
<body>
<main>
<header>
  <h1>nfl-predict</h1>
  <p class="lede">A model picks every NFL game before kickoff, writes the pick down where it cannot be changed, and gets
     graded against the closing Vegas line. This page is the ledger.</p>
</header>
{this_week}
{season_html}
{past_html}
{bt_html}
{about_section()}
<footer class="fine"><p>Built {e(fmt_et(now.isoformat()))} from the committed data. No JavaScript, no tracking.</p></footer>
</main>
</body>
</html>'''


def main() -> int:
    from datetime import UTC
    now = datetime.now(UTC)
    page = render(load_predictions(), load_results(), load_history(), load_backtest(), now)
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {SITE_DIR / 'index.html'} ({len(page):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
