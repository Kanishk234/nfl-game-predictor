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

#: Primary colour, secondary colour, full name. From nflverse's teams table; fixed facts, so
#: hardcoded rather than downloaded at build time.
TEAM_COLORS = {
    "ARI": ("#97233F", "#000000", "Arizona Cardinals"), "ATL": ("#A71930", "#000000", "Atlanta Falcons"),
    "BAL": ("#241773", "#9E7C0C", "Baltimore Ravens"), "BUF": ("#00338D", "#C60C30", "Buffalo Bills"),
    "CAR": ("#0085CA", "#000000", "Carolina Panthers"), "CHI": ("#0B162A", "#E64100", "Chicago Bears"),
    "CIN": ("#FB4F14", "#000000", "Cincinnati Bengals"), "CLE": ("#FF3C00", "#311D00", "Cleveland Browns"),
    "DAL": ("#002244", "#B0B7BC", "Dallas Cowboys"), "DEN": ("#002244", "#FB4F14", "Denver Broncos"),
    "DET": ("#0076B6", "#B0B7BC", "Detroit Lions"), "GB": ("#203731", "#FFB612", "Green Bay Packers"),
    "HOU": ("#03202F", "#A71930", "Houston Texans"), "IND": ("#002C5F", "#A5ACAF", "Indianapolis Colts"),
    "JAX": ("#006778", "#000000", "Jacksonville Jaguars"), "KC": ("#E31837", "#FFB612", "Kansas City Chiefs"),
    "LA": ("#003594", "#FFD100", "Los Angeles Rams"), "LAC": ("#007BC7", "#FFC20E", "Los Angeles Chargers"),
    "LV": ("#000000", "#A5ACAF", "Las Vegas Raiders"), "MIA": ("#008E97", "#F58220", "Miami Dolphins"),
    "MIN": ("#4F2683", "#FFC62F", "Minnesota Vikings"), "NE": ("#002244", "#C60C30", "New England Patriots"),
    "NO": ("#D3BC8D", "#000000", "New Orleans Saints"), "NYG": ("#0B2265", "#A71930", "New York Giants"),
    "NYJ": ("#003F2D", "#000000", "New York Jets"), "PHI": ("#004C54", "#A5ACAF", "Philadelphia Eagles"),
    "PIT": ("#000000", "#FFB612", "Pittsburgh Steelers"), "SEA": ("#002244", "#69BE28", "Seattle Seahawks"),
    "SF": ("#AA0000", "#B3995D", "San Francisco 49ers"), "TB": ("#A71930", "#322F2B", "Tampa Bay Buccaneers"),
    "TEN": ("#4495D2", "#D50A0A", "Tennessee Titans"), "WAS": ("#5A1414", "#FFB612", "Washington Commanders"),
}


def team_color(abbr: str) -> str:
    return TEAM_COLORS.get(abbr, ("#62707E",))[0]


def team_nick(abbr: str) -> str:
    """'Seahawks' from 'Seattle Seahawks'; falls back to the abbreviation."""
    return TEAM_COLORS[abbr][2].rsplit(" ", 1)[-1] if abbr in TEAM_COLORS else abbr


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
    """Away share on the left, home share on the right. The picked team's share is drawn in that
    team's colour; the other side is neutral. A small triangle marks where Vegas puts it."""
    w, h = 300, 14
    split = w * (1 - p_home)
    pick_home = p_home >= 0.5
    away_fill = "var(--away)" if pick_home else team_color(away)
    home_fill = team_color(home) if pick_home else "var(--away)"
    parts = [f'<svg class="bar" viewBox="0 -1 {w} {h + 10}" role="img" aria-label="{e(away)} {pct(1 - p_home)}, {e(home)} {pct(p_home)}'
             + (f", Vegas has {e(home)} at {pct(p_vegas)}" if p_vegas is not None else "") + '">',
             f'<rect x="0" y="0" width="{split:.1f}" height="{h}" rx="3" fill="{away_fill}"/>',
             f'<rect x="{split + 2:.1f}" y="0" width="{w - split - 2:.1f}" height="{h}" rx="3" fill="{home_fill}"/>']
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

def spread_scale(ours: float, line: float | None, actual: float | None, home: str, away: str) -> str:
    """A labelled number line: away side on the left, home side on the right, 'even' in the
    middle. Our margin is a round marker labelled above the axis, the Vegas line a diamond
    labelled below, so the two can never overprint. The final margin, once known, is a bar."""
    w, h, pad, half = 300, 64, 34, 14.0
    axis_y = 36
    def x(v: float) -> float:
        v = max(-half, min(half, v))
        return pad + (v + half) / (2 * half) * (w - 2 * pad)
    def words(v: float) -> str:
        return by_team(v, home, away)
    parts = [f'<svg class="scale" viewBox="0 0 {w} {h}" role="img" aria-label="our margin {e(words(ours))}'
             + (f', Vegas line {e(words(line))}' if line is not None else "") + (f', final {e(words(actual))}' if actual is not None else "") + '">',
             f'<line class="scale-axis" x1="{pad}" y1="{axis_y}" x2="{w - pad}" y2="{axis_y}"/>',
             f'<line class="scale-zero" x1="{x(0):.1f}" y1="{axis_y - 6}" x2="{x(0):.1f}" y2="{axis_y + 6}"/>',
             f'<text class="scale-end" x="{pad - 6}" y="{axis_y + 4}" text-anchor="end">{e(away)}</text>',
             f'<text class="scale-end" x="{w - pad + 6}" y="{axis_y + 4}">{e(home)}</text>',
             f'<text class="scale-tick" x="{x(0):.1f}" y="{h - 2}" text-anchor="middle">even</text>']
    if actual is not None:
        parts.append(f'<rect class="scale-actual" x="{x(actual) - 1.5:.1f}" y="{axis_y - 9}" width="3" height="18"/>')
    ox = x(ours)
    parts.append(f'<circle class="scale-ours" cx="{ox:.1f}" cy="{axis_y}" r="5.5"/>')
    parts.append(f'<text class="scale-label" x="{ox:.1f}" y="{axis_y - 12}" text-anchor="middle">Us: {e(words(ours))}</text>')
    if line is not None:
        vx = x(line)
        parts.append(f'<polygon class="scale-vegas" points="{vx:.1f},{axis_y - 6} {vx + 6:.1f},{axis_y} {vx:.1f},{axis_y + 6} {vx - 6:.1f},{axis_y}"/>')
        parts.append(f'<text class="scale-label vegas" x="{vx:.1f}" y="{axis_y + 20}" text-anchor="middle">Vegas: {e(words(line))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def plain_call(p: dict) -> str:
    """The prediction in one sentence each for us and for Vegas."""
    home, away = p["home_team"], p["away_team"]
    m = abs(p["pred_margin"])
    fav = home if p["pred_margin"] >= 0 else away
    ours = f"Too close to call, {fav} by a hair." if m < 1 else f"{fav} should win by about {round(m)}."
    v = p.get("vegas")
    if not v:
        return ours + " No Vegas line yet."
    line = v["spread_line"]
    vegas = "Vegas calls it even." if abs(line) < 0.05 else f"Vegas has {home if line > 0 else away} by {abs(line):g}."
    return f"{ours} {vegas}"


def game_card(pass_name: str, p: dict, g: dict | None) -> str:
    home, away = p["home_team"], p["away_team"]
    v = p.get("vegas")
    p_home = p["p_home"]
    pick = p["pick"]
    if v and v["p_home_moneyline"] is not None:
        v_fav = home if v["p_home_moneyline"] >= 0.5 else away
        v_conf = v["p_home_moneyline"] if v_fav == home else 1 - v["p_home_moneyline"]
        vegas_row = (f'<div class="call vegas"><span class="who">Vegas favorite</span>'
                     f'<span class="chip" style="background:{team_color(v_fav)}"></span><strong>{e(v_fav)}</strong>'
                     f'<span class="conf">{pct(v_conf)}</span><span class="towin">to win</span></div>')
    else:
        vegas_row = '<div class="call vegas"><span class="who">Vegas favorite</span><span class="none">no line yet</span></div>'

    if g:
        ok = g["model"]["correct"]
        verdict = "a tie" if ok is None else ('<span class="hit">✓ we were right</span>' if ok else '<span class="miss">✗ we were wrong</span>')
        ats_txt = {"win": "and our side covered the spread", "loss": "but our side did not cover the spread",
                   "push": "and the spread was a push"}.get(g["model"].get("ats"), "")
        winner = f'{e(g["winner"])} won' if g["winner"] != "tie" else "it was a tie"
        outcome = (f'<p class="final"><span class="score">{e(away)} {g["away_score"]}, {e(home)} {g["home_score"]}</span><br>'
                   f'{winner}, {verdict}' + (f' {e(ats_txt)}' if ats_txt else "") + '.</p>')
        actual, status = g["margin"], "played"
    else:
        outcome, actual, status = "", None, "upcoming"

    return f'''<article class="card {status}" style="--team:{team_color(pick)}">
  <header><h3>{e(team_nick(away))} at {e(team_nick(home))}</h3><time datetime="{e(p["kickoff_utc"])}">{e(fmt_et(p["kickoff_utc"]))}</time></header>
  <div class="calls">
    <div class="call ours"><span class="who">Our pick</span><span class="chip" style="background:{team_color(pick)}"></span>
      <strong>{e(pick)}</strong><span class="conf">{pct(p_pick(p))}</span><span class="towin">to win</span></div>
    {vegas_row}
  </div>
  {prob_bar(p_home, v["p_home_moneyline"] if v else None, home, away)}
  <p class="bar-ends"><span>{e(away)} {pct(1 - p_home)}</span><span>{e(home)} {pct(p_home)}</span></p>
  <p class="plain">{e(plain_call(p))}</p>
  {spread_scale(p["pred_margin"], v["spread_line"] if v else None, actual, home, away)}
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
<p class="how">One card per game. <strong>Our pick</strong> is the model's call; <strong>Vegas favorite</strong> is the betting
   market's, for comparison. The bar is the win probability, coloured for the team we pick; the small triangle under it is
   where Vegas puts it. The line at the bottom is the point spread: how much each of us expects the winner to win by.</p>
{summary_strip(result["summary"] if result else None, "Official predictions only: the latest pass published before each game's kickoff.")}
<div class="cards">{cards}</div>
<h3 class="table-title">All games this week</h3>
{week_table(rows, graded)}
{provenance(passes)}'''


def season_body(history: dict | None, season: int | None, backtest: dict | None) -> str:
    parts = ['<h2>Track record</h2>',
             '<p class="how">How the picks are doing. Updated every Tuesday, after the previous week\'s games are graded.</p>']
    s = (history or {}).get("seasons", {}).get(str(season)) if season else None
    if s and s["summary"].get("n"):
        weeks = [w for w in s["weeks"] if w["summary"].get("n")]
        series = {"Model": [(w["week"], w["summary"]["model"]["accuracy"]) for w in weeks],
                  "Vegas": [(w["week"], w["summary"]["vegas"]["accuracy"]) for w in weeks if "vegas" in w["summary"]]}
        chart = line_chart(series, "picks right", 0.3, 1.0, ref=0.5) if len(weeks) >= 2 else ""
        parts.append(f'<h3 class="sub">{season} season so far</h3>' + summary_strip(s["summary"], "Every graded game, official predictions only.")
                     + (f'<figure><figcaption>Share of picks that were right, week by week. The dashed line is a coin flip.</figcaption>{chart}{legend_two()}</figure>' if chart else ""))
    else:
        parts.append('<p class="empty">Nothing graded yet. The first results land the Tuesday after Week 1. Until then, the dry run below is the best guide to what to expect.</p>')
    if backtest:
        parts.append(backtest_section(backtest))
    parts.append(about_section())
    return "".join(parts)


def backtest_section(bt: dict) -> str:
    o = bt["overall"]; m, v = o["model"], o["vegas"]
    first, last = bt["holdout_seasons"][0], bt["holdout_seasons"][-1]
    rows = "".join(
        f'<tr><td>{e(s)}</td><td class="num">{pct(x["model"]["win"]["accuracy"], 1)}</td><td class="num">{pct(x["vegas"]["win"]["accuracy"], 1)}</td>'
        f'<td class="num">{x["model"]["spread"]["mae"]:.1f}</td><td class="num">{x["vegas"]["spread"]["mae"]:.1f}</td></tr>'
        for s, x in bt["by_season"].items())
    return f'''<section id="backtest">
  <h3 class="sub">The dry run: {first} to {last}</h3>
  <p>Before going live, we ran the model over the last five seasons as if they were happening week by week, never letting it
     see a game before it was played. Over {bt["n_games"]:,} games it <strong>picked the winner {pct(m["win"]["accuracy"], 1)} of the
     time; Vegas picked {pct(v["win"]["accuracy"], 1)}.</strong> Its spreads missed the final margin by {m["spread"]["mae"]:.1f} points per
     game on average; the Vegas line missed by {v["spread"]["mae"]:.1f}. A solid model that has not beaten the market. That is
     the bar for this season.</p>
  <details class="more"><summary>Season by season, and how honest the percentages are</summary>
  <table class="compact"><thead><tr><th>Season</th><th class="num">Our picks right</th><th class="num">Vegas</th><th class="num">Our spread miss</th><th class="num">Vegas</th></tr></thead>
  <tbody>{rows}<tr class="total"><td>All</td><td class="num">{pct(m["win"]["accuracy"], 1)}</td><td class="num">{pct(v["win"]["accuracy"], 1)}</td>
  <td class="num">{m["spread"]["mae"]:.1f}</td><td class="num">{v["spread"]["mae"]:.1f}</td></tr></tbody></table>
  <figure><figcaption>When we said a team had a 70% chance, did it win about 70% of the time? Each dot is a group of games;
     dots on the dashed line mean the percentages were honest.</figcaption>
  {calibration_chart(bt["calibration"])}{legend_two()}</figure>
  <p class="fine"><a href="{REPO_URL}/blob/main/data/backtest.json">The numbers</a> and
     <a href="{REPO_URL}/blob/main/docs/reports/phase2_model_backtest.md">the full report</a>, including everything tried and rejected.</p>
  </details>
</section>'''


def about_section() -> str:
    return f'''<section id="about">
  <h3 class="sub">Why you can trust the record</h3>
  <p>Every pick is saved to a public repository <em>before</em> kickoff, with the time it was made and the exact version
     of the model that made it. The Vegas line is saved at the same moment, beside it. After the games, the results are
     written by a separate step that can read the picks but cannot change them. A wrong pick stays wrong on the record.</p>
  <p>The model uses only things known before a game starts: team strength ratings, recent form from play-by-play data,
     the starting quarterbacks, rest days, and the schedule. It re-learns from every finished game since 2002 before each
     set of picks.</p>
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
.card {{ border-top: 4px solid var(--team); }}
.calls {{ margin: .8rem 0 .7rem; display: grid; gap: .35rem; }}
.call {{ display: grid; grid-template-columns: 7.2rem 1rem auto auto 1fr; align-items: center; gap: .5rem; }}
.call .who {{ color: var(--muted); font-size: .9rem; }}
.call .chip {{ width: .8rem; height: .8rem; border-radius: 50%; display: inline-block; }}
.call strong {{ font-family: "Bricolage Grotesque", system-ui, sans-serif; font-size: 1.35rem; }}
.call .conf {{ font-family: "Bricolage Grotesque", system-ui, sans-serif; font-weight: 600; font-size: 1.15rem; }}
.call.vegas strong {{ font-size: 1.1rem; }} .call.vegas .conf {{ font-size: 1rem; font-weight: 500; color: var(--muted); }}
.call .towin, .call .none {{ color: var(--muted); font-size: .9rem; }}
.plain {{ margin: .5rem 0 .2rem; font-size: 1rem; }}
.scale {{ display: block; width: 100%; height: auto; margin-top: .2rem; }}
.scale-axis {{ stroke: var(--rule); stroke-width: 2; }} .scale-zero {{ stroke: var(--muted); stroke-width: 1; }}
.scale-end, .scale-tick, .scale-label {{ font-size: 11px; fill: var(--muted); font-family: "Source Sans 3", system-ui, sans-serif; }}
.scale-label {{ fill: var(--ink); font-weight: 600; }} .scale-label.vegas {{ fill: var(--vegas); }}
.scale-ours {{ fill: var(--team); stroke: var(--panel); stroke-width: 2; }} .scale-vegas {{ fill: var(--vegas); stroke: var(--panel); stroke-width: 2; }}
.scale-actual {{ fill: var(--ink); opacity: .55; }}
.sub {{ font-size: 1.3rem; margin: 2rem 0 .5rem; }} .empty {{ padding: 1rem 1.1rem; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; }}
.more {{ margin: 1rem 0; }} .more summary {{ cursor: pointer; color: var(--model); }}
.bar {{ display: block; width: 100%; height: auto; overflow: visible; }}
.bar-vegas {{ fill: var(--vegas); }}
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
