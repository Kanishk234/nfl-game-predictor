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
MODEL_LIGHT, VEGAS_LIGHT = "#1F6FB5", "#C43F8B"
#: Vegas is drawn in pink on purpose: no NFL team uses pink in either of its colours, so the
#: market's marker can never be mistaken for a team's. Chosen by measuring CIELAB distance from
#: every team colour (and every lightened variant the cards can show) - pink sits ~34 away from
#: its nearest neighbour, where the gold it replaced sat ~15 from Vikings/Steelers/Packers gold.
MODEL_DARK, VEGAS_DARK = "#3F7FC6", "#F06BB0"

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


LIGHT_PANEL, DARK_PANEL = "#FFFFFF", "#111111"


def _luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _mix(hex_color: str, toward: str, amount: float) -> str:
    c = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    d = [int(toward[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * amount):02X}" for x, y in zip(c, d))


def _is_achromatic(hex_color: str) -> bool:
    c = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return max(c) - min(c) < 20


def _scale(hex_color: str, k: float) -> str:
    """Multiply each channel: keeps hue and saturation, unlike mixing toward white/black."""
    c = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{min(255, max(0, round(x * k))):02X}" for x in c)


def readable_team_color(abbr: str, surface: str) -> str:
    """The team colour that shows on this surface, keeping the team's hue.

    Prefer the primary. If it does not reach 3:1 (the WCAG floor for graphics), brighten or
    darken it by scaling its channels, which keeps the hue and saturation: Rams navy becomes a
    vivid lighter blue on a dark panel, not yellow and not grey. Only when the primary is black
    or grey, with no hue to keep, does the secondary take over (Steelers gold), and if that is
    grey too (Raiders silver) it is mixed toward the opposite of the surface.
    """
    primary, secondary = TEAM_COLORS.get(abbr, ("#62707E", "#62707E"))[:2]
    # A black primary has no hue to keep: use the secondary (Steelers gold; Raiders silver).
    base = secondary if _is_achromatic(primary) else primary
    lighten = _luminance(surface) < 0.5
    color, k = base, 1.0
    for _ in range(40):
        if _contrast(color, surface) >= 3.0:
            return color
        k *= 1.08 if lighten else 0.92
        nxt = _scale(base, k)
        if nxt == color:  # channels saturated or zero: scaling cannot move it further
            break
        color = nxt
    opposite = "#FFFFFF" if lighten else "#000000"
    amount = 0.0
    while _contrast(color, surface) < 3.0 and amount < 1.0:
        amount += 0.05
        color = _mix(base, opposite, amount)
    return color


#: ESPN's team logo CDN, the source nflverse uses. Two codes differ from ours.
_LOGO_CODE = {"LA": "lar", "WAS": "wsh"}


def logo_url(abbr: str) -> str:
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{_LOGO_CODE.get(abbr, abbr.lower())}.png"


def logo(abbr: str, cls: str = "logo") -> str:
    """A team logo on a light disc, so black and navy marks read on the dark panel.

    Loaded eagerly on purpose. ESPN serves one 500px PNG per team and ignores resize params, so
    a lazy logo pops in as a bare white circle a moment after the card paints. There are only 32
    distinct images in a week and the browser caches them across pages, so eager is the better
    trade: a slightly heavier first load, and nothing that flashes.
    """
    return (f'<img class="{cls}" src="{logo_url(abbr)}" alt="" width="40" height="40" '
            f'decoding="async">')


def team_color(abbr: str) -> str:
    """Light-surface colour; the dark-surface one is set beside it as a CSS variable."""
    return readable_team_color(abbr, LIGHT_PANEL)


def team_vars(abbr: str) -> str:
    """Inline style with the team colour that reads on the site's (dark) panel."""
    return f"--team:{readable_team_color(abbr, DARK_PANEL)}"


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


def slot_label(iso: str) -> str:
    """'Sunday 1:00 pm', 'Thursday night'... the way people think about an NFL week."""
    d = datetime.fromisoformat(iso).astimezone(ET)
    day = d.strftime("%A")
    if d.hour >= 19:
        return f"{day} night"
    hour = d.hour % 12 or 12
    return f"{day} {hour}:{d:%M} {'am' if d.hour < 12 else 'pm'}"


def group_by_slot(rows: list[tuple[str, dict]]) -> list[tuple[str, list[tuple[str, dict]]]]:
    groups: list[tuple[str, list]] = []
    for pn, p in rows:
        label = slot_label(p["kickoff_utc"])
        if not groups or groups[-1][0] != label:
            groups.append((label, []))
        groups[-1][1].append((pn, p))
    return groups


def disagreements(rows: list[tuple[str, dict]]) -> list[dict]:
    out = []
    for _, p in rows:
        v = p.get("vegas")
        if v and v.get("p_home_moneyline") is not None:
            v_fav = p["home_team"] if v["p_home_moneyline"] >= 0.5 else p["away_team"]
            if v_fav != p["pick"]:
                out.append(p)
    return out


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
    away_fill = "var(--away)" if pick_home else "var(--team)"
    home_fill = "var(--team)" if pick_home else "var(--away)"
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
    def Y(v):
        # Clamp: a value outside the axis range must not be drawn outside the plot.
        v = max(y_min, min(y_max, v))
        return mt + (1 - (v - y_min) / (y_max - y_min)) * (h - mt - mb)
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

def spread_rows(ours: float, line: float | None, actual: float | None, home: str, away: str) -> str:
    """Two (three, once played) aligned rows in one grid: a caption spanning them on the left,
    then who / how much / a bar growing from a shared centre. Text is never positioned by value,
    so labels cannot collide, and the axis needs no header row of its own."""
    half = 10.0
    rows = [("Us", ours, "ours")]
    if line is not None:
        rows.append(("Vegas", line, "vegas"))
    if actual is not None:
        rows.append(("Final", actual, "final"))
    out = ['<div class="spread"><span class="spread-label">Points</span><div class="spread-rows">']
    for who, v, cls in rows:
        v_c = max(-half, min(half, v))
        width = abs(v_c) / half * 50
        left = 50 if v_c >= 0 else 50 - width
        fav = home if v >= 0 else away
        style = f"--fav:{team_vars_pair(fav)}" if cls == "ours" else ""
        out.append(f'<div class="spread-row {cls}" style="{e(style)}">'
                   f'<span class="spread-who">{who}</span>'
                   f'<span class="spread-val">{e(by_team(v, home, away))}</span>'
                   f'<span class="spread-track"><span class="spread-bar"></span>'
                   f'<span class="spread-fill" style="left:{left:.1f}%;width:{width:.1f}%"></span></span></div>')
    out.append("</div>")
    out.append("</div>")
    return "".join(out)


def team_vars_pair(abbr: str) -> str:
    return readable_team_color(abbr, DARK_PANEL)


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
        agree = v_fav == pick
        vegas_row = (f'<p class="vegas-call"><span class="who">Vegas</span> {logo(v_fav, "logo small")}'
                     f'<strong>{e(v_fav)}</strong> {pct(v_conf)}'
                     + ("" if agree else ' <span class="disagree">we disagree</span>') + '</p>')
    else:
        vegas_row = '<p class="vegas-call"><span class="who">Vegas</span> no line yet</p>'

    if g:
        ok = g["model"]["correct"]
        # The result has to read at a glance, not be parsed: one glyph and a tint carry the
        # verdict, the score is picked out by winner, and the two things we were judged on -
        # the pick and the spread - are separate chips rather than clauses in a sentence.
        # A tie is neither right nor wrong: the pick simply does not count towards accuracy.
        if g["winner"] == "tie":
            tone, mark, pick_chip = "tie", "=", ("tie", "Pick —", "a tie: the pick does not count")
        elif ok:
            tone, mark, pick_chip = "hit", "✓", ("hit", "Pick ✓", "we picked the winner")
        else:
            tone, mark, pick_chip = "miss", "✗", ("miss", "Pick ✗", "we picked the loser")
        ats_chip = {"win": ("hit", "Spread ✓", "our side covered the spread"),
                    "loss": ("miss", "Spread ✗", "our side did not cover the spread"),
                    "push": ("tie", "Spread —", "the spread was a push")}.get(g["model"].get("ats"))
        chips = "".join(f'<span class="chip {c}" title="{e(title)}">{e(text)}</span>'
                        for c, text, title in (pick_chip, ats_chip) if c)
        won_home = g["winner"] == home
        won_away = g["winner"] == away
        outcome = (f'<div class="result {tone}"><span class="mark" aria-hidden="true">{mark}</span>'
                   f'<div class="result-body">'
                   f'<p class="score"><span class="side{" won" if won_away else ""}">{e(away)} {g["away_score"]}</span>'
                   f'<span class="side{" won" if won_home else ""}">{e(home)} {g["home_score"]}</span></p>'
                   f'<p class="chips">{chips}</p></div></div>')
        actual, status = g["margin"], "played"
    else:
        outcome, actual, status = "", None, "upcoming"

    return f'''<article class="card {status}" id="{e(p["game_id"])}" style="{team_vars(pick)}">
  <header>
    <p class="matchup">{logo(away)}<span class="vs">{e(team_nick(away))} <small>at</small> {e(team_nick(home))}</span>{logo(home)}</p>
    <time datetime="{e(p["kickoff_utc"])}">{e(fmt_et(p["kickoff_utc"]))}</time>
  </header>
  <div class="zone">
    <p class="our-call">{logo(pick, "logo big")}<strong>{e(pick)}</strong><span class="conf">{pct(p_pick(p))}</span><span class="towin">to win</span></p>
    <div class="bar-row"><span class="bar-end">{e(away)} {pct(1 - p_home)}</span>{prob_bar(p_home, v["p_home_moneyline"] if v else None, home, away)}<span class="bar-end">{e(home)} {pct(p_home)}</span></div>
    {vegas_row}
  </div>
  <div class="zone">{spread_rows(p["pred_margin"], v["spread_line"] if v else None, actual, home, away)}</div>
  {outcome}
</article>'''


def week_table(rows: list[tuple[str, dict]], graded: dict[str, dict]) -> str:
    out = ['<table class="ledger"><thead><tr><th>Kickoff</th><th>Game</th><th>Pick</th>',
           '<th class="num">Our odds</th><th class="num">Vegas on our pick</th>',
           '<th>Our spread</th><th>Line</th><th>Pass</th><th>Result</th></tr></thead><tbody>']
    for pass_name, p in rows:
        home, away = p["home_team"], p["away_team"]
        g = graded.get(p["game_id"]); v = p.get("vegas")
        if g:
            ok = g["model"]["correct"]
            mark = "" if ok is None else (' <span class="hit">✓</span>' if ok else ' <span class="miss">✗</span>')
            res = f'{e(g["winner"])}{mark} <small>{g["away_score"]}–{g["home_score"]}</small>'
        else:
            res = '<span class="pending">—</span>'
        # Both probability columns describe the team *we* picked, so they compare directly.
        v_same_side = None
        if v and v.get("p_home_moneyline") is not None:
            v_same_side = v["p_home_moneyline"] if p["pick"] == home else 1 - v["p_home_moneyline"]
        out.append(f'<tr><td>{e(fmt_et(p["kickoff_utc"]))}</td><td>{e(away)} at {e(home)}</td><td><strong>{e(p["pick"])}</strong></td>'
                   f'<td class="num">{pct(p_pick(p))}</td><td class="num">{pct(v_same_side)}</td>'
                   f'<td>{e(by_team(p["pred_margin"], home, away))}</td>'
                   f'<td>{e(by_team(v["spread_line"], home, away)) if v else "—"}</td>'
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
    def slot_html(label: str, items: list) -> str:
        """A full-width divider then the slot's cards, all inside one continuous grid.

        The NFL week is lopsided by nature — one Thursday game, nine on Sunday afternoon, one at
        night. Separate grids per slot made the single-game slots look broken, so every card
        lives in the same grid and the labels span all of its columns.
        """
        cards = "".join(game_card(pn, p, graded.get(p["game_id"])) for pn, p in items)
        n = len(items)
        played = sum(1 for _, p in items if p["game_id"] in graded)
        count = f'{n} game{"s" if n != 1 else ""}' + (f', {played} played' if played else "")
        return f'<h3 class="slot"><span>{e(label)}</span><small>{e(count)}</small></h3>{cards}'

    groups = ('<div class="week-grid">'
              + "".join(slot_html(label, items) for label, items in group_by_slot(rows))
              + "</div>")
    return f'''<h2>Week {week}, {season}{e(status)}</h2>
<details class="howto"><summary>How to read a card</summary>
<p><strong>Our pick</strong> is the model's call, in that team's colour. The bar is the win probability; the small pink
   triangle under it is where Vegas puts it. <strong>Vegas</strong> is the betting favourite, for comparison. The spread
   rows show how much each of us expects the winner to win by: ours in the team colour, Vegas in pink. Once a game is
   played, a green or red band appears at the bottom with the score and two chips: whether we
   picked the winner, and whether our side covered the spread.</p></details>
{summary_strip(result["summary"] if result else None, "Official predictions only: the latest pass published before each game's kickoff.")}
{groups}
<h3 class="table-title">All games this week</h3>
{week_table(rows, graded)}
{provenance(passes)}'''


def season_body(history: dict | None, season: int | None, backtest: dict | None) -> str:
    parts = ['<h2>Track record</h2>',
             ('<p class="lede-2">Every pick is published before kickoff and scored afterwards against the '
              'Vegas line. This page updates as each week is graded.</p>')]
    s = (history or {}).get("seasons", {}).get(str(season)) if season else None
    if s and s["summary"].get("n"):
        graded = [w for w in s["weeks"] if w["summary"].get("n")]
        # Only finished weeks go on the chart. A week with two of sixteen games played has a
        # "weekly accuracy" of 0% or 100% and would swamp the line with noise.
        weeks = [w for w in graded if w.get("complete")]
        series = {"Model": [(w["week"], w["summary"]["model"]["accuracy"]) for w in weeks],
                  "Vegas": [(w["week"], w["summary"]["vegas"]["accuracy"]) for w in weeks if "vegas" in w["summary"]]}
        chart = line_chart(series, "picks right", 0.3, 1.0, ref=0.5) if len(weeks) >= 2 else ""
        partial = len(graded) - len(weeks)
        caption = ("Share of picks that were right, week by week. The dashed line is a coin flip."
                   + (" A week in progress joins the line once all its games are played." if partial else ""))
        parts.append(f'<h3 class="sub">{season} season so far</h3>' + summary_strip(s["summary"], "Every graded game, official predictions only.")
                     + (f'<figure><figcaption>{caption}</figcaption>{chart}{legend_two()}</figure>' if chart else ""))
    else:
        parts.append('<p class="empty">Nothing graded yet. The first results land the Tuesday after Week 1. '
                     'Until then, the dry run below is the best guide to what to expect.</p>')
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
  <div class="split">
    <div>
      <p>Before going live, we ran the model over the last five seasons as if they were happening week by week, never
         letting it see a game before it was played. A solid model that has not beaten the market — and that is the bar
         for this season.</p>
      <dl class="strip vertical">
        <div><dt>Our picks right</dt><dd>{pct(m["win"]["accuracy"], 1)}</dd></div>
        <div><dt>Vegas</dt><dd class="muted">{pct(v["win"]["accuracy"], 1)}</dd></div>
        <div><dt>Our spread miss</dt><dd>{m["spread"]["mae"]:.1f} pts</dd></div>
        <div><dt>Vegas</dt><dd class="muted">{v["spread"]["mae"]:.1f} pts</dd></div>
      </dl>
      <p class="fine">Across {bt["n_games"]:,} games and {bt["n_folds"]} retrains.
         <a href="{REPO_URL}/blob/main/data/backtest.json">The numbers</a>,
         <a href="{REPO_URL}/blob/main/docs/reports/phase2_model_backtest.md">the full report</a>.</p>
    </div>
    <div>
      <table class="compact"><thead><tr><th>Season</th><th class="num">Us</th><th class="num">Vegas</th><th class="num">Our miss</th><th class="num">Vegas</th></tr></thead>
      <tbody>{rows}<tr class="total"><td>All</td><td class="num">{pct(m["win"]["accuracy"], 1)}</td><td class="num">{pct(v["win"]["accuracy"], 1)}</td>
      <td class="num">{m["spread"]["mae"]:.1f}</td><td class="num">{v["spread"]["mae"]:.1f}</td></tr></tbody></table>
      <figure><figcaption>When we said a team had a 70% chance, did it win about 70% of the time? Dots on the dashed
         line mean the percentages were honest.</figcaption>
      {calibration_chart(bt["calibration"])}{legend_two()}</figure>
    </div>
  </div>
</section>'''


def about_section() -> str:
    return f'''<section id="about">
  <h3 class="sub">Why you can trust the record</h3>
  <div class="split">
    <p>Every pick is saved to a public repository <em>before</em> kickoff, with the time it was made and the exact
       version of the model that made it. The Vegas line is saved at the same moment, beside it. After the games, the
       results are written by a separate step that can read the picks but cannot change them. A wrong pick stays wrong
       on the record.</p>
    <p>The model uses only things known before a game starts: team strength ratings, recent form from play-by-play
       data, the starting quarterbacks, rest days, and the schedule. It re-learns from every finished game since 2002
       before each set of picks.</p>
  </div>
  <p><a href="{REPO_URL}">Repository</a>, <a href="{REPO_URL}/tree/main/data/predictions">predictions</a>,
     <a href="{REPO_URL}/tree/main/data/odds">odds snapshots</a>, <a href="{REPO_URL}/tree/main/data/results">results</a>,
     <a href="{REPO_URL}/actions">the scheduled jobs</a>.</p>
</section>'''


CSS = f"""
:root {{ --bg: #0A0A0A; --panel: #111111; --panel2: #171717; --ink: #EDEDED; --muted: #8A8F98; --rule: #262626; --rule2: #333333;
         --model: {MODEL_DARK}; --vegas: {VEGAS_DARK}; --away: #2A2F36; --hit: #3FB950; --miss: #F85149; }}
* {{ box-sizing: border-box; }}
html {{ color-scheme: dark; background: var(--bg); }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.5 "Source Sans 3", "Segoe UI", system-ui, sans-serif;
        font-variant-numeric: tabular-nums; -webkit-font-smoothing: antialiased; }}
main {{ max-width: 84rem; margin: 0 auto; padding: 1.5rem 1.5rem 5rem; }}
h1, h2, h3, dd, .num, .conf, .score {{ font-family: "Bricolage Grotesque", "Source Sans 3", system-ui, sans-serif; }}
h1 {{ font-size: 1.35rem; margin: 0; font-weight: 600; letter-spacing: -.01em; }}
h1 a {{ color: inherit; text-decoration: none; }}
h2 {{ font-size: 1.7rem; margin: 1.5rem 0 .25rem; font-weight: 600; letter-spacing: -.01em; }}
h3 {{ font-size: 1.15rem; margin: 0; font-weight: 600; }}
p {{ max-width: 74ch; }}
a {{ color: var(--model); text-underline-offset: .15em; }}
a:focus-visible, summary:focus-visible {{ outline: 2px solid var(--model); outline-offset: 3px; }}
.top {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: .5rem 1.5rem; padding-bottom: .9rem; border-bottom: 1px solid var(--rule); }}
.top .lede {{ color: var(--muted); margin: 0; font-size: .95rem; max-width: none; }}
.weeks {{ display: flex; flex-wrap: wrap; gap: .35rem; margin: .9rem 0 0; }}
.weeks a {{ text-decoration: none; padding: .3rem .75rem; border: 1px solid var(--rule2); border-radius: 999px; color: var(--ink); font-size: .9rem; background: var(--panel); }}
.weeks a[aria-current] {{ background: var(--ink); color: var(--bg); border-color: var(--ink); }}
.summary {{ font-size: 1.05rem; margin: .25rem 0 .5rem; }}
.how, .fine, small {{ color: var(--muted); font-size: .9rem; }}
.howto {{ color: var(--muted); font-size: .9rem; margin: 0 0 .5rem; }} .howto summary {{ cursor: pointer; color: var(--model); }}
.strip {{ display: flex; flex-wrap: wrap; gap: .5rem 2rem; margin: 1rem 0 .25rem; padding: .9rem 1.1rem; background: var(--panel);
          border: 1px solid var(--rule); border-radius: 10px; }}
.strip div {{ min-width: 6rem; }} .strip dt {{ font-size: .8rem; color: var(--muted); }}
.strip dd {{ margin: 0; font-size: 1.35rem; font-weight: 600; }} .strip dd.muted {{ color: var(--muted); }}
.strip.vertical {{ gap: .5rem 2.5rem; }}
.lede-2 {{ color: var(--muted); font-size: 1.05rem; margin: .25rem 0 1rem; }}
.split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem 2.5rem; align-items: start; }}
.split p {{ max-width: none; }}
.week-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(21rem, 100%), 1fr)); gap: 1.1rem; align-items: start; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(min(21rem, 100%), 1fr)); gap: 1.1rem; }}
.slot {{ grid-column: 1 / -1; display: flex; align-items: baseline; gap: .75rem; margin: 1.5rem 0 .1rem; color: var(--ink);
         font-weight: 600; font-size: 1rem; }}
.slot:first-child {{ margin-top: .5rem; }}
.slot small {{ color: var(--muted); font-weight: 400; font-size: .85rem; white-space: nowrap; }}
.slot::after {{ content: ""; flex: 1; height: 1px; background: var(--rule); }}
.card {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 14px; padding: 1.15rem 1.25rem 1.25rem; scroll-margin-top: 1rem; min-width: 0; overflow: hidden; }}
.zone {{ margin-top: 1.15rem; padding-top: 1.1rem; border-top: 1px solid var(--rule); }}
.card:target {{ border-color: var(--team); box-shadow: 0 0 0 1px var(--team); }}
.card header {{ display: block; }}
.matchup {{ display: flex; align-items: center; gap: .5rem; margin: 0; }}
.matchup .vs {{ flex: 1; text-align: center; font-family: "Bricolage Grotesque", system-ui, sans-serif; font-weight: 600; font-size: 1.1rem; line-height: 1.2; }}
.matchup small {{ color: var(--muted); font-weight: 400; }}
.card time {{ display: block; text-align: center; margin-top: .45rem; color: var(--muted); font-size: .85rem; }}
.logo {{ width: 40px; height: 40px; object-fit: contain; flex: none; background: #F2F3F5; border-radius: 50%; padding: 5px; }}
.logo.small {{ width: 24px; height: 24px; padding: 3px; vertical-align: -7px; margin: 0 .2rem 0 .35rem; }}
.logo.big {{ width: 48px; height: 48px; padding: 6px; margin: 0 .7rem 0 0; }}
.our-call {{ display: flex; align-items: center; gap: .1rem; margin: 0 0 .8rem; min-width: 0; }}
.our-call strong {{ font-family: "Bricolage Grotesque", system-ui, sans-serif; font-size: 2rem; color: var(--team); letter-spacing: -.01em; margin-right: .55rem; }}
.our-call .conf {{ font-weight: 600; font-size: 1.55rem; }}
.our-call .towin {{ color: var(--muted); font-size: .85rem; margin-left: .4rem; align-self: flex-end; padding-bottom: .3rem; }}
.bar-row {{ display: flex; align-items: center; gap: .6rem; min-width: 0; margin-bottom: .7rem; }}
.bar-end {{ color: var(--muted); font-size: .8rem; white-space: nowrap; min-width: 3.6rem; }} .bar-end:last-child {{ text-align: right; }}
.bar {{ display: block; flex: 1 1 0; min-width: 0; width: 100%; height: auto; overflow: visible; }}
.bar-vegas {{ fill: var(--vegas); }}
.vegas-call {{ margin: 0; font-size: .9rem; color: var(--muted); }}
.vegas-call .who {{ display: inline-block; width: 3.4rem; color: var(--vegas); font-weight: 600; }}
.vegas-call strong {{ color: var(--ink); font-size: 1rem; }}
.disagree {{ color: var(--vegas); font-weight: 600; margin-left: .35rem; }}
/* Each row is its own grid with the same template, so the bars line up by construction. A
   single grid with a row-spanning caption does not: `grid-row: 1 / -1` only spans the explicit
   grid, so the caption took a cell and pushed every later row one column out of line. */
.spread {{ display: flex; align-items: center; gap: .7rem; min-width: 0; font-size: .85rem; }}
.spread-label {{ flex: none; width: 2.9rem; color: var(--muted); font-size: .78rem; line-height: 1.15; }}
.spread-rows {{ flex: 1; min-width: 0; display: grid; gap: .5rem; }}
.spread-row {{ display: grid; grid-template-columns: 2.9rem minmax(5.2rem, auto) 1fr; align-items: center;
               gap: .5rem; min-width: 0; --favc: var(--fav); }}
.spread-who {{ color: var(--muted); }} .spread-row.vegas .spread-who {{ color: var(--vegas); font-weight: 600; }}
.spread-val {{ font-weight: 600; color: var(--ink); }}
.spread-track {{ position: relative; height: 10px; min-width: 0; }}
.spread-bar {{ position: absolute; inset: 0; background: var(--away); border-radius: 5px; }}
.spread-track::after {{ content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px; width: 2px; background: var(--rule2); z-index: 1; }}
.spread-fill {{ position: absolute; top: 0; height: 10px; border-radius: 5px; background: var(--favc); min-width: 3px; }}
.spread-row.vegas .spread-fill {{ background: var(--vegas); }}
.spread-row.final .spread-fill {{ background: var(--ink); }}
/* The played-game footer. The tint and the glyph say right or wrong before any word is read;
   the chips split the two judgements so neither has to be found inside a sentence. Colour is
   never the only signal - every state also carries its own glyph. */
.result {{ display: flex; align-items: center; gap: .85rem; margin: 1.15rem -1.25rem -1.25rem;
           padding: .8rem 1.25rem; border-top: 1px solid var(--rule); }}
.result.hit {{ background: rgba(63, 185, 80, .10); border-top-color: rgba(63, 185, 80, .35); }}
.result.miss {{ background: rgba(248, 81, 73, .10); border-top-color: rgba(248, 81, 73, .35); }}
.result.tie {{ background: rgba(138, 143, 152, .10); }}
.result .mark {{ flex: none; font-size: 1.6rem; line-height: 1; font-weight: 700; width: 1.3rem; text-align: center; }}
.result.hit .mark {{ color: var(--hit); }} .result.miss .mark {{ color: var(--miss); }}
.result.tie .mark {{ color: var(--muted); }}
.result-body {{ min-width: 0; flex: 1; }}
.result .score {{ display: flex; gap: 1.1rem; margin: 0; font-size: 1.05rem; font-weight: 600; color: var(--muted); }}
.result .side.won {{ color: var(--ink); }}
.chips {{ display: flex; flex-wrap: wrap; gap: .4rem; margin: .35rem 0 0; }}
.chip {{ font-size: .78rem; font-weight: 600; padding: .1rem .5rem; border-radius: 999px;
         border: 1px solid var(--rule2); color: var(--muted); white-space: nowrap; }}
.chip.hit {{ color: var(--hit); border-color: rgba(63, 185, 80, .45); }}
.chip.miss {{ color: var(--miss); border-color: rgba(248, 81, 73, .45); }}
.hit {{ color: var(--hit); font-weight: 700; }} .miss {{ color: var(--miss); font-weight: 700; }}
.pending {{ color: var(--muted); }}
.table-title {{ margin: 2.25rem 0 .5rem; }}
table {{ border-collapse: collapse; width: 100%; }}
.ledger {{ display: block; overflow-x: auto; font-size: .9rem; }}
.ledger th {{ text-align: left; font-weight: 600; font-size: .8rem; color: var(--muted); border-bottom: 1px solid var(--rule2); padding: .4rem .6rem; white-space: nowrap; }}
.ledger td {{ padding: .5rem .6rem; border-bottom: 1px solid var(--rule); white-space: nowrap; }}
.ledger .num, .compact .num {{ text-align: right; }}
.prov {{ margin: 1.5rem 0; color: var(--muted); font-size: .9rem; max-width: 72ch; }}
.prov summary {{ cursor: pointer; }} .prov ul {{ padding-left: 1.2rem; }} .prov li {{ margin: .4rem 0; }}
.legend {{ font-size: .85rem; color: var(--muted); margin: .5rem 0 0; }}
.swatch {{ display: inline-block; width: .8em; height: .8em; border-radius: 50%; margin: 0 .35em 0 1em; vertical-align: -.05em; }}
.swatch.model {{ background: var(--model); }} .swatch.vegas {{ background: var(--vegas); }} .legend .swatch:first-child {{ margin-left: 0; }}
.chart {{ width: 100%; max-width: 640px; height: auto; display: block; }}
.chart .grid {{ stroke: var(--rule2); stroke-width: 1; }} .chart .ref {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 4 4; }}
.chart .tick, .chart .label {{ font-size: 12px; fill: var(--muted); font-family: "Source Sans 3", system-ui, sans-serif; }}
.chart .line {{ fill: none; stroke-width: 2; }} .chart .line.model {{ stroke: var(--model); }} .chart .line.vegas {{ stroke: var(--vegas); }}
.chart .dot {{ stroke: var(--bg); stroke-width: 2; }} .chart .dot.model {{ fill: var(--model); }} .chart .dot.vegas {{ fill: var(--vegas); }}
.chart .dot[fill="none"] {{ fill: var(--bg) !important; stroke-width: 2; }} .chart .dot[fill="none"].model {{ stroke: var(--model); }} .chart .dot[fill="none"].vegas {{ stroke: var(--vegas); }}
.chart .label.model {{ fill: var(--model); }} .chart .label.vegas {{ fill: var(--vegas); }}
figure {{ margin: 1.5rem 0; }} figcaption {{ color: var(--muted); font-size: .9rem; margin-bottom: .5rem; max-width: 62ch; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; align-items: start; }}
.compact td, .compact th {{ padding: .35rem .5rem; border-bottom: 1px solid var(--rule); font-size: .9rem; }}
.compact th {{ text-align: left; color: var(--muted); font-weight: 600; font-size: .8rem; }} .compact .total td {{ font-weight: 600; border-top: 1px solid var(--rule2); }}
.sub {{ font-size: 1.25rem; margin: 2rem 0 .5rem; }} .empty {{ padding: 1rem 1.1rem; background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; }}
.more {{ margin: 1rem 0; }} .more summary {{ cursor: pointer; color: var(--model); }}
footer {{ margin-top: 3rem; }}
@media (max-width: 860px) {{ .split {{ grid-template-columns: 1fr; }} }}
@media (max-width: 720px) {{ .two-col {{ grid-template-columns: 1fr; }} .cards, .week-grid {{ grid-template-columns: 1fr; }} body {{ font-size: 15px; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
"""


def page(title: str, body: str, weeks: list[tuple[int, int]], current: tuple[int, int] | None, root: str, now: datetime) -> str:
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
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
