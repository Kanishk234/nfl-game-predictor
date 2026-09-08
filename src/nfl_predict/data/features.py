"""Feature engineering, with an explicit as-of time on every derived value.

The discipline that makes the leakage gate meaningful: no feature for a game may be built from
information that was not knowable before that game's kickoff. Two distinct hazards:

1. A team's own prior games. Safe by construction if we only ever look backwards in kickoff
   order, since a team never plays twice within one game's span.
2. *Other* teams' simultaneous games. This is the subtle one. Elo ratings are league-wide, so a
   naive implementation that walks games in order and updates ratings immediately would let a
   Sunday 1:00pm game learn from another Sunday 1:00pm game that had not finished yet. Elo here
   uses a pending-update queue keyed on each game's *completion* time, so a rating only absorbs
   results that had actually concluded before the game being predicted kicked off.

Every feature frame carries an as-of column: the latest moment any input to that row was
knowable. The gate asserts that column is strictly earlier than kickoff for every row.
"""

from __future__ import annotations

import nflreadpy as nfl
import polars as pl

from nfl_predict.data.schedule import GAME_DURATION

#: Elo tuning. Deliberately conventional -- Phase 2 owns model tuning; this is just a feature.
ELO_START = 1500.0
ELO_K = 20.0
ELO_HOME_ADVANTAGE = 65.0
#: Fraction of a team's deviation from the mean carried into the next season.
ELO_SEASON_CARRYOVER = 2 / 3

#: Games in the rolling form window.
FORM_WINDOW = 8

#: nflreadpy's schedules keep the abbreviation a franchise used *at the time*, while its team
#: stats use the current one. Joining the two without this mapping silently yields null EPA for
#: every San Diego, St. Louis and Oakland game -- a quiet quality loss rather than a loud error,
#: which is exactly the kind of bug that survives to production. Verified exhaustive: these are
#: the only three codes present in schedules and absent from team stats over 2002-2025.
TEAM_CODE_ALIASES = {"SD": "LAC", "STL": "LA", "OAK": "LV"}


def canonical_team(column: str = "team") -> pl.Expr:
    """Map a historical franchise abbreviation onto the one team stats uses today."""
    return pl.col(column).replace(TEAM_CODE_ALIASES)


def elo_ratings(games: pl.DataFrame) -> pl.DataFrame:
    """Pre-game Elo for both teams, plus the as-of time of the latest result baked in.

    Walks games in kickoff order, but defers each result's rating update until that game has
    actually finished, so simultaneous kickoffs cannot see each other.
    """
    played = games.filter(pl.col("is_played")).sort("kickoff_utc", "game_id")

    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    # (completion_time, home, away, home_delta) awaiting application, kept in time order.
    pending: list[tuple] = []
    # Latest completion time already folded into `ratings`: the as-of time of every rating.
    absorbed_through = None
    rows = []

    def rating_for(team: str, season: int) -> float:
        r = ratings.get(team, ELO_START)
        if last_season.get(team) not in (None, season):
            # Regress toward the mean between seasons: rosters and coaching change.
            r = ELO_START + (r - ELO_START) * ELO_SEASON_CARRYOVER
        return r

    for g in played.iter_rows(named=True):
        kickoff = g["kickoff_utc"]

        # Apply every update that had concluded *strictly before* this kickoff. Strictness
        # matters: under the 4h GAME_DURATION assumption a 4:15pm ET game completes at exactly
        # 8:15pm ET, the moment the Sunday night game kicks off. A result known only at the
        # instant of kickoff is not knowable before it.
        while pending and pending[0][0] < kickoff:
            done, home_t, away_t, delta = pending.pop(0)
            ratings[home_t] = ratings.get(home_t, ELO_START) + delta
            ratings[away_t] = ratings.get(away_t, ELO_START) - delta
            absorbed_through = done if absorbed_through is None else max(absorbed_through, done)

        season = g["season"]
        home, away = g["home_team"], g["away_team"]
        home_elo, away_elo = rating_for(home, season), rating_for(away, season)
        ratings[home], ratings[away] = home_elo, away_elo
        last_season[home] = last_season[away] = season

        rows.append(
            {
                "game_id": g["game_id"],
                "home_elo_pre": home_elo,
                "away_elo_pre": away_elo,
                "elo_diff": home_elo + ELO_HOME_ADVANTAGE - away_elo,
                # Ratings encode results up to `absorbed_through`. Null means the ratings drew
                # on nothing at all yet (everyone still at ELO_START), which is trivially safe.
                "elo_as_of_utc": absorbed_through,
            }
        )

        margin = g["home_score"] - g["away_score"]
        expected_home = 1.0 / (1.0 + 10 ** (-(home_elo + ELO_HOME_ADVANTAGE - away_elo) / 400.0))
        actual_home = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        # Margin-of-victory multiplier, damped for favourites (FiveThirtyEight's form).
        mov = ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * abs(home_elo - away_elo))
        delta = ELO_K * mov * (actual_home - expected_home)

        pending.append((kickoff + GAME_DURATION, home, away, delta))
        pending.sort(key=lambda p: p[0])

    return pl.DataFrame(rows).with_columns(
        pl.col("elo_as_of_utc").cast(pl.Datetime("us", "UTC"))
    )


def _team_game_epa(seasons: list[int]) -> pl.DataFrame:
    """Per team-game offensive EPA. Seasons with no published stats file yet are skipped."""
    frames = []
    for season in seasons:
        try:
            frames.append(
                nfl.load_team_stats(seasons=[season], summary_level="week").select(
                    "game_id", "team", "passing_epa", "rushing_epa", "attempts", "carries"
                )
            )
        except (ConnectionError, OSError):
            # A not-yet-started season has no stats release. Not an error: those games have no
            # results to summarise yet.
            continue
    if not frames:
        return pl.DataFrame(
            schema={"game_id": pl.String, "team": pl.String, "off_epa_per_play": pl.Float64}
        )
    plays = pl.col("attempts").fill_null(0) + pl.col("carries").fill_null(0)
    return (
        pl.concat(frames)
        .with_columns(
            (
                (pl.col("passing_epa").fill_null(0) + pl.col("rushing_epa").fill_null(0))
                / pl.max_horizontal(plays, pl.lit(1))
            ).alias("off_epa_per_play")
        )
        .select("game_id", "team", "off_epa_per_play")
    )


def rolling_form(games: pl.DataFrame, window: int = FORM_WINDOW) -> pl.DataFrame:
    """Rolling offensive/defensive EPA and scoring margin over each team's last `window` games.

    Strictly backward-looking: every value is shifted by one game, so a team's row for a game
    never includes that game. The as-of time is the completion of the team's previous game.
    """
    played = games.filter(pl.col("is_played"))
    epa = _team_game_epa(sorted(played["season"].unique().to_list()))

    # One row per team per game.
    long = pl.concat(
        [
            played.select(
                "game_id",
                "kickoff_utc",
                "season",
                pl.col("home_team").alias("team"),
                pl.col("away_team").alias("opponent"),
                pl.col("home_score").alias("points_for"),
                pl.col("away_score").alias("points_against"),
            ),
            played.select(
                "game_id",
                "kickoff_utc",
                "season",
                pl.col("away_team").alias("team"),
                pl.col("home_team").alias("opponent"),
                pl.col("away_score").alias("points_for"),
                pl.col("home_score").alias("points_against"),
            ),
        ]
    ).with_columns(canonical_team().alias("team_canonical"))

    long = long.join(
        epa.rename({"team": "team_canonical"}), on=["game_id", "team_canonical"], how="left"
    )

    # Defensive EPA allowed is the opponent's offensive EPA in that same game.
    long = long.join(
        long.select(
            "game_id",
            pl.col("team").alias("opponent"),
            pl.col("off_epa_per_play").alias("def_epa_per_play"),
        ),
        on=["game_id", "opponent"],
        how="left",
    ).sort("team", "kickoff_utc")

    prior_completion = pl.col("kickoff_utc").shift(1).over("team") + GAME_DURATION
    return long.with_columns(
        pl.col("off_epa_per_play")
        .shift(1)
        .rolling_mean(window, min_samples=1)
        .over("team")
        .alias("off_epa_form"),
        pl.col("def_epa_per_play")
        .shift(1)
        .rolling_mean(window, min_samples=1)
        .over("team")
        .alias("def_epa_form"),
        (pl.col("points_for") - pl.col("points_against"))
        .shift(1)
        .rolling_mean(window, min_samples=1)
        .over("team")
        .alias("margin_form"),
        prior_completion.alias("form_as_of_utc"),
    ).select("game_id", "team", "off_epa_form", "def_epa_form", "margin_form", "form_as_of_utc")
