"""Assemble the modelling frame: one row per game, features + targets + as-of times.

Run as `python -m nfl_predict.data.pipeline` to refresh `data/processed/games.parquet`.

The frame carries three kinds of column, kept strictly separate:

- **features** - everything the model may look at. Each is knowable before kickoff.
- **targets** - `home_win` and `margin`, known only after the game.
- **baseline** - `spread_line` / `total_line`, the closing market. NOT a feature. It exists so
  Phase 2 can score the model against Vegas on the same games, and using it as an input would
  make the comparison meaningless.

`as_of_utc` is the latest moment any feature input for that row was knowable. The leakage gate
asserts it is strictly earlier than `kickoff_utc`.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nfl_predict.data.features import elo_ratings, qb_features, rolling_form
from nfl_predict.data.games import load_games

PROCESSED_PATH = Path("data/processed/games.parquet")

#: 2020 was played without crowds (and with COVID disruption). Rather than drop the season, it
#: is flagged: see docs/reports/phase1_data_pipeline.md for the measured justification.
COVID_SEASON = 2020

FEATURE_COLUMNS = [
    "elo_diff",
    "home_elo_pre",
    "away_elo_pre",
    "off_epa_form_diff",
    "def_epa_form_diff",
    "margin_form_diff",
    "home_off_epa_form",
    "home_def_epa_form",
    "away_off_epa_form",
    "away_def_epa_form",
    "home_margin_form",
    "away_margin_form",
    "qb_rating_diff",
    "qb_change_delta",
    "qb_exp_diff",
    "home_qb_rating",
    "away_qb_rating",
    "rest_diff",
    "home_rest",
    "away_rest",
    "div_game",
    "is_neutral_site",
    "no_crowd",
    "week",
]

TARGET_COLUMNS = ["home_win", "margin"]
BASELINE_COLUMNS = ["spread_line", "total_line"]


class LeakageError(AssertionError):
    """A feature row draws on information not knowable before its own kickoff."""


def build_frame() -> pl.DataFrame:
    games = load_games()
    elo = elo_ratings(games)
    form = rolling_form(games)
    qb = qb_features(games)

    home_form = form.rename(
        {
            "team": "home_team",
            "off_epa_form": "home_off_epa_form",
            "def_epa_form": "home_def_epa_form",
            "margin_form": "home_margin_form",
            "form_as_of_utc": "home_form_as_of_utc",
        }
    )
    away_form = form.rename(
        {
            "team": "away_team",
            "off_epa_form": "away_off_epa_form",
            "def_epa_form": "away_def_epa_form",
            "margin_form": "away_margin_form",
            "form_as_of_utc": "away_form_as_of_utc",
        }
    )

    frame = (
        games.join(elo, on="game_id", how="left")
        .join(home_form, on=["game_id", "home_team"], how="left")
        .join(away_form, on=["game_id", "away_team"], how="left")
        .join(qb, on="game_id", how="left")
        .with_columns(
            (pl.col("home_off_epa_form") - pl.col("away_off_epa_form")).alias("off_epa_form_diff"),
            (pl.col("home_def_epa_form") - pl.col("away_def_epa_form")).alias("def_epa_form_diff"),
            (pl.col("home_margin_form") - pl.col("away_margin_form")).alias("margin_form_diff"),
            (pl.col("home_rest") - pl.col("away_rest")).alias("rest_diff"),
            (pl.col("location") != "Home").cast(pl.Int8).alias("is_neutral_site"),
            (pl.col("season") == COVID_SEASON).cast(pl.Int8).alias("no_crowd"),
            (pl.col("result") > 0).cast(pl.Int8).alias("home_win"),
            pl.col("result").alias("margin"),
            # Rest days and the schedule itself are known when the schedule is published, so
            # they contribute no as-of constraint. Only result-derived features do.
            pl.max_horizontal(
                "elo_as_of_utc", "home_form_as_of_utc", "away_form_as_of_utc", "qb_as_of_utc"
            ).alias("as_of_utc"),
        )
    )
    return frame.select(
        "game_id", "season", "game_type", "kickoff_utc", "as_of_utc",
        "home_team", "away_team", "is_played",
        *FEATURE_COLUMNS, *TARGET_COLUMNS, *BASELINE_COLUMNS,
    )


def assert_no_leakage(frame: pl.DataFrame) -> None:
    """The gate. Every feature input must predate the kickoff it is used to predict."""
    late = frame.filter(pl.col("as_of_utc").is_not_null() & (pl.col("as_of_utc") >= pl.col("kickoff_utc")))
    if late.height:
        sample = late.select("game_id", "kickoff_utc", "as_of_utc").head(5).to_dicts()
        raise LeakageError(
            f"{late.height} game(s) use information not knowable before kickoff: {sample}"
        )


def main() -> int:
    frame = build_frame()
    assert_no_leakage(frame)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(PROCESSED_PATH)

    played = frame.filter(pl.col("is_played"))
    print(f"wrote {PROCESSED_PATH}  rows={frame.height}  completed={played.height}")
    print(f"seasons {frame['season'].min()}-{frame['season'].max()}  features={len(FEATURE_COLUMNS)}")
    print("leakage gate: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
