"""Clean the reviews, keep users and games with enough history, and split each user's history by time.

Users are split into two groups:
  core  = users with 5+ reviews -> last review = test, second last = val, rest = train
  light = 2-4 reviews -> last review = test, rest = train (no val; tuning uses core users only)
Both groups are used for training. Main scores are reported on core users; light users get their own score.

Input:  data/processed/recommendations.parquet
Output: data/processed/interactions.parquet  (one row per review, with `split` and `user_group` columns)
        data/processed/user_map.parquet       (user_id -> user_idx, user_group)
        data/processed/item_map.parquet       (app_id  -> item_idx)
        data/processed/prepare_stats.json     (row counts after each step)

Run from the project root:  .venv/bin/python src/prepare.py
"""

import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]  # project folder, wherever the script is run from
DATA = ROOT / "data" / "processed"
MIN_USER_REVIEWS = 2   # minimum to have 1 review to learn from + 1 to test on
MIN_GAME_REVIEWS = 5   # games with fewer reviews can't be learned from who reviewed them
CORE_USER_REVIEWS = 5  # users with this many reviews also get a validation review


def load_and_clean() -> pl.DataFrame:
    """Apply the cleaning rules decided in the EDA."""
    return (
        pl.scan_parquet(DATA / "recommendations.parquet")
        .select("user_id", "app_id", "date", "is_recommended", "hours", "helpful", "review_id")
        # Rule 1: a thumbs up with 0 hours played is not a considered a valid recommendation, so drop those rows.
        .filter(~((pl.col("hours") == 0) & pl.col("is_recommended")))
        # Rule 2: duplicate (user, game) pairs -> keep the latest review.
        # review_id is not chronological, so it is only a reproducible tie-break.
        .sort("date", "review_id")
        .unique(subset=["user_id", "app_id"], keep="last")
        .with_columns(pl.col("helpful").cast(pl.Int32))
        .collect(engine="streaming")
    )


def filter_min_reviews(df: pl.DataFrame, min_user: int, min_game: int) -> pl.DataFrame:
    """Repeatedly drop users with < min_user reviews and games with < min_game reviews.

    Dropping a game can push a user below the minimum (and the reverse), so repeat until nothing changes.
    """
    rnd = 0
    while True:
        rnd += 1
        before = df.height
        df = df.filter(pl.len().over("user_id") >= min_user)
        df = df.filter(pl.len().over("app_id") >= min_game)
        print(f"  round {rnd}: {df.height:,} rows")
        if df.height == before:
            return df


def add_index_columns(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Map Steam ids to 0..n-1 indices (needed by ALS and SASRec) and label each user core or light."""
    user_map = (
        df.group_by("user_id")
        .len()
        .sort("user_id")
        .with_row_index("user_idx")
        .with_columns(
            pl.when(pl.col("len") >= CORE_USER_REVIEWS)
            .then(pl.lit("core"))
            .otherwise(pl.lit("light"))
            .alias("user_group")
        )
        .drop("len")
    )
    item_map = df.select("app_id").unique().sort("app_id").with_row_index("item_idx")
    df = df.join(user_map, on="user_id").join(item_map, on="app_id")
    return df, user_map, item_map


def leave_last_out(df: pl.DataFrame) -> pl.DataFrame:
    """Per user: latest review = test; second latest = val for core users only; everything else = train."""
    return (
        df.sort("user_idx", "date", "review_id")
        .with_columns(
            (pl.len().over("user_idx") - pl.int_range(pl.len()).over("user_idx")).alias("pos_from_end")
        )
        .with_columns(
            pl.when(pl.col("pos_from_end") == 1).then(pl.lit("test"))
            .when((pl.col("pos_from_end") == 2) & (pl.col("user_group") == "core")).then(pl.lit("val"))
            .otherwise(pl.lit("train"))
            .alias("split")
        )
        .drop("pos_from_end")
    )


def main() -> None:
    stats = {"raw_rows": pl.scan_parquet(DATA / "recommendations.parquet").select(pl.len()).collect().item()}

    print("Cleaning...")
    df = load_and_clean()
    stats["after_cleaning"] = df.height

    print(f"Keeping users with >= {MIN_USER_REVIEWS} and games with >= {MIN_GAME_REVIEWS} reviews...")
    df = filter_min_reviews(df, MIN_USER_REVIEWS, MIN_GAME_REVIEWS)
    stats["after_filter"] = df.height

    df, user_map, item_map = add_index_columns(df)
    df = leave_last_out(df)

    # Games that only appear in val/test can never be learned from training data
    train_items = df.filter(pl.col("split") == "train").select("item_idx").unique()
    stats.update(
        users=user_map.height,
        core_users=int((user_map["user_group"] == "core").sum()),
        light_users=int((user_map["user_group"] == "light").sum()),
        items=item_map.height,
        train_rows=int((df["split"] == "train").sum()),
        val_rows=int((df["split"] == "val").sum()),
        test_rows=int((df["split"] == "test").sum()),
        train_rows_from_light_users=int(df.filter((pl.col("split") == "train") & (pl.col("user_group") == "light")).height),
        items_missing_from_train=item_map.height - train_items.height,
        share_recommended=float(df["is_recommended"].mean()),
        test_share_recommended_core=float(
            df.filter((pl.col("split") == "test") & (pl.col("user_group") == "core"))["is_recommended"].mean()
        ),
        test_share_recommended_light=float(
            df.filter((pl.col("split") == "test") & (pl.col("user_group") == "light"))["is_recommended"].mean()
        ),
        date_min=str(df["date"].min()),
        date_max=str(df["date"].max()),
    )

    df.select(
        "user_idx", "item_idx", "user_id", "app_id", "date",
        "is_recommended", "hours", "helpful", "review_id", "user_group", "split",
    ).write_parquet(DATA / "interactions.parquet")
    user_map.write_parquet(DATA / "user_map.parquet")
    item_map.write_parquet(DATA / "item_map.parquet")
    (DATA / "prepare_stats.json").write_text(json.dumps(stats, indent=2))

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
