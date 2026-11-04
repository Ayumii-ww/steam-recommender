"""Build the small files the demo app needs, so the app never loads the 33M-row dataset.

Writes to models/demo/ (gitignored):
  games.parquet       one row per game the model knows: item_idx, title, year, price, tags, review count
  daily_counts.npz    reviews per game per day (sparse), for "trending on a chosen date"

Run from the project root after src/prepare.py:  .venv/bin/python app/build_assets.py
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "models" / "demo"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- games ----------
reviews = pl.scan_parquet(DATA / "interactions.parquet").group_by("item_idx").agg(pl.len().alias("n_reviews"))
games = (
    pl.scan_parquet(DATA / "item_map.parquet")
    .join(reviews, on="item_idx", how="left")
    .join(pl.scan_parquet(DATA / "games.parquet").with_columns(pl.col("app_id").cast(pl.Int32)), on="app_id", how="left")
    .join(pl.scan_parquet(DATA / "games_metadata.parquet").with_columns(pl.col("app_id").cast(pl.Int32)), on="app_id", how="left")
    .select(
        "item_idx", "app_id", "title", "n_reviews",
        pl.col("date_release").dt.year().alias("year"),
        pl.col("price_final").alias("price"),
        "positive_ratio",
        pl.col("tags").list.head(4).list.join(", ").alias("tags"),
    )
    .sort("item_idx")
    .collect()
)
games.write_parquet(OUT / "games.parquet")
print(f"games: {games.height:,} (missing titles: {games['title'].null_count()})")

# ---------- reviews per game per day ----------
days = (
    pl.scan_parquet(DATA / "interactions.parquet")
    .select(pl.col("date").cast(pl.Int32).alias("day"), "item_idx")
    .group_by("day", "item_idx").len()
    .collect(engine="streaming")
)
first_day = int(days["day"].min())
n_days = int(days["day"].max()) - first_day + 1
counts = sp.csr_matrix(
    (days["len"].to_numpy().astype(np.int32), (days["day"].to_numpy() - first_day, days["item_idx"].to_numpy())),
    shape=(n_days, games.height),
)
sp.save_npz(OUT / "daily_counts.npz", counts)
(OUT / "meta.json").write_text(json.dumps({"first_day": first_day, "n_days": n_days}))
print(f"daily counts: {n_days:,} days, {counts.nnz:,} non-zero cells")

for f in sorted(OUT.iterdir()):
    print(f"{f.name:20s} {f.stat().st_size / 1e6:6.1f} MB")
