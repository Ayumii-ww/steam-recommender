"""Convert raw Kaggle CSV/JSON files to parquet to avoid repeated parsing and type inference."""

from pathlib import Path

import polars as pl

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# Show the columns first, so any mismatch is obvious
print(pl.scan_csv(RAW / "recommendations.csv").collect_schema())

# stream the CSV to parquet in chunks, so the 2 GB file never has to fit in memory at once
(
    pl.scan_csv(RAW / "recommendations.csv", try_parse_dates=True)
    .with_columns(
        pl.col("user_id").cast(pl.Int32),
        pl.col("app_id").cast(pl.Int32),
        pl.col("hours").cast(pl.Float32),
    )
    .sink_parquet(OUT / "recommendations.parquet")
)
print("recommendations parquet done")

pl.read_csv(RAW / "games.csv", try_parse_dates=True).write_parquet(OUT / "games.parquet")
pl.read_csv(RAW / "users.csv").write_parquet(OUT / "users.parquet")
pl.read_ndjson(RAW / "games_metadata.json").write_parquet(OUT / "games_metadata.parquet")
print("all done")

for f in sorted(OUT.glob("*.parquet")):
    print(f"{f.name:30s} {f.stat().st_size / 1e6:8.1f} MB")