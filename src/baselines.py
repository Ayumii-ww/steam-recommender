"""Simple baselines that every real model has to beat.

Run from the project root:  .venv/bin/python -m src.baselines            (all-time popularity)
                            .venv/bin/python -m src.baselines --recent   (recent popularity: tune window, then test)
(or with the VS Code Run button)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

# The Run button runs this file on its own, so Python doesn't know about the `src` package.
# Adding the project folder to the import path makes `from src...` work both ways.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import RESULTS, evaluate, load_eval_data, run, summarise  # noqa: E402

TUNE_USERS = 100_000


class PopularityModel:
    """Recommend the most-reviewed games to everyone (minus games the user already reviewed).

    It knows nothing about the user, so it shows how much a personalised model really adds.
    """

    def fit(self, train: pl.DataFrame, n_users: int, n_items: int) -> None:
        counts = train.group_by("item_idx").len()
        self.item_scores = np.zeros(n_items, dtype=np.float32)
        self.item_scores[counts["item_idx"].to_numpy()] = counts["len"].to_numpy()

    def score(self, user_idx: np.ndarray, history: sp.csr_matrix) -> np.ndarray:
        # Same scores for every user; evaluate.py removes games they already reviewed
        return np.broadcast_to(self.item_scores, (len(user_idx), len(self.item_scores)))


class RecentPopularityModel:
    """Recommend the games most reviewed in the `window_days` days before the moment we recommend.

    Still knows nothing about the user's tastes, only *when* they are. It measures how much of a
    model's score comes from knowing which games were in fashion at that time.

    anchor="target":       the moment = the date of the hidden review (as if the user opened the app that day)
    anchor="last_history": the moment = the date of the user's last known review (no knowledge of the future date)
    Only reviews strictly before the moment are counted, so no future data is used.
    Ties (e.g. zero reviews in a short window) are broken by all-time popularity.
    """

    def __init__(self, window_days: int = 30, anchor: str = "target"):
        if anchor not in ("target", "last_history"):
            raise ValueError(f"unknown anchor {anchor!r}")
        self.window_days = window_days
        self.anchor = anchor

    @property
    def name(self) -> str:
        return f"recent_pop_w{self.window_days}_{self.anchor}"

    def fit(self, train: pl.DataFrame, n_users: int, n_items: int) -> None:
        day = train["date"].cast(pl.Int32).to_numpy()   # days since 1970-01-01
        self.first_day = int(day.min())
        n_days = int(day.max()) - self.first_day + 1
        day = day - self.first_day

        # cum[d, item] = reviews of the item on days before day d (a running total, like a cumulative sum in SQL)
        # Window count for days [d - W, d) = cum[d] - cum[d - W]
        daily = sp.csr_matrix(
            (np.ones(len(day), dtype=np.int32), (day + 1, train["item_idx"].to_numpy())),
            shape=(n_days + 1, n_items),
        )
        self.cum = daily.toarray()                       # (n_days + 1) x n_items int32, ~0.6 GB
        np.cumsum(self.cum, axis=0, out=self.cum)

        all_time = self.cum[-1].astype(np.float64)
        self.tiebreak = (0.5 * all_time / (all_time.max() + 1)).astype(np.float32)  # < 1, so never beats a real count

        # Default moment = day after the user's last known review (used by anchor="last_history")
        last = train.group_by("user_idx").agg(pl.col("date").max().cast(pl.Int32))
        self.moment = np.full(n_users, n_days, dtype=np.int32)
        self.moment[last["user_idx"].to_numpy()] = last["date"].to_numpy() - self.first_day + 1

    def set_recommendation_time(self, times: pl.DataFrame) -> None:
        if self.anchor == "target":
            d = times["date"].cast(pl.Int32).to_numpy() - self.first_day
            self.moment[times["user_idx"].to_numpy()] = np.clip(d, 0, self.cum.shape[0] - 1)

    def score(self, user_idx: np.ndarray, history: sp.csr_matrix) -> np.ndarray:
        end = self.moment[user_idx]
        start = np.maximum(end - self.window_days, 0)
        counts = (self.cum[end] - self.cum[start]).astype(np.float32)
        return counts + self.tiebreak


def tune_recent() -> pl.DataFrame:
    """Compare window lengths on a validation sample of core users, then the best window with the other anchor."""
    data = load_eval_data("val")
    rows = []

    def try_setting(model: RecentPopularityModel) -> dict:
        model.fit(data.train, data.n_users, data.n_items)
        s = summarise(evaluate(model, data, max_users=TUNE_USERS), model.name, "val").filter(pl.col("users") == "core")
        play, enjoy = s.filter(pl.col("target") == "play_next"), s.filter(pl.col("target") == "enjoy")
        row = {"model": model.name, "window_days": model.window_days, "anchor": model.anchor,
               "recall@10": play["recall@10"][0], "ndcg@10": play["ndcg@10"][0],
               "enjoy_recall@10": enjoy["recall@10"][0]}
        print(row, flush=True)
        rows.append(row)
        return row

    best = max((try_setting(RecentPopularityModel(w, "target")) for w in (1, 7, 30, 90, 365)),
               key=lambda r: r["ndcg@10"])
    try_setting(RecentPopularityModel(best["window_days"], "last_history"))

    table = pl.DataFrame(rows)
    table.write_csv(RESULTS / "recent_pop_tuning.csv")
    with pl.Config(tbl_rows=20, tbl_cols=8, float_precision=4):
        print(table)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent", action="store_true", help="recent popularity: tune window, then test")
    args = parser.parse_args()

    if args.recent:
        table = tune_recent()
        best_window = int(table.filter(pl.col("anchor") == "target").sort("ndcg@10", descending=True)["window_days"][0])
        run(RecentPopularityModel(best_window, "target"), "recent_pop", stage="test")
        run(RecentPopularityModel(best_window, "last_history"), "recent_pop_lasthist", stage="test")
    else:
        run(PopularityModel(), "popularity", stage="val")
        run(PopularityModel(), "popularity", stage="test")
