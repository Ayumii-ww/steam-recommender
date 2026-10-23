"""Item-KNN: "people who played this also played...".

Two games are similar when the same users reviewed both (cosine similarity).
Each game keeps only its K most similar games. A user's score for a game is the sum of
similarities from the games they already reviewed, so every game they reviewed "votes" for its neighbours.

Run from the project root:  .venv/bin/python -m src.knn            (tune on validation, then final test)
                            .venv/bin/python -m src.knn --test-only --k 200 --weighting log_hours
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from implicit.nearest_neighbours import BM25Recommender, CosineRecommender

# The Run button runs this file on its own, so Python doesn't know about the `src` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import RESULTS, evaluate, load_eval_data, run, summarise  # noqa: E402

TUNE_USERS = 100_000  # validation sample size for comparing settings


class ItemKNNModel:
    """Item-based nearest neighbours.

    k:          neighbours kept per game
    weighting:  how much each review counts when measuring similarity and voting
                "binary"    = every review counts 1
                "log_hours" = log(1 + hours played), so a 300-hour review counts more than a 1-hour one
    similarity: "cosine" or "bm25" (BM25 down-weights very popular games more strongly)
    """

    def __init__(self, k: int = 100, weighting: str = "binary", similarity: str = "cosine"):
        if weighting not in ("binary", "log_hours"):
            raise ValueError(f"unknown weighting {weighting!r}")
        if similarity not in ("cosine", "bm25"):
            raise ValueError(f"unknown similarity {similarity!r}")
        self.k = k
        self.weighting = weighting
        self.similarity_name = similarity

    @property
    def name(self) -> str:
        return f"item_knn_{self.similarity_name}_{self.weighting}_k{self.k}"

    def fit(self, train: pl.DataFrame, n_users: int, n_items: int) -> None:
        if self.weighting == "binary":
            values = np.ones(train.height, dtype=np.float32)
        else:
            values = np.log1p(train["hours"].to_numpy()).astype(np.float32) + 1.0  # +1 so 0-hour reviews still count

        # users x games matrix: row = user, column = game, value = how much that review counts
        self.user_items = sp.csr_matrix(
            (values, (train["user_idx"].to_numpy(), train["item_idx"].to_numpy())),
            shape=(n_users, n_items),
        )

        knn = CosineRecommender(K=self.k) if self.similarity_name == "cosine" else BM25Recommender(K=self.k)
        knn.fit(self.user_items, show_progress=False)
        # games x games: row i holds game i's K nearest neighbours and their similarity
        self.item_similarity = knn.similarity.astype(np.float32).tocsr()

    def score(self, user_idx: np.ndarray, history: sp.csr_matrix) -> np.ndarray:
        # Each reviewed game votes for its neighbours: (users x games) @ (games x games)
        return (self.user_items[user_idx] @ self.item_similarity).toarray()


def tune() -> pl.DataFrame:
    """Compare settings on a validation sample; the test set stays untouched."""
    grid = [
        ItemKNNModel(k, weighting, "cosine")
        for weighting in ("binary", "log_hours")
        for k in (50, 100, 200, 400)
    ] + [ItemKNNModel(200, "binary", "bm25"), ItemKNNModel(200, "log_hours", "bm25")]

    print(f"Tuning {len(grid)} settings on {TUNE_USERS:,} validation users...")
    data = load_eval_data("val")  # load once, reuse for every setting

    rows = []
    for model in grid:
        t0 = time.time()
        model.fit(data.train, data.n_users, data.n_items)
        fit_s = time.time() - t0
        per_user = evaluate(model, data, max_users=TUNE_USERS)
        s = summarise(per_user, model.name, "val").filter(
            (pl.col("users") == "core") & (pl.col("target") == "play_next")
        )
        row = {
            "model": model.name, "k": model.k, "weighting": model.weighting,
            "similarity": model.similarity_name,
            "recall@10": s["recall@10"][0], "ndcg@10": s["ndcg@10"][0], "fit_seconds": round(fit_s, 1),
        }
        print(row)
        rows.append(row)

    table = pl.DataFrame(rows).sort("ndcg@10", descending=True)
    RESULTS.mkdir(exist_ok=True)
    table.write_csv(RESULTS / "knn_tuning.csv")
    with pl.Config(tbl_rows=20, tbl_cols=10, float_precision=4):
        print(table)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="skip tuning and run the final test")
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--weighting", default="binary")
    parser.add_argument("--similarity", default="cosine")
    args = parser.parse_args()

    if args.test_only:
        best = ItemKNNModel(args.k, args.weighting, args.similarity)
    else:
        table = tune()
        top = table.row(0, named=True)
        best = ItemKNNModel(top["k"], top["weighting"], top["similarity"])
        print(f"Best setting on validation: {best.name}")

    run(best, "item_knn", stage="test")
