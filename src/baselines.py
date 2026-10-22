"""Simple baselines that every real model has to beat.

Run from the project root:  .venv/bin/python -m src.baselines
"""

import numpy as np
import polars as pl
import scipy.sparse as sp

from src.evaluate import run


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


if __name__ == "__main__":
    run(PopularityModel(), "popularity", stage="val")
    run(PopularityModel(), "popularity", stage="test")
