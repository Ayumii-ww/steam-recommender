"""ALS (matrix factorisation): learn hidden "taste sliders" for every user and every game.

Each user and each game gets a short list of numbers (factors). A user's score for a game is the
dot product of the two lists: high when the user's tastes line up with what the game offers.
Hours played sets the *confidence* of each review: a 300-hour review pulls the sliders harder than a 1-hour one.

Thumbs-down handling (tested during tuning):
  "same"     = 👍 and 👎 reviews get the same confidence (only hours matter)
  "half"     = 👎 reviews get half the confidence of a 👍 with the same hours
  "drop"     = 👎 reviews are left out of training entirely (thumbs-up only)

Run from the project root:  .venv/bin/python -m src.als            (tune on validation, then final test)
                            .venv/bin/python -m src.als --test-only --factors 64 --negatives half
"""

import os

# implicit runs its own threads; letting numpy's BLAS also spawn threads makes it much slower
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from implicit.cpu.als import AlternatingLeastSquares  # noqa: E402

# The Run button runs this file on its own, so Python doesn't know about the `src` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import RESULTS, evaluate, load_eval_data, run, summarise  # noqa: E402

TUNE_USERS = 100_000  # validation sample size for comparing settings


class ALSModel:
    """Implicit-feedback ALS with log(hours) confidence.

    factors:        number of hidden taste sliders per user and game (memory: 6.2M users x factors x 4 bytes)
    regularization: penalty that stops the sliders from memorising noise
    alpha:          multiplies every confidence value (how much stronger a review is than "no review")
    iterations:     learning rounds
    negatives:      "same", "half" or "drop" (see module docstring)
    """

    def __init__(self, factors: int = 64, regularization: float = 0.05, alpha: float = 1.0,
                 iterations: int = 15, negatives: str = "same"):
        if negatives not in ("same", "half", "drop"):
            raise ValueError(f"unknown negatives option {negatives!r}")
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.negatives = negatives

    @property
    def name(self) -> str:
        return (f"als_f{self.factors}_reg{self.regularization}_a{self.alpha}"
                f"_it{self.iterations}_neg-{self.negatives}")

    def fit(self, train: pl.DataFrame, n_users: int, n_items: int) -> None:
        if self.negatives == "drop":
            train = train.filter(pl.col("is_recommended"))

        confidence = np.log1p(train["hours"].to_numpy()).astype(np.float32) + 1.0
        if self.negatives == "half":
            confidence = np.where(train["is_recommended"].to_numpy(), confidence, confidence * 0.5)

        user_items = sp.csr_matrix(
            (confidence.astype(np.float32), (train["user_idx"].to_numpy(), train["item_idx"].to_numpy())),
            shape=(n_users, n_items),
        )
        als = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=42,
        )
        # Start the sliders as small float32 random numbers ourselves; implicit's default
        # creates them as float64 first, which needs ~2x the memory for 6.2M users
        rng = np.random.default_rng(42)
        als.user_factors = (rng.standard_normal((n_users, self.factors), dtype=np.float32) * 0.01)
        als.item_factors = (rng.standard_normal((n_items, self.factors), dtype=np.float32) * 0.01)
        del train, confidence
        als.fit(user_items, show_progress=False)
        self.user_factors = als.user_factors   # n_users x factors
        self.item_factors = als.item_factors   # n_items x factors

    def score(self, user_idx: np.ndarray, history: sp.csr_matrix) -> np.ndarray:
        # How well each user's sliders line up with each game's sliders
        return self.user_factors[user_idx] @ self.item_factors.T


def tune() -> pl.DataFrame:
    """Two rounds on a validation sample: first how to treat 👎, then the main ALS settings one at a time."""
    print(f"Loading validation data (tuning on {TUNE_USERS:,} users)...")
    data = load_eval_data("val")
    rows = []

    def try_setting(model: ALSModel) -> dict:
        t0 = time.time()
        model.fit(data.train, data.n_users, data.n_items)
        fit_s = time.time() - t0
        per_user = evaluate(model, data, max_users=TUNE_USERS)
        s = summarise(per_user, model.name, "val").filter(pl.col("users") == "core")
        play = s.filter(pl.col("target") == "play_next")
        enjoy = s.filter(pl.col("target") == "enjoy")
        row = {
            "model": model.name, "factors": model.factors, "regularization": model.regularization,
            "alpha": model.alpha, "iterations": model.iterations, "negatives": model.negatives,
            "recall@10": play["recall@10"][0], "ndcg@10": play["ndcg@10"][0],
            "enjoy_recall@10": enjoy["recall@10"][0], "fit_seconds": round(fit_s, 1),
        }
        print(row, flush=True)
        rows.append(row)
        return row

    # Round 1: how to treat thumbs-down reviews (other settings fixed)
    round1 = [try_setting(ALSModel(negatives=neg)) for neg in ("same", "half", "drop")]
    best_neg = max(round1, key=lambda r: r["ndcg@10"])["negatives"]
    print(f"Round 1 winner: negatives={best_neg}", flush=True)

    # Round 2: change one setting at a time from the base (64 factors, reg 0.05, alpha 1).
    # Each fit takes ~6.5 min on 6.2M users, so a full grid would take many hours.
    for factors, regularization, alpha in [
        (32, 0.05, 1.0), (96, 0.05, 1.0),   # fewer / more taste sliders (128+ needs >3 GB on 8 GB RAM)
        (64, 0.01, 1.0), (64, 0.5, 1.0),    # weaker / stronger regularisation
        (64, 0.05, 0.5), (64, 0.05, 4.0),   # weaker / stronger confidence from hours
    ]:
        try_setting(ALSModel(factors, regularization, alpha, negatives=best_neg))

    table = pl.DataFrame(rows).sort("ndcg@10", descending=True)
    RESULTS.mkdir(exist_ok=True)
    table.write_csv(RESULTS / "als_tuning.csv")
    with pl.Config(tbl_rows=30, tbl_cols=12, float_precision=4):
        print(table)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="skip tuning and run the final test")
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--regularization", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--negatives", default="same")
    args = parser.parse_args()

    if args.test_only:
        best = ALSModel(args.factors, args.regularization, args.alpha, args.iterations, args.negatives)
    else:
        top = tune().row(0, named=True)
        best = ALSModel(top["factors"], top["regularization"], top["alpha"], top["iterations"], top["negatives"])
        print(f"Best setting on validation: {best.name}", flush=True)

    run(best, "als", stage="test")
