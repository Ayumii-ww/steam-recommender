"""Shared evaluation for every model: hide each user's last review and check where the model ranks it.

How it works for one user:
  1. The model sees the user's history (all reviews before the hidden one).
  2. The model gives every game a score.
  3. Games already in the history are removed (you can't recommend what they already reviewed).
  4. We find the rank of the hidden game among the remaining ~35k games.
  5. Recall@10 = 1 if rank is in the top 10, else 0.
     NDCG@10   = 1 / log2(rank + 1) if in the top 10, else 0 (rank 1 -> 1.0, rank 10 -> 0.29).

Scores are reported for four groups:
  core / light users  x  all hidden reviews ("play next") / only thumbs-up hidden reviews ("enjoy")

Every model must provide:
  fit(train: pl.DataFrame, n_users: int, n_items: int) -> None
  score(user_idx: np.ndarray, history: scipy.sparse.csr_matrix) -> np.ndarray of shape (len(user_idx), n_items)
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

DATA = Path("data/processed")
RESULTS = Path("results")
K = 10
BATCH_SIZE = 2048  # users scored at once: 2048 x 35k games x 4 bytes = ~290 MB


@dataclass
class EvalData:
    """Everything a model needs for one evaluation round (validation or test)."""

    stage: str              # "val" or "test"
    train: pl.DataFrame     # rows the model may learn from
    history: sp.csr_matrix  # users x items, 1 = user reviewed the game before the hidden one
    targets: pl.DataFrame   # one hidden review per evaluated user
    n_users: int
    n_items: int


def load_eval_data(stage: str) -> EvalData:
    """Build training rows, history matrix and hidden targets for one stage.

    stage="val":  learn from train rows, hide the val review (core users only).
    stage="test": learn from train + val rows, hide the test review (core and light users).
    """
    if stage not in ("val", "test"):
        raise ValueError(f"stage must be 'val' or 'test', got {stage!r}")

    df = pl.read_parquet(
        DATA / "interactions.parquet",
        columns=["user_idx", "item_idx", "date", "is_recommended", "hours", "user_group", "split"],
    )
    n_users = int(df["user_idx"].max()) + 1
    n_items = int(df["item_idx"].max()) + 1

    learn_splits = ["train"] if stage == "val" else ["train", "val"]
    train = df.filter(pl.col("split").is_in(learn_splits))
    targets = df.filter(pl.col("split") == stage).select("user_idx", "item_idx", "is_recommended", "user_group")
    del df

    history = sp.csr_matrix(
        (
            np.ones(train.height, dtype=np.float32),
            (train["user_idx"].to_numpy(), train["item_idx"].to_numpy()),
        ),
        shape=(n_users, n_items),
    )
    return EvalData(stage, train, history, targets, n_users, n_items)


def rank_of_targets(scores: np.ndarray, history: sp.csr_matrix, target_items: np.ndarray) -> np.ndarray:
    """1-based rank of each hidden game among the games the user hasn't reviewed yet.

    Ties are broken by item_idx (lower index ranks first), so results are reproducible.
    """
    scores = scores.astype(np.float32, copy=True)

    # Remove already-reviewed games: they can never be recommended
    rows = np.repeat(np.arange(history.shape[0]), np.diff(history.indptr))
    scores[rows, history.indices] = -np.inf

    rows = np.arange(len(target_items))
    target_scores = scores[rows, target_items][:, None]
    higher = (scores > target_scores).sum(axis=1)
    tied_before = ((scores == target_scores) & (np.arange(scores.shape[1]) < target_items[:, None])).sum(axis=1)
    return higher + tied_before + 1


def evaluate(model, data: EvalData, max_users: int | None = None, seed: int = 42) -> pl.DataFrame:
    """Rank every hidden review and return one row per evaluated user with rank, recall and ndcg."""
    targets = data.targets
    if max_users is not None and targets.height > max_users:
        targets = targets.sample(n=max_users, seed=seed)

    user_idx = targets["user_idx"].to_numpy()
    target_items = targets["item_idx"].to_numpy()
    ranks = np.empty(len(user_idx), dtype=np.int64)

    start = time.time()
    for b in range(0, len(user_idx), BATCH_SIZE):
        users = user_idx[b : b + BATCH_SIZE]
        hist = data.history[users]
        scores = model.score(users, hist)
        ranks[b : b + BATCH_SIZE] = rank_of_targets(scores, hist, target_items[b : b + BATCH_SIZE])
        done = min(b + BATCH_SIZE, len(user_idx))
        if (b // BATCH_SIZE) % 200 == 0:
            print(f"  {done:,}/{len(user_idx):,} users  ({time.time() - start:.0f}s)")

    in_top_k = ranks <= K
    return targets.with_columns(
        pl.Series("rank", ranks),
        pl.Series("recall", in_top_k.astype(np.float64)),
        pl.Series("ndcg", np.where(in_top_k, 1.0 / np.log2(ranks + 1), 0.0)),
    )


def summarise(per_user: pl.DataFrame, model_name: str, stage: str) -> pl.DataFrame:
    """Average recall and ndcg for each user group, for all hidden reviews and thumbs-up only."""
    groups = [
        ("all_users", pl.lit(True)),
        ("core", pl.col("user_group") == "core"),
        ("light", pl.col("user_group") == "light"),
    ]
    rows = []
    for group_name, group_filter in groups:
        for target_name, target_filter in [("play_next", pl.lit(True)), ("enjoy", pl.col("is_recommended"))]:
            subset = per_user.filter(group_filter & target_filter)
            if subset.height == 0:
                continue
            rows.append({
                "model": model_name,
                "stage": stage,
                "users": group_name,
                "target": target_name,
                "n": subset.height,
                f"recall@{K}": subset["recall"].mean(),
                f"ndcg@{K}": subset["ndcg"].mean(),
            })
    return pl.DataFrame(rows)


def save_results(summary: pl.DataFrame) -> None:
    """Append to results/results.csv, replacing earlier rows for the same model and stage."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "results.csv"
    if path.exists():
        old = pl.read_csv(path)
        keys = summary.select("model", "stage").unique()
        old = old.join(keys, on=["model", "stage"], how="anti")
        summary = pl.concat([old, summary], how="diagonal_relaxed")
    summary.sort("model", "stage", "users", "target").write_csv(path)


def run(model, model_name: str, stage: str = "test", max_users: int | None = None) -> pl.DataFrame:
    """Fit the model on the stage's training rows, evaluate, save and print the summary."""
    print(f"[{model_name}] loading {stage} data...")
    data = load_eval_data(stage)

    print(f"[{model_name}] fitting on {data.train.height:,} rows...")
    t0 = time.time()
    model.fit(data.train, data.n_users, data.n_items)
    fit_seconds = time.time() - t0

    print(f"[{model_name}] evaluating...")
    t0 = time.time()
    per_user = evaluate(model, data, max_users=max_users)
    eval_seconds = time.time() - t0

    summary = summarise(per_user, model_name, stage).with_columns(
        pl.lit(round(fit_seconds, 1)).alias("fit_seconds"),
        pl.lit(round(eval_seconds, 1)).alias("eval_seconds"),
    )
    if max_users is None:
        save_results(summary)
    with pl.Config(tbl_rows=20, tbl_cols=12, float_precision=4):
        print(summary)
    return summary


if __name__ == "__main__":
    # Quick self-check: a model that gives random scores should score close to random chance.
    class RandomModel:
        def fit(self, train, n_users, n_items):
            self.n_items = n_items
            self.rng = np.random.default_rng(0)

        def score(self, user_idx, history):
            return self.rng.random((len(user_idx), self.n_items), dtype=np.float32)

    summary = run(RandomModel(), "random", stage="test", max_users=50_000)
    print(json.dumps({"expected_recall_about": K / 35_000}, indent=2))
