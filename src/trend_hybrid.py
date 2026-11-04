"""SASRec + trending: combine a user's personal taste with what everyone reviewed yesterday.

Score for each game = SASRec log-probability + beta * log(1 + reviews of the game in the last W days)
  beta = 0      -> plain SASRec (taste only)
  beta large    -> recent popularity (trend only)
If the blend beats both, personal history adds value on top of "what is hot right now".

Uses the SASRec networks already saved by src/sasrec.py (models/sasrec_val.pt, models/sasrec_test.pt),
so nothing is retrained.

Run from the project root:  .venv/bin/python -m src.trend_hybrid
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import RecentPopularityModel  # noqa: E402
from src.evaluate import RESULTS, evaluate, load_eval_data, run, summarise  # noqa: E402
from src.sasrec import MODELS, SASRecModel  # noqa: E402

TUNE_USERS = 100_000


class TrendHybridModel:
    def __init__(self, sasrec_path: Path, beta: float = 1.0, window_days: int = 1):
        self.sasrec_path = sasrec_path
        self.beta = beta
        self.window_days = window_days

    @property
    def name(self) -> str:
        return f"sasrec_trend_b{self.beta}_w{self.window_days}"

    def fit(self, train: pl.DataFrame, n_users: int, n_items: int) -> None:
        self.sasrec = SASRecModel()
        self.sasrec.load(self.sasrec_path, train, n_users, n_items)
        self.trend = RecentPopularityModel(self.window_days, anchor="target")
        self.trend.fit(train, n_users, n_items)

    def set_recommendation_time(self, times: pl.DataFrame) -> None:
        self.trend.set_recommendation_time(times)

    def score(self, user_idx: np.ndarray, history: sp.csr_matrix) -> np.ndarray:
        taste = self.sasrec.score(user_idx, history)
        taste -= logsumexp(taste, axis=1, keepdims=True)   # logits -> log-probabilities, same scale for every user
        return taste + self.beta * np.log1p(self.trend.score(user_idx, history))


def tune() -> pl.DataFrame:
    """Try blend weights on a validation sample (core users), reusing one fitted model."""
    data = load_eval_data("val")
    model = TrendHybridModel(MODELS / "sasrec_val.pt")
    model.fit(data.train, data.n_users, data.n_items)
    rows = []

    def try_setting(beta: float, window_days: int) -> dict:
        model.beta = beta
        if window_days != model.trend.window_days:
            model.trend.window_days = model.window_days = window_days
        s = summarise(evaluate(model, data, max_users=TUNE_USERS), model.name, "val").filter(pl.col("users") == "core")
        play, enjoy = s.filter(pl.col("target") == "play_next"), s.filter(pl.col("target") == "enjoy")
        row = {"beta": beta, "window_days": window_days, "recall@10": play["recall@10"][0],
               "ndcg@10": play["ndcg@10"][0], "enjoy_recall@10": enjoy["recall@10"][0]}
        print(row, flush=True)
        rows.append(row)
        return row

    best = max((try_setting(b, 1) for b in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 1000.0)), key=lambda r: r["ndcg@10"])
    try_setting(best["beta"], 7)

    table = pl.DataFrame(rows)
    table.write_csv(RESULTS / "sasrec_trend_tuning.csv")
    with pl.Config(tbl_rows=20, float_precision=4):
        print(table)
    return table


if __name__ == "__main__":
    best = tune().sort("ndcg@10", descending=True).row(0, named=True)
    print(f"Best on validation: beta={best['beta']}, window={best['window_days']} days", flush=True)
    run(TrendHybridModel(MODELS / "sasrec_test.pt", best["beta"], best["window_days"]), "sasrec_trend", stage="test")
