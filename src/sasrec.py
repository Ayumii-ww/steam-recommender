"""SASRec: read each user's games in order and predict the next one (a small transformer).

How it works:
  1. Each game gets a learned vector (embedding); each position in the history gets one too.
  2. The user's last MAX_LEN games, oldest to newest, go through a few self-attention layers.
     Attention lets every position look back at earlier games and decide which ones matter most.
  3. The output at the newest position is a summary of "where this user is heading".
  4. Score for every game = dot product of that summary with the game's embedding.

Training: at every position the model must pick the true next game out of that game plus
N_NEGATIVES randomly sampled games (sampled softmax). Comparing against all 35k games every
step would not fit in 8 GB.

Run from the project root:
  .venv/bin/python -m src.sasrec --smoke                 (quick speed test on a small sample)
  .venv/bin/python -m src.sasrec                         (train on validation to pick epochs, then final test)
  .venv/bin/python -m src.sasrec --test-only --epochs 6  (final test only)
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import RESULTS, ROOT, evaluate, load_eval_data, run, summarise  # noqa: E402

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODELS = ROOT / "models"
TUNE_USERS = 100_000


class SASRecNet(nn.Module):
    """Item embeddings + position embeddings + causal transformer layers."""

    def __init__(self, n_items: int, max_len: int, dim: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)  # id 0 = padding, games are 1..n_items
        self.pos_emb = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        # Causal mask: position t may only look at positions <= t (no peeking at the future)
        mask = torch.triu(torch.full((max_len, max_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", mask)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (batch, max_len) item ids, left-padded with 0. Returns (batch, max_len, dim)."""
        positions = torch.arange(seq.shape[1], device=seq.device)
        x = self.item_emb(seq) * math.sqrt(self.item_emb.embedding_dim) + self.pos_emb(positions)
        # Zero out padded positions (as in the original SASRec). A key-padding mask would leave
        # padded positions with nothing to attend to, which produces NaNs.
        not_padding = (seq != 0).unsqueeze(-1)
        x = self.dropout(x) * not_padding
        x = self.layers(x, mask=self.causal_mask, is_causal=True)
        return self.norm(x) * not_padding


class SASRecModel:
    """Wrapper with the fit/score interface that evaluate.py expects."""

    def __init__(self, max_len: int = 50, dim: int = 64, n_layers: int = 2, n_heads: int = 2,
                 dropout: float = 0.2, epochs: int = 5, batch_size: int = 256, lr: float = 1e-3,
                 n_negatives: int = 256, max_train_users: int | None = None, seed: int = 42):
        self.max_len = max_len
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_negatives = n_negatives
        self.max_train_users = max_train_users
        self.seed = seed
        self.on_epoch_end = None  
        
    @property
    def name(self) -> str:
        return f"sasrec_len{self.max_len}_d{self.dim}_l{self.n_layers}_ep{self.epochs}"

    # ---------- data ----------

    def _build_sequences(self, train: pl.DataFrame, n_users: int) -> None:
        """Store every user's games in time order as one flat array plus start offsets (like a CSR row index).

        train arrives sorted by user, then date, then review_id (prepare.py writes it that way),
        so no re-sorting is needed.
        """
        users = train["user_idx"].to_numpy()
        if not (np.diff(users.astype(np.int64)) >= 0).all():
            raise ValueError("train rows must be sorted by user_idx (and by date within each user)")
        self.flat_items = (train["item_idx"].to_numpy() + 1).astype(np.int32)  # +1: 0 is padding
        counts = np.bincount(users, minlength=n_users)
        self.offsets = np.zeros(n_users + 1, dtype=np.int64)
        np.cumsum(counts, out=self.offsets[1:])

    def _last_items(self, user_idx: np.ndarray, length: int) -> np.ndarray:
        """Each user's most recent `length` games, left-padded with 0."""
        out = np.zeros((len(user_idx), length), dtype=np.int64)
        starts, ends = self.offsets[user_idx], self.offsets[user_idx + 1]
        for row, (s, e) in enumerate(zip(starts, ends)):
            s = max(s, e - length)
            if e > s:
                out[row, length - (e - s):] = self.flat_items[s:e]
        return out

    # ---------- training ----------

    def fit(self, train: pl.DataFrame, n_users: int, n_items: int) -> None:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.n_items = n_items
        self._build_sequences(train, n_users)

        # Users with at least 2 games have something to learn from (input -> next game)
        lengths = np.diff(self.offsets)
        trainable = np.flatnonzero(lengths >= 2)
        if self.max_train_users is not None and len(trainable) > self.max_train_users:
            trainable = rng.choice(trainable, self.max_train_users, replace=False)
        print(f"  training sequences: {len(trainable):,} users on {DEVICE}", flush=True)

        self.net = SASRecNet(n_items, self.max_len, self.dim, self.n_layers, self.n_heads, self.dropout).to(DEVICE)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            order = rng.permutation(trainable)
            total_loss, n_batches, t0 = 0.0, 0, time.time()
            for b in range(0, len(order), self.batch_size):
                users = order[b : b + self.batch_size]
                # Window of max_len + 1 games: inputs are games 1..L, targets are games 2..L+1
                window = self._last_items(users, self.max_len + 1)
                inputs = torch.from_numpy(window[:, :-1]).to(DEVICE)
                targets = torch.from_numpy(window[:, 1:]).to(DEVICE)

                hidden = self.net(inputs)                                     # (B, L, dim)
                pos_emb = self.net.item_emb(targets)                          # (B, L, dim)
                neg_ids = torch.randint(1, n_items + 1, (self.n_negatives,), device=DEVICE)
                neg_emb = self.net.item_emb(neg_ids)                          # (N, dim), shared by the batch

                pos_logit = (hidden * pos_emb).sum(-1, keepdim=True)         # (B, L, 1)
                neg_logit = hidden @ neg_emb.T                                # (B, L, N)
                # A sampled game that happens to be the true target must not count as a wrong answer
                neg_logit = neg_logit.masked_fill(targets.unsqueeze(-1) == neg_ids, float("-inf"))
                logits = torch.cat([pos_logit, neg_logit], dim=-1)

                valid = targets != 0                                          # skip padded positions
                labels = torch.zeros(int(valid.sum()), dtype=torch.long, device=DEVICE)  # true game is column 0
                loss = F.cross_entropy(logits[valid], labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                if n_batches % 2000 == 0:
                    rate = n_batches / (time.time() - t0)
                    left = (len(order) / self.batch_size - n_batches) / rate
                    print(f"    epoch {epoch}: batch {n_batches:,}, loss {total_loss / n_batches:.3f}, "
                          f"{rate:.0f} batches/s, ~{left / 60:.0f} min left", flush=True)

            print(f"  epoch {epoch}/{self.epochs}: loss {total_loss / max(n_batches, 1):.4f}, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch, self)

    # ---------- scoring ----------

    @torch.no_grad()
    def score(self, user_idx: np.ndarray, history: sp.csr_matrix) -> np.ndarray:
        self.net.eval()
        seq = torch.from_numpy(self._last_items(user_idx, self.max_len)).to(DEVICE)
        summary = self.net(seq)[:, -1, :]                       # newest position = "where the user is heading"
        item_vectors = self.net.item_emb.weight[1:]             # drop the padding row -> (n_items, dim)
        return (summary @ item_vectors.T).float().cpu().numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)


def train_on_validation(epochs: int, max_train_users: int | None = None, val_users: int = TUNE_USERS) -> pl.DataFrame:
    """Train on the val stage and score a validation sample after every epoch, to choose the epoch count."""
    data = load_eval_data("val")
    rows = []

    def check(epoch: int, model: SASRecModel) -> None:
        t0 = time.time()
        s = summarise(evaluate(model, data, max_users=val_users), model.name, "val").filter(pl.col("users") == "core")
        play, enjoy = s.filter(pl.col("target") == "play_next"), s.filter(pl.col("target") == "enjoy")
        row = {"epoch": epoch, "recall@10": play["recall@10"][0], "ndcg@10": play["ndcg@10"][0],
               "enjoy_recall@10": enjoy["recall@10"][0], "eval_seconds": round(time.time() - t0, 1)}
        print(f"  validation after epoch {epoch}: {row}", flush=True)
        rows.append(row)

    model = SASRecModel(epochs=epochs, max_train_users=max_train_users)
    model.on_epoch_end = check
    model.fit(data.train, data.n_users, data.n_items)
    model.save(MODELS / "sasrec_val.pt")
    return pl.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="1 epoch on 200k users, score 20k users")
    parser.add_argument("--test-only", action="store_true", help="skip validation and run the final test")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    if args.smoke:
        print(train_on_validation(epochs=1, max_train_users=200_000, val_users=20_000))
        sys.exit()

    if args.test_only:
        epochs = args.epochs
    else:
        curve = train_on_validation(epochs=args.epochs)
        RESULTS.mkdir(exist_ok=True)
        curve.write_csv(RESULTS / "sasrec_tuning.csv")
        epochs = int(curve.sort("ndcg@10", descending=True)["epoch"][0])
        print(curve)
        print(f"Best epoch count on validation: {epochs}", flush=True)

    final = SASRecModel(epochs=epochs)
    run(final, "sasrec", stage="test")
    final.save(MODELS / "sasrec_test.pt")
