"""Gradio demo: pick the games you played, in order, and get 10 recommendations.

Three lists:
  Taste + trending    SASRec plus what everyone reviewed the day before the chosen date (best model)
  Your taste          SASRec reads your games in order and predicts the next one
  Trending only       just what everyone reviewed that day (ignores your games)

Needs models/sasrec_test.pt (from src/sasrec.py) and models/demo/ (from app/build_assets.py).
Run from the project root:  .venv/bin/python app/app.py   then open http://localhost:7860
"""

import html
import json
import sys
from datetime import date
from pathlib import Path

import gradio as gr
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sasrec import SASRecNet  # noqa: E402

DEMO = ROOT / "models" / "demo"
MAX_LEN = 50
BETA = 1.0              # weight of the trending score, tuned on validation (src/trend_hybrid.py)
N_CHOICES = 10_000      # most-reviewed games offered in the dropdown (the model knows all 35k)
TOP_K = 10
IMAGE = "https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{}/header.jpg"
STORE = "https://store.steampowered.com/app/{}"

# ---------- load everything once at start-up ----------
games = pl.read_parquet(DEMO / "games.parquet")
n_items = games.height
daily = sp.load_npz(DEMO / "daily_counts.npz").tocsr()
meta = json.loads((DEMO / "meta.json").read_text())
FIRST_DATE = date.fromordinal(date(1970, 1, 1).toordinal() + meta["first_day"])
LAST_DATE = date.fromordinal(FIRST_DATE.toordinal() + meta["n_days"] - 1)

net = SASRecNet(n_items, MAX_LEN, dim=64, n_layers=2, n_heads=2, dropout=0.2)
net.load_state_dict(torch.load(ROOT / "models" / "sasrec_test.pt", map_location="cpu"))
net.eval()

all_time = games["n_reviews"].fill_null(0).to_numpy().astype(np.float64)
TIEBREAK = 0.5 * all_time / (all_time.max() + 1)   # rank days with few reviews by all-time popularity

# Dropdown labels like "Terraria (2011)"; add the app id if two games share a label
popular = games.sort("n_reviews", descending=True).head(N_CHOICES)
labels = [f"{t} ({y})" if y else t for t, y in zip(popular["title"], popular["year"])]
label_to_item = {}
for label, item, app_id in zip(labels, popular["item_idx"], popular["app_id"]):
    if label in label_to_item:
        label = f"{label} #{app_id}"
    label_to_item[label] = int(item)
CHOICES = sorted(label_to_item, key=str.casefold)


# ---------- scoring ----------
def taste_scores(history: list[int]) -> np.ndarray:
    """SASRec log-probability of every game being next, given the games in order (oldest first)."""
    seq = np.zeros(MAX_LEN, dtype=np.int64)
    recent = [i + 1 for i in history[-MAX_LEN:]]      # +1: id 0 is padding in the model
    seq[MAX_LEN - len(recent):] = recent
    with torch.no_grad():
        summary = net(torch.from_numpy(seq)[None])[0, -1]
        logits = summary @ net.item_emb.weight[1:].T
        return torch.log_softmax(logits, dim=0).numpy().astype(np.float64)


def trending_counts(day: date) -> np.ndarray:
    """Reviews of every game on the day before `day` (yesterday), as in the tuned hybrid."""
    row = day.toordinal() - FIRST_DATE.toordinal() - 1
    if not 0 <= row < daily.shape[0]:
        return np.zeros(n_items)
    return daily[row].toarray().ravel().astype(np.float64)


def top_items(scores: np.ndarray, exclude: list[int]) -> list[int]:
    scores = scores.copy()
    scores[exclude] = -np.inf
    return np.argsort(-scores, kind="stable")[:TOP_K].tolist()


# ---------- HTML ----------
def card(item: int, rank: int) -> str:
    g = games.row(item, named=True)
    title = html.escape(g["title"] or "")
    price = "" if g["price"] is None else ("Free" if g["price"] == 0 else f"${g['price']:.2f}")
    ratio = g["positive_ratio"]
    mood = "good" if ratio is not None and ratio >= 80 else "mixed" if ratio is not None and ratio >= 60 else "poor"
    facts = " · ".join(str(x) for x in (g["year"], price) if x not in (None, ""))
    score = f'<span class="score {mood}" title="{ratio}% of Steam reviews are positive">{ratio}% positive</span>' if ratio is not None else ""
    return (
        f'<a class="card" href="{STORE.format(g["app_id"])}" target="_blank" rel="noopener">'
        f'<div class="thumb"><img src="{IMAGE.format(g["app_id"])}" alt="" loading="lazy"><span class="rank">{rank}</span></div>'
        f'<div class="body"><div class="title" title="{title}">{title}</div>'
        f'<div class="meta"><span>{facts}</span>{score}</div></div></a>'
    )


def grid(items: list[int], caption: str) -> str:
    return f'<p class="caption">{caption}</p><div class="grid">' + "".join(
        card(item, rank) for rank, item in enumerate(items, 1)) + "</div>"


def history_strip(history: list[int], day: date) -> str:
    chips = '<span class="arrow">→</span>'.join(
        f'<span class="chip"><img src="{IMAGE.format(games["app_id"][i])}" alt="">'
        f'{html.escape(games["title"][i])}</span>' for i in history)
    return (f'<div class="strip"><span class="label">Based on</span>{chips}'
            f'<span class="label">on {day.strftime("%-d %b %Y")}</span></div>')


def recommend(chosen: list[str], day_text: str):
    if not chosen:
        raise gr.Error("Pick at least one game you have played.")
    try:
        day = date.fromisoformat(day_text.strip())
    except ValueError:
        raise gr.Error("Date must look like 2022-06-15.")
    day = min(max(day, FIRST_DATE), LAST_DATE)

    history = [label_to_item[c] for c in chosen]
    taste = taste_scores(history)
    trend = trending_counts(day)
    yesterday = date.fromordinal(day.toordinal() - 1).strftime("%-d %b %Y")

    return (
        history_strip(history, day),
        grid(top_items(taste + BETA * np.log1p(trend + TIEBREAK), history),
             "Your taste from SASRec, boosted by what everyone reviewed the day before. "
             "Best model on the test set: the next game is in this top 10 for 23.5% of users with 5+ reviews."),
        grid(top_items(taste, history),
             "SASRec alone reads your games in order and predicts the next one. It ignores the date."),
        grid(top_items(trend + TIEBREAK, history),
             f"Most-reviewed games on {yesterday} ({int(trend.sum()):,} reviews that day). Ignores your games."),
    )


# ---------- page ----------
CSS = """
.gradio-container { max-width: 1180px !important; margin: auto; }
#hero h1 { font-size: 2rem; margin-bottom: .2rem; }
#hero p { opacity: .8; margin-top: 0; }
.caption { opacity: .75; font-size: .92rem; margin: .2rem 0 .8rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 14px; }
.card { display: block; border-radius: 10px; overflow: hidden; text-decoration: none !important; color: inherit !important;
        background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
        transition: transform .15s ease, box-shadow .15s ease; }
.card:hover { transform: translateY(-3px); box-shadow: 0 8px 22px rgba(0,0,0,.25); }
.thumb { position: relative; aspect-ratio: 460 / 215; background: var(--neutral-800); }
.thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
.rank { position: absolute; top: 6px; left: 6px; min-width: 26px; padding: 2px 7px; border-radius: 999px;
        background: rgba(0,0,0,.72); color: #fff; font-weight: 700; font-size: .85rem; text-align: center; }
.body { padding: 9px 11px 11px; text-align: left; }
.title { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.meta { display: flex; justify-content: space-between; gap: 6px; font-size: .8rem; opacity: .85; margin-top: 3px; white-space: nowrap; }
.score.good { color: #4fa3e3; } .score.mixed { color: #c9a227; } .score.poor { color: #d0614f; }
.strip { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: .9rem; }
.strip .label { opacity: .7; }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px 3px 3px; border-radius: 999px;
        background: var(--block-background-fill); border: 1px solid var(--border-color-primary); }
.chip img { height: 22px; border-radius: 999px; aspect-ratio: 1; object-fit: cover; }
.arrow { opacity: .6; }
"""
THEME = gr.themes.Soft(primary_hue="sky", neutral_hue="slate", font=gr.themes.GoogleFont("Inter"))

with gr.Blocks(title="Steam game recommender") as demo:
    gr.Markdown(
        "# What will you play next?\n"
        "Pick games you have played **in the order you played them**, choose a date, and get 10 games you are "
        "likely to play next. Trained on 33M Steam reviews (2010–2022). "
        "[How it works and how well it scores →](https://github.com/Ayumii-ww/steam-recommender)",
        elem_id="hero",
    )
    with gr.Row(equal_height=True):
        chosen = gr.Dropdown(CHOICES, multiselect=True, max_choices=MAX_LEN, scale=4,
                             label="Games you played, oldest first", info="Type to search 10,000 games")
        day_text = gr.Textbox(value=LAST_DATE.isoformat(), scale=1, label="Date",
                              info=f"{FIRST_DATE.year}–{LAST_DATE.year}, as YYYY-MM-DD")
        button = gr.Button("Recommend", variant="primary", scale=1, size="lg")
    # Results are created first (not shown yet) so the examples above them can fill them in
    strip, hybrid_view, taste_view, trend_view = (gr.HTML(render=False) for _ in range(4))
    outputs = [strip, hybrid_view, taste_view, trend_view]
    gr.Examples(
        examples=[
            [["Terraria (2011)", "Stardew Valley (2016)"], "2022-06-15"],
            [["DARK SOULS™ III (2016)", "Sekiro™: Shadows Die Twice - GOTY Edition (2019)"], "2022-03-01"],
            [["Counter-Strike: Global Offensive (2012)", "Tom Clancy's Rainbow Six® Siege (2015)"], "2021-11-20"],
        ],
        inputs=[chosen, day_text],
        outputs=outputs,
        fn=recommend,
        run_on_click=True,
        cache_examples=False,
        label="Try an example",
    )
    strip.render()
    with gr.Tabs():
        with gr.Tab("Taste + trending (best)"):
            hybrid_view.render()
        with gr.Tab("Your taste only"):
            taste_view.render()
        with gr.Tab("Trending only"):
            trend_view.render()

    button.click(recommend, [chosen, day_text], outputs)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
