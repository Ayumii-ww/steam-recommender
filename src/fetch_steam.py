"""Fetch game data (not user data) from SteamSpy and the Steam store, for games the Kaggle data doesn't cover.

Three steps, each resumable (rerunning skips what is already saved):
  list   SteamSpy "all" pages (sorted by owners) -> every app with MIN_REVIEWS+ reviews       ~20 min
  tags   SteamSpy appdetails -> user tags with vote counts, for every app in the list        1 request/s
  store  Steam store appdetails -> type, release date, genres, price, description, image,
         only for apps the model doesn't know (not in item_map)                              ~200 requests / 5 min

Output in data/raw/steam/ (gitignored): applist.parquet, tags.jsonl, store.jsonl

Run from the project root:
  .venv/bin/python -m src.fetch_steam list
  .venv/bin/python -m src.fetch_steam tags     (can run at the same time as store: different servers)
  .venv/bin/python -m src.fetch_steam store
"""

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "steam"
MIN_REVIEWS = 1_000

SPY = "https://steamspy.com/api.php"
STORE = "https://store.steampowered.com/api/appdetails"
HEADERS = {"User-Agent": "steam-recommender portfolio project (github.com/Ayumii-ww/steam-recommender)"}


def get_json(url: str, params: dict, wait: float) -> dict | None:
    """GET with polite pauses; back off on rate limits and server errors, give up after 5 tries."""
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                time.sleep(wait)
                return r.json()
            print(f"  HTTP {r.status_code} for {params}, retrying", flush=True)
        except (requests.RequestException, ValueError) as e:
            print(f"  {type(e).__name__} for {params}, retrying", flush=True)
        time.sleep(60 * (attempt + 1))   # 1, 2, 3... minutes
    return None


def done_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(line)["appid"] for line in path.open()}


def fetch_list() -> None:
    """Page through SteamSpy until a whole page has no app with MIN_REVIEWS+ reviews (pages are sorted by owners)."""
    rows, page = [], 0
    while True:
        data = get_json(SPY, {"request": "all", "page": page}, wait=60)   # SteamSpy allows 1 "all" request per minute
        if not data:
            break
        keep = [
            {"appid": int(a["appid"]), "name": a["name"], "positive": a["positive"], "negative": a["negative"],
             "owners": a["owners"], "price_cents": int(a["price"] or 0), "ccu": a["ccu"]}
            for a in data.values()
            if a["positive"] + a["negative"] >= MIN_REVIEWS
        ]
        print(f"page {page}: {len(data)} apps, {len(keep)} with {MIN_REVIEWS}+ reviews", flush=True)
        if not keep:
            break
        rows += keep
        page += 1
    apps = pl.DataFrame(rows).unique("appid")
    apps.write_parquet(OUT / "applist.parquet")
    print(f"saved {apps.height:,} apps", flush=True)


def fetch_tags() -> None:
    apps = pl.read_parquet(OUT / "applist.parquet")["appid"].to_list()
    path = OUT / "tags.jsonl"
    todo = [a for a in apps if a not in done_ids(path)]
    print(f"tags: {len(todo):,} of {len(apps):,} apps to fetch (~{len(todo) / 3600:.1f} h)", flush=True)
    with path.open("a") as f:
        for n, appid in enumerate(todo, 1):
            d = get_json(SPY, {"request": "appdetails", "appid": appid}, wait=1.0)
            if d is None:
                continue
            tags = d.get("tags") or {}
            f.write(json.dumps({"appid": appid, "genre": d.get("genre", ""),
                                "tags": tags if isinstance(tags, dict) else {}}) + "\n")
            f.flush()
            if n % 500 == 0:
                print(f"  tags {n:,}/{len(todo):,}", flush=True)


def fetch_store() -> None:
    known = set(pl.read_parquet(ROOT / "data" / "processed" / "item_map.parquet")["app_id"].to_list())
    apps = [a for a in pl.read_parquet(OUT / "applist.parquet")["appid"].to_list() if a not in known]
    path = OUT / "store.jsonl"
    todo = [a for a in apps if a not in done_ids(path)]
    print(f"store: {len(todo):,} of {len(apps):,} unknown apps to fetch (~{len(todo) * 1.6 / 3600:.1f} h)", flush=True)
    with path.open("a") as f:
        for n, appid in enumerate(todo, 1):
            d = get_json(STORE, {"appids": appid, "cc": "us", "l": "en"}, wait=1.6)   # ~190 requests / 5 min
            if d is None:
                continue
            entry = d.get(str(appid), {})
            info = entry.get("data", {}) if entry.get("success") else {}
            f.write(json.dumps({
                "appid": appid,
                "success": bool(entry.get("success")),
                "type": info.get("type"),
                "name": info.get("name"),
                "release_date": (info.get("release_date") or {}).get("date"),
                "coming_soon": (info.get("release_date") or {}).get("coming_soon"),
                "is_free": info.get("is_free"),
                "price_cents": (info.get("price_overview") or {}).get("initial"),
                "genres": [g["description"] for g in info.get("genres", [])],
                "categories": [c["description"] for c in info.get("categories", [])],
                "short_description": info.get("short_description"),
                "header_image": info.get("header_image"),
                "developers": info.get("developers", []),
            }) + "\n")
            f.flush()
            if n % 500 == 0:
                print(f"  store {n:,}/{len(todo):,}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "tags", "store"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"list": fetch_list, "tags": fetch_tags, "store": fetch_store}[args.step]()
    sys.exit(0)
