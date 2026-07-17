"""
TMDb content-based movie recommender.

Fixes over the original notebook:
  - API key from env var (TMDB_API_KEY), not input()
  - detail calls run concurrently and are cached to disk
  - recommendations keep their similarity ranking (no set())
  - dead TMDb /ratings collaborative branch removed; optional MovieLens
    item-based CF can be layered on via `blend_with_cf`
  - artifacts saved with joblib, and only what's needed at inference
"""

from __future__ import annotations

import os
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = os.environ.get("TMDB_API_KEY")
if not API_KEY:
    raise SystemExit("Set TMDB_API_KEY in your environment.")

BASE = "https://api.themoviedb.org/3"
CACHE = pathlib.Path(".tmdb_cache")
CACHE.mkdir(exist_ok=True)

_session = requests.Session()


# --------------------------------------------------------------------------
# Data acquisition
# --------------------------------------------------------------------------

def _get(path: str, **params) -> dict:
    params.update(api_key=API_KEY, language="en-US")
    r = _session.get(f"{BASE}{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_popular(pages: int = 25) -> pd.DataFrame:
    """Fetch popular movies. 25 pages ~= 500 titles, a far better pool than 100."""
    rows = []
    for page in range(1, pages + 1):
        rows.extend(_get("/movie/popular", page=page)["results"])
    df = pd.DataFrame(rows).drop_duplicates(subset="id").reset_index(drop=True)
    return df


def fetch_details(movie_id: int) -> dict:
    """Detail lookup, cached on disk so reruns cost nothing."""
    f = CACHE / f"{movie_id}.json"
    if f.exists():
        return json.loads(f.read_text())
    data = _get(f"/movie/{movie_id}", append_to_response="keywords,credits")
    f.write_text(json.dumps(data))
    return data


def attach_details(df: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        df["details"] = list(pool.map(fetch_details, df["id"]))
    return df


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

def _genres(d: dict) -> str:
    return " ".join(g["name"] for g in d.get("genres", []))


def _keywords(d: dict) -> str:
    kw = d.get("keywords", {}).get("keywords", [])
    return " ".join(k["name"].replace(" ", "_") for k in kw)


def _cast(d: dict, n: int = 5) -> str:
    cast = d.get("credits", {}).get("cast", [])[:n]
    return " ".join(c["name"].replace(" ", "_") for c in cast)


def build_content(df: pd.DataFrame) -> pd.DataFrame:
    df["genres"] = df["details"].apply(_genres)
    df["keywords"] = df["details"].apply(_keywords)
    df["cast"] = df["details"].apply(_cast)
    df["overview"] = df["overview"].fillna("")
    # genres repeated so they carry more weight than a single overview mention
    df["content"] = (
        (df["genres"] + " ") * 3 + df["keywords"] + " " + df["cast"] + " " + df["overview"]
    )
    return df


def fit_similarity(df: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    matrix = tfidf.fit_transform(df["content"])
    sim = cosine_similarity(matrix, matrix)
    np.fill_diagonal(sim, 0.0)  # a movie is never its own recommendation
    return tfidf, sim


# --------------------------------------------------------------------------
# Recommendation
# --------------------------------------------------------------------------

class Recommender:
    def __init__(self, df: pd.DataFrame, sim: np.ndarray, tfidf: TfidfVectorizer):
        self.df = df.reset_index(drop=True)
        self.sim = sim
        self.tfidf = tfidf
        self.index_of = {mid: i for i, mid in enumerate(self.df["id"])}
        self.title_of = dict(zip(self.df["id"], self.df["title"]))

    def recommend(self, movie_id: int, k: int = 10) -> list[tuple[int, str, float]]:
        """Return top-k as (id, title, score), ordered by score. Order is preserved."""
        i = self.index_of.get(movie_id)
        if i is None:
            raise KeyError(f"movie_id {movie_id} not in catalogue")
        scores = self.sim[i]
        top = np.argsort(scores)[::-1][:k]
        return [(int(self.df.at[j, "id"]), self.df.at[j, "title"], float(scores[j])) for j in top]

    def recommend_by_title(self, title: str, k: int = 10):
        hits = self.df[self.df["title"].str.lower() == title.lower()]
        if hits.empty:
            raise KeyError(f"no movie titled {title!r}")
        return self.recommend(int(hits.iloc[0]["id"]), k)

    def blend_with_cf(
        self, movie_id: int, cf_scores: dict[int, float], alpha: float = 0.5, k: int = 10
    ):
        """
        Rank-blend content scores with an external collaborative-filtering signal
        (e.g. item-item similarity fitted on MovieLens ratings).
        alpha=1.0 is pure content, 0.0 is pure CF.
        """
        i = self.index_of[movie_id]
        content = {int(self.df.at[j, "id"]): float(self.sim[i][j]) for j in range(len(self.df))}
        combined = {}
        for mid in set(content) | set(cf_scores):
            if mid == movie_id:
                continue
            combined[mid] = alpha * content.get(mid, 0.0) + (1 - alpha) * cf_scores.get(mid, 0.0)
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(mid, self.title_of.get(mid, "?"), s) for mid, s in ranked]


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

SLIM_COLS = ["id", "title", "genres", "overview", "content", "release_date", "vote_average"]


def save(rec: Recommender, path: str = "recommender.joblib") -> None:
    joblib.dump(
        {"df": rec.df[SLIM_COLS], "sim": rec.sim.astype(np.float32), "tfidf": rec.tfidf},
        path,
        compress=3,
    )


def load(path: str = "recommender.joblib") -> Recommender:
    a = joblib.load(path)
    return Recommender(a["df"], a["sim"], a["tfidf"])


# --------------------------------------------------------------------------

def build() -> Recommender:
    df = attach_details(fetch_popular())
    df = build_content(df)
    tfidf, sim = fit_similarity(df)
    return Recommender(df, sim, tfidf)


if __name__ == "__main__":
    rec = build()
    save(rec)
    print(f"Catalogue: {len(rec.df)} movies\n")

    seed = int(rec.df.iloc[0]["id"])
    print(f"Because you watched: {rec.title_of[seed]}\n")
    for mid, title, score in rec.recommend(seed):
        print(f"  {score:.3f}  {title}")
