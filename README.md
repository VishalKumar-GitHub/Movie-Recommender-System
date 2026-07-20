# Movie Recommendation System
<img width="2560" height="1280" alt="Shimlana_banner_dark" src="https://github.com/user-attachments/assets/e4a7f900-b313-4c3d-83f8-2972ef7b2738" />

A content-based movie recommender built on TMDB metadata. Given a film, it returns the
ten most similar titles, ranked by cosine similarity over a TF-IDF representation of
genres, keywords, cast, director, and plot overview.

## Approach

| Stage | Detail |
|---|---|
| Data | TMDB API — ~500 popular films, detail responses cached to disk |
| Features | Genres (weighted 3x), keywords, top-5 cast, director (2x), overview |
| Vectorisation | TF-IDF, unigrams + bigrams, `min_df=2`, English stop words removed |
| Similarity | Cosine, self-similarity zeroed |
| Evaluation | Mean genre Jaccard overlap against a random baseline |

Multi-word names are underscore-joined so `Christopher_Nolan` is a single token. Genres are
repeated because TF-IDF weights by frequency, and one genre mention would otherwise be
drowned out by a 100-word overview.

## Why content-based and not collaborative?

Collaborative filtering learns from user rating patterns. TMDB's public API exposes only
aggregate vote averages, not per-user ratings, so a genuine CF model cannot be trained from
it. Rather than ship a CF component that silently returns nothing, this project is
content-based by design. Section 7 of the notebook shows how to blend in a real CF signal
fitted on the MovieLens ratings dataset.

## Setup

```bash
pip install -r requirements.txt

cp .env.example .env        # then add your key
export TMDB_API_KEY=your_key_here
```

Get a free key at https://www.themoviedb.org/settings/api (Developer plan, non-commercial).

## Usage

```bash
python recommender.py       # builds the model and prints a sample
jupyter notebook Movie_Recommendation_System.ipynb
```

```python
from recommender import build, save, load

rec = build()
save(rec)

for movie_id, title, score in rec.recommend_by_title("Dune: Part Two"):
    print(f"{score:.3f}  {title}")
```

First run fetches ~500 films and takes a couple of minutes. Subsequent runs read from
`.tmdb_cache/` and are near-instant.

## Limitations

- Catalogue is 500 *popular* films — skews recent and mainstream, and drifts week to week.
- Content-based only: no personalisation, no serendipity. The model finds films that look
  alike on paper, never "people like you also enjoyed".
- Evaluation uses genre overlap as a proxy. It confirms the model beats random; it does not
  confirm the recommendations are good.

## Roadmap

- `/discover/movie` for a larger, time-stratified catalogue
- MovieLens CF signal blended with the content score
- FastAPI serving endpoint

## Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB.

## Deployment (Streamlit)

Run locally with:

```bash
streamlit run app.py
```

To deploy on Streamlit Cloud (or similar) using GitHub:

1. Initialize a git repository and commit your files:

```bash
git init
git add .
git commit -m "Initial movie recommender + Streamlit app"
```

2. Create a GitHub repo and push:

```bash
git remote add origin https://github.com/<your-username>/<repo>.git
git branch -M main
git push -u origin main
```

3. In Streamlit Cloud, connect your GitHub repo, set an environment variable `TMDB_API_KEY` in the app settings, and deploy.

Notes:
- Add `recommender.joblib` to `.gitignore` (the app will build it on first run if missing).
- Ensure `requirements.txt` includes `streamlit`.
