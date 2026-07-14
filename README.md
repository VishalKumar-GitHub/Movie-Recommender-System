# 🎬 Movie Recommender System — Item-Based Collaborative Filtering

A movie recommendation engine built on the **MovieLens 100K** dataset. Given a handful of movies you've rated, the system recommends what to watch next by finding films that other users rated similarly — the same core idea behind Netflix and Amazon recommendations.

## How It Works

**Item-based collaborative filtering** recommends movies based on relationships between *items* (movies) rather than users:

1. Build a **user × movie ratings matrix** from 100,000 ratings
2. Compute **Pearson correlations** between every pair of movies (`min_periods=80` to filter out movies with too few ratings and avoid noisy correlations)
3. Take the user's own ratings, look up each rated movie's correlation vector, and **weight similarities by the user's rating**
4. Rank the results — the top-scoring movies are the recommendations

The item-based approach scales better than user-based filtering (movies number in the thousands, users in the billions) and is more stable over time, since a movie's characteristics don't change the way user tastes do.

## Example

Rating *Titanic (1997)* highly surfaces correlated films like *Romeo and Juliet* — movies that the same audience rated the same way. The final recommender takes a personal ratings file (`My_Ratings.csv`) and produces a ranked top-10 watchlist.

## Project Structure

```
├── Project_8_-_Movie_Recommender_System.ipynb   # Main notebook (EDA → filtering → recommendations)
├── u.data                                        # MovieLens 100K ratings (user_id, item_id, rating, timestamp)
├── Movie_Id_Titles                               # Movie ID → title mapping
├── My_Ratings.csv                                # Your own ratings used to generate recommendations
└── README.md
```

## Pipeline

| Step | What Happens |
|------|--------------|
| 1. Data loading | Merge ratings with movie titles into a single dataframe |
| 2. EDA | Distribution of mean ratings and rating counts; most-rated vs. highest-rated movies |
| 3. Single-movie filter | Correlate one movie (e.g. Titanic) against all others as a proof of concept |
| 4. Full recommender | Movie–movie correlation matrix + personal ratings → ranked recommendations |

## Getting Started

```bash
git clone https://github.com/<your-username>/movie-recommender-system.git
cd movie-recommender-system
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook Project_8_-_Movie_Recommender_System.ipynb
```

To get your own recommendations, edit `My_Ratings.csv` with movies you've seen (exact titles from the dataset) and your 1–5 ratings, then run the final section of the notebook.

## Tech Stack

- **Python** — pandas, NumPy
- **Visualization** — Matplotlib, Seaborn
- **Method** — Item-based collaborative filtering with Pearson correlation

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 100,000 ratings from 943 users on 1,682 movies, collected by GroupLens Research.

## Possible Extensions

- Matrix factorization (SVD) for latent-factor recommendations
- Handling the cold-start problem for new users/movies
- Serving recommendations via a FastAPI endpoint

## Author

**Vishal Kumar** — ML/AI Engineer
[LinkedIn](https://www.linkedin.com/in/vishal-kumar-819585275/)
