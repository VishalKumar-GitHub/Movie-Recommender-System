import os
import joblib
import streamlit as st
from recommender import build, load

MODEL_PATH = "recommender.joblib"

st.set_page_config(page_title="Shimlana — Movie Recommender", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
      .stApp { background: #17121A; color: #0E0B10; font-family: Inter, Helvetica, Arial, sans-serif; }
      .shimlana-logo { position:fixed; left:16px; top:12px; z-index:999; color:#FF2E4C; font-size:50px; font-weight:900; letter-spacing:0.08em; text-transform:uppercase; }
      .hero { position:relative; min-height:520px; border-bottom:1px solid #2B2230; display:flex; align-items:flex-end; padding:48px 64px; overflow:hidden; }
      .hero::before { content:''; position:absolute; inset:0; background:linear-gradient(90deg, rgba(23,18,26,0.96) 0%, rgba(23,18,26,0.45) 35%, rgba(23,18,26,0.16) 100%); }
      .hero-bg { position:absolute; inset:0; background-size:cover; background-position:center; filter:brightness(0.75); }
      .hero-copy { position:relative; max-width:640px; z-index:1; }
      .hero-eyebrow { text-transform:uppercase; letter-spacing:0.3em; font-size:12px; color:#FF2E4C; margin-bottom:16px; }
      .hero-title { font-family:'Bricolage Grotesque', Inter, Helvetica, Arial, sans-serif; font-size:62px; line-height:0.96; margin:0 0 18px; color:#FFFFFF; }
      .hero-overview { max-width:600px; color:#D4D0D6; font-size:18px; line-height:1.8; margin:0; }
      .sticky-search { position:sticky; top:0; z-index:998; background:#17121A; border-bottom:1px solid #2B2230; padding:20px 64px 18px; }
      .search-row { display:flex; gap:16px; max-width:1080px; margin:0 auto; }
      .search-input input { width:100%; background:#1F1726 !important; color:#FFFFFF !important; border:1px solid #2B2230 !important; border-radius:12px !important; padding:16px 18px !important; font-size:16px; }
      .search-input input::placeholder { color:#7A7482 !important; }
      .stButton>button { min-height:52px; background:#FF2E4C; color:#0E0B10; border-radius:12px; border:none; font-weight:700; padding:0 24px; }
      .stButton>button:hover { background:#ff616f; }
      .row-section { padding:38px 0 0; }
      .row-header { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:18px; }
      .row-label { font-size:22px; font-weight:700; color:#FFFFFF; }
      .row-note { color:#7A7482; font-size:14px; }
      .row-scroll { display:flex; gap:16px; overflow-x:auto; padding-bottom:4px; }
      .row-scroll::-webkit-scrollbar { height:10px; }
      .row-scroll::-webkit-scrollbar-thumb { background:#2B2230; border-radius:6px; }
      .poster-card { width:220px; min-width:220px; background:#17121A; border:1px solid #2B2230; border-radius:18px; overflow:hidden; transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease; }
      .poster-card:hover { transform:translateY(-6px); border-color:#FF2E4C; box-shadow:0 16px 48px rgba(0,0,0,0.35); }
      .poster-card img { width:100%; height:auto; display:block; }
      .poster-meta { padding:14px 14px 18px; }
      .poster-title { font-size:15px; font-weight:700; color:#FFFFFF; margin:0 0 4px; min-height:38px; line-height:1.2; }
      .poster-year { color:#7A7482; font-size:13px; margin-bottom:12px; }
      .poster-score { display:flex; align-items:center; justify-content:space-between; gap:8px; font-family:'JetBrains Mono', monospace; font-size:13px; color:#FFFFFF; }
      .poster-score span { color:#FF2E4C; font-weight:700; }
      .hero-stat { display:inline-flex; align-items:center; gap:10px; margin-top:28px; }
      .hero-stat div { padding:10px 16px; background:rgba(255,46,76,0.12); border:1px solid rgba(255,46,76,0.18); border-radius:999px; color:#FFFFFF; font-size:13px; }
      .app-footer { padding:32px 64px 48px; color:#7A7482; font-size:13px; }
      .app-section { padding:0 64px; }
      .stTextInput>label { color:#FFFFFF; font-size:14px; margin-bottom:8px; }
      .stTextInput>div>div>input { background:#1F1726 !important; border:1px solid #2B2230 !important; color:#FFFFFF !important; }
      .stTextInput>div>div>input:focus { border-color:#FF2E4C !important; box-shadow:0 0 0 3px rgba(255,46,76,0.14) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_model(path: str = MODEL_PATH):
    if os.path.exists(path):
        return load(path)
    rec = build()
    save(rec, path)
    return rec

rec = load_model()

if "selected_title" not in st.session_state:
    st.session_state["selected_title"] = rec.df.iloc[0]["title"] if not rec.df.empty else ""

if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""

if "history" not in st.session_state:
    st.session_state["history"] = []

if "favorites" not in st.session_state:
    st.session_state["favorites"] = []


def poster_url(path: str | None) -> str | None:
    return f"https://image.tmdb.org/t/p/w500{path}" if path else None


def backdrop_url(path: str | None) -> str | None:
    return f"https://image.tmdb.org/t/p/original{path}" if path else None


def year_of(release_date: str | None) -> str:
    if not release_date:
        return "—"
    return release_date[:4]


def score_map_for(title: str) -> dict[int, int]:
    idx = rec.index_of.get(title)
    if idx is None:
        return {}
    scores = {int(rec.df.at[j, "id"]): int(rec.sim[idx][j] * 100) for j in range(len(rec.df))}
    return scores


def genre_movies(primary_genre: str) -> list[dict]:
    if not primary_genre:
        return []
    mask = rec.df["genres"].str.contains(fr"\b{primary_genre}\b", case=False, na=False)
    return rec.df[mask].nlargest(10, "vote_average").to_dict(orient="records")


def make_card(movie: dict, score: int) -> str:
    poster = poster_url(movie.get("poster_path"))
    year = year_of(movie.get("release_date"))
    title = movie.get("title", "Untitled")
    score_text = f"{score}%" if score else "—"
    poster_img = f"<img src='{poster}' alt='{title}'/>" if poster else "<div class='no-image'>No image</div>"
    return f"""
    <div class='poster-card'>
      {poster_img}
      <div class='poster-meta'>
        <div class='poster-title'>{title}</div>
        <div class='poster-year'>{year}</div>
        <div class='poster-score'><span>{score_text}</span><span>Match</span></div>
      </div>
    </div>
    """


def render_row(label: str, subtitle: str, movies: list[dict], scores: dict[int, int]) -> str:
    cards = "".join(make_card(movie, scores.get(int(movie["id"]), 0)) for movie in movies)
    return f"""
    <div class='row-section'>
      <div class='row-header'>
        <div>
          <div class='row-label'>{label}</div>
          <div class='row-note'>{subtitle}</div>
        </div>
      </div>
      <div class='row-scroll'>{cards}</div>
    </div>
    """


st.markdown("<div class='sticky-search'>", unsafe_allow_html=True)
search_col1, search_col2 = st.columns([5, 1], gap="small")
with search_col1:
    search_query = st.text_input("Search", key="search_query", placeholder="Search for a movie title...", label_visibility="collapsed")
with search_col2:
    if st.button("Find"):
        query = (search_query or "").strip().lower()
        if query:
            matches = rec.df[rec.df["title"].str.contains(query, case=False, na=False)]
            if not matches.empty:
                selected = matches.iloc[0]["title"]
                history = st.session_state["history"]
                if selected not in history:
                    st.session_state["history"] = [selected] + history[:9]
                st.session_state["selected_title"] = selected
            else:
                st.warning("No match found.")

st.markdown("</div>", unsafe_allow_html=True)

selected_title = st.session_state["selected_title"]
selected_row = rec.df[rec.df["title"].str.lower() == selected_title.lower()]
if selected_row.empty:
    selected_row = rec.df.iloc[[0]]
    selected_title = selected_row.iloc[0]["title"]

selected_movie = selected_row.iloc[0].to_dict()
hero_backdrop = backdrop_url(selected_movie.get("backdrop_path")) or poster_url(selected_movie.get("poster_path"))
hero_title = selected_movie.get("title", "Shimlana")
hero_overview = selected_movie.get("overview", "A content-based movie recommender that finds titles with the same cinematic fingerprint.")
hero_stats = [f"{selected_movie.get('vote_average', 0):.1f} rating", year_of(selected_movie.get("release_date"))]

sim_scores = score_map_for(selected_title)

featured = rec.df.nlargest(10, "vote_average").to_dict(orient="records")
recent = rec.df[rec.df["release_date"].notna()].sort_values("release_date", ascending=False).head(10).to_dict(orient="records")
primary_genre = selected_movie.get("genres", "").split()[0] if selected_movie.get("genres") else "" 
if primary_genre:
    genre_list = genre_movies(primary_genre)
else:
    genre_list = rec.df.nlargest(10, "vote_average").to_dict(orient="records")

# hero section
st.markdown(
    f"""
    <section class='hero'>
      <div class='hero-bg' style='background-image:url("{hero_backdrop}");'></div>
      <div class='hero-copy'>
        <div class='hero-eyebrow'>Shimlana</div>
        <h1 class='hero-title'>{hero_title}</h1>
        <p class='hero-overview'>{hero_overview}</p>
        <div class='hero-stat'>
          <div>{hero_stats[0]}</div>
          <div>{hero_stats[1]}</div>
        </div>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='sticky-search'></div>", unsafe_allow_html=True)

# content rows
st.markdown(render_row("Top rated", "Highest rated films in the Shimlana catalogue.", featured, sim_scores), unsafe_allow_html=True)
st.markdown(render_row("Recently added", "Newest releases added to the catalogue.", recent, sim_scores), unsafe_allow_html=True)
if primary_genre:
    st.markdown(render_row(f"By genre — {primary_genre}", f"Closest matches in {primary_genre} films.", genre_list, sim_scores), unsafe_allow_html=True)

st.markdown(
    """
    <div class='app-footer'>Type the name of a film above to update the recommendations and row focus.</div>
    """,
    unsafe_allow_html=True,
)
