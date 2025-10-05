"""
movie_recommender.py

Single-file Streamlit app that builds a hybrid movie recommender using MovieLens ml-latest-small.
Run:
    pip install -r requirements.txt
    streamlit run movie_recommender.py

requirements.txt:
    streamlit
    pandas
    numpy
    scikit-learn
    scipy
    joblib
"""

import os
import zipfile
import urllib.request
from io import BytesIO
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from scipy.sparse import csr_matrix
import joblib

# ---------- CONFIG ----------
# DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
# CACHE_DIR = "ml_cache"
# RANDOM_STATE = 42

# ---------- UTIL / DOWNLOAD ----------
# def ensure_data():
#     os.makedirs(CACHE_DIR, exist_ok=True)
#     zip_path = os.path.join(CACHE_DIR, "ml-latest-small.zip")
#     extract_dir = os.path.join(CACHE_DIR, "ml-latest-small")
#     if not os.path.exists(extract_dir):
#         st.info("Downloading MovieLens small dataset (~1 MB)...")
#         urllib.request.urlretrieve(DATA_URL, zip_path)
#         with zipfile.ZipFile(zip_path, "r") as z:
#             z.extractall(extract_dir)
#     return extract_dir

# @st.cache_data(show_spinner=False)
# def load_ratings_movies():
#     data_dir = ensure_data()
#     ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
#     movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
#     # keep a small safe sample? We use full ml-latest-small
#     return ratings, movies

# Replace the previous ensure_data() and load_ratings_movies() with this:

import urllib.request
from io import BytesIO
import zipfile

DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

@st.cache_data(show_spinner=False)
def load_ratings_movies():
    """
    Downloads the MovieLens small zip into memory (if not cached by Streamlit),
    then reads 'ratings.csv' and 'movies.csv' directly from the zipfile into pandas.
    This avoids extraction issues on Windows and missing-file errors.
    """
    try:
        # Download zip into memory
        with urllib.request.urlopen(DATA_URL) as resp:
            zip_bytes = resp.read()
    except Exception as e:
        st.error(f"Could not download MovieLens dataset: {e}")
        raise

    try:
        z = zipfile.ZipFile(BytesIO(zip_bytes))
    except zipfile.BadZipFile as e:
        st.error("Downloaded file is not a valid zip archive.")
        raise

    # find the path inside the zip that contains ratings.csv and movies.csv
    # (some mirrors may include a top folder 'ml-latest-small/')
    names = z.namelist()
    # helper to find the first matching filename ignoring directories
    def find_in_zip(filename):
        for n in names:
            if n.lower().endswith("/" + filename) or n.lower() == filename:
                return n
        return None

    ratings_name = find_in_zip("ratings.csv")
    movies_name = find_in_zip("movies.csv")

    if ratings_name is None or movies_name is None:
        st.error("ratings.csv or movies.csv not found inside the downloaded zip.")
        raise FileNotFoundError("Missing files in MovieLens zip")

    # read CSVs into pandas directly from zipfile (as bytes)
    with z.open(ratings_name) as f:
        ratings = pd.read_csv(f)
    with z.open(movies_name) as f:
        movies = pd.read_csv(f)

    return ratings, movies


# ---------- PREPROCESS ----------
@st.cache_data(show_spinner=False)
def preprocess(ratings, movies, min_ratings_per_movie=5, min_ratings_per_user=5):
    # Basic filter to remove very rare movies/users (stability)
    movie_counts = ratings['movieId'].value_counts()
    keep_movies = movie_counts[movie_counts >= min_ratings_per_movie].index
    user_counts = ratings['userId'].value_counts()
    keep_users = user_counts[user_counts >= min_ratings_per_user].index

    ratings_f = ratings[ratings['movieId'].isin(keep_movies) & ratings['userId'].isin(keep_users)].copy()
    # Build id maps
    user_ids = sorted(ratings_f['userId'].unique())
    movie_ids = sorted(ratings_f['movieId'].unique())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    movie_to_idx = {m: i for i, m in enumerate(movie_ids)}
    idx_to_movie = {i: m for m, i in movie_to_idx.items()}

    rows = ratings_f['userId'].map(user_to_idx)
    cols = ratings_f['movieId'].map(movie_to_idx)
    data = ratings_f['rating'].astype(np.float32)

    user_item = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
    movies_sub = movies[movies['movieId'].isin(movie_ids)].copy()
    # reorder movies_sub to match movie_ids order
    movies_sub['movie_idx'] = movies_sub['movieId'].map(movie_to_idx)
    movies_sub = movies_sub.set_index('movie_idx').loc[range(len(movie_ids))].reset_index(drop=True)

    return {
        'user_item': user_item,
        'user_ids': user_ids,
        'movie_ids': movie_ids,
        'user_to_idx': user_to_idx,
        'movie_to_idx': movie_to_idx,
        'idx_to_movie': idx_to_movie,
        'movies_sub': movies_sub
    }

# ---------- MODEL BUILD ----------
# @st.cache_resource(show_spinner=False)
# def build_models(user_item, n_components=50):
#     """
#     1) TruncatedSVD on user-item (to produce item latent features)
#     2) cosine similarity between item latent vectors
#     """
#     # Use SVD on the (users x items) matrix, then use item factors
#     svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
#     U = svd.fit_transform(user_item)            # users x n_components
#     Sigma = svd.singular_values_
#     Vt = svd.components_                        # n_components x items
#     item_factors = Vt.T                         # items x n_components

#     # Normalize item vectors for cosine computations
#     item_norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
#     item_factors_normed = item_factors / np.maximum(item_norms, 1e-9)

#     # Precompute cosine similarity (may be moderately large: items x items)
#     item_sim = cosine_similarity(item_factors_normed)  # dense; ok for ml-latest-small
#     return {
#         'svd': svd,
#         'user_factors': U,
#         'item_factors': item_factors,
#         'item_sim': item_sim
#     }

def build_models(user_item, n_components=50):
    """
    1) TruncatedSVD on user-item (to produce item latent features)
    2) cosine similarity between item latent vectors
    """
    # Use SVD on the (users x items) matrix, then use item factors
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(user_item)            # users x n_components
    Sigma = svd.singular_values_
    Vt = svd.components_                        # n_components x items
    item_factors = Vt.T                         # items x n_components

    # Normalize item vectors for cosine computations
    item_norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    item_factors_normed = item_factors / np.maximum(item_norms, 1e-9)

    # Precompute cosine similarity (may be moderately large: items x items)
    item_sim = cosine_similarity(item_factors_normed)  # dense; ok for ml-latest-small
    return {
        'svd': svd,
        'user_factors': U,
        'item_factors': item_factors,
        'item_sim': item_sim
    }

# ---------- CONTENT-BASED ----------
@st.cache_resource(show_spinner=False)
def build_content_model(movies_sub):
    # create a single text field: title + genres
    texts = (movies_sub['title'].fillna('') + ' ' + movies_sub['genres'].fillna('')).values
    vect = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vect.fit_transform(texts)
    # item-item content similarity
    content_sim = cosine_similarity(X)
    return {
        'tfidf': vect,
        'content_sim': content_sim
    }

# ---------- RECOMMENDATION LOGIC ----------
def recommend_for_user(user_id, data_objs, models, content_mod, top_n=10, weights=(0.6, 0.3, 0.1)):
    """
    weights: (mf_weight, item_sim_weight, content_weight)
    - MF (matrix-factorization predicted scores)
    - item similarity from user's liked items
    - content-based similarity
    """
    user_to_idx = data_objs['user_to_idx']
    movie_to_idx = data_objs['movie_to_idx']
    idx_to_movie = data_objs['idx_to_movie']
    user_item = data_objs['user_item']
    movies_sub = data_objs['movies_sub']

    if user_id not in user_to_idx:
        return None, f"User id {user_id} not found (dataset user count: {len(user_to_idx)})"

    uidx = user_to_idx[user_id]

    # MF score: reconstruct approximate user ratings using U * Vt
    user_vector = models['user_factors'][uidx]                        # size n_components
    mf_scores = models['item_factors'].dot(user_vector)              # items

    # item-sim: aggregate similarities from items user rated highly
    user_row = user_item.getrow(uidx).toarray().ravel()
    rated_idx = np.where(user_row > 0)[0]
    if len(rated_idx) == 0:
        item_sim_scores = np.zeros_like(mf_scores)
    else:
        # weight by user's normalized rating for those items
        ratings_for_rated = user_row[rated_idx]
        ratings_norm = (ratings_for_rated - ratings_for_rated.mean()) if ratings_for_rated.size>0 else ratings_for_rated
        sims = models['item_sim'][rated_idx]                          # (k x items)
        weighted = sims.T.dot(ratings_norm)
        item_sim_scores = weighted / (np.linalg.norm(ratings_norm)+1e-9)

    # content-based: sum similarities from the same rated movies (using content_sim)
    if hasattr(content_mod, 'get') or isinstance(content_mod, dict):
        content_sim = content_mod['content_sim']
    else:
        content_sim = content_mod
    if len(rated_idx) == 0:
        content_scores = np.zeros_like(mf_scores)
    else:
        content_scores = content_sim[rated_idx].sum(axis=0)
        content_scores = content_scores / (np.linalg.norm(content_scores)+1e-9)

    # Combine
    mf_w, sim_w, cont_w = weights
    raw_score = mf_w * normalize(mf_scores) + sim_w * normalize(item_sim_scores) + cont_w * normalize(content_scores)

    # exclude already rated movies
    already = set(rated_idx)
    candidates = [(i, raw_score[i]) for i in range(len(raw_score)) if i not in already]
    candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_n]
    results = []
    for idx, score in candidates_sorted:
        mid = idx_to_movie[idx]
        row = movies_sub.iloc[idx]
        results.append({'movieId': int(mid), 'title': row['title'], 'score': float(score)})
    return results, None

def recommend_for_movie_titles(liked_titles, data_objs, models, content_mod, top_n=10, weights=(0.2,0.6,0.2)):
    """
    Cold-start style: user provides liked movie titles; we find closest items via item_sim and content.
    """
    movies_sub = data_objs['movies_sub']
    title_to_idx = build_title_index(movies_sub)
    found_idxs = []
    for t in liked_titles:
        t_lower = t.strip().lower()
        if t_lower in title_to_idx:
            found_idxs.append(title_to_idx[t_lower])
        else:
            # try contains match
            matches = movies_sub[movies_sub['title'].str.lower().str.contains(t_lower, na=False)]
            if not matches.empty:
                found_idxs.append(matches.index[0])

    if not found_idxs:
        return None, "No liked movies matched titles in dataset."

    # item-sim aggregate
    item_sim = models['item_sim']
    sim_scores = item_sim[found_idxs].sum(axis=0)
    sim_scores = normalize(sim_scores)

    # content-based
    content_scores = content_mod['content_sim'][found_idxs].sum(axis=0)
    content_scores = normalize(content_scores)

    # MF: use average of item latent vectors from liked items, score by dot product
    item_factors = models['item_factors']
    avg_vec = item_factors[found_idxs].mean(axis=0)
    mf_scores = item_factors.dot(avg_vec)
    mf_scores = normalize(mf_scores)

    mf_w, sim_w, cont_w = weights
    raw_score = mf_w*mf_scores + sim_w*sim_scores + cont_w*content_scores

    # exclude the liked items themselves
    liked_set = set(found_idxs)
    candidates = [(i, raw_score[i]) for i in range(len(raw_score)) if i not in liked_set]
    candidates_sorted = sorted(candidates, key=lambda x:x[1], reverse=True)[:top_n]

    results = []
    for idx, score in candidates_sorted:
        mid = data_objs['idx_to_movie'][idx]
        row = movies_sub.iloc[idx]
        results.append({'movieId': int(mid), 'title': row['title'], 'score': float(score)})
    return results, None

# ---------- HELPERS ----------
def normalize(arr):
    arr = np.array(arr, dtype=float)
    if arr.max() - arr.min() <= 1e-9:
        return np.zeros_like(arr)
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr

def build_title_index(movies_sub):
    # lowercase title -> idx (first occurrence)
    d = {}
    for i, t in enumerate(movies_sub['title'].fillna('')):
        d[t.strip().lower()] = i
    return d

# ---------- MAIN: Streamlit UI ----------
def main():
    st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
    st.title("🎬 Hybrid Movie Recommender (Streamlit)")

    with st.spinner("Loading dataset..."):
        ratings, movies = load_ratings_movies()
        data_objs = preprocess(ratings, movies)

    st.sidebar.header("Model build & settings")
    n_components = st.sidebar.slider("SVD components (latent dim)", 20, 200, 50)
    mf_weight = st.sidebar.slider("Matrix-factorization weight", 0.0, 1.0, 0.6)
    item_sim_weight = st.sidebar.slider("Item-sim weight", 0.0, 1.0, 0.3)
    content_weight = st.sidebar.slider("Content weight", 0.0, 1.0, 0.1)
    # normalize weights
    sumw = mf_weight + item_sim_weight + content_weight
    if sumw <= 0:
        mf_weight, item_sim_weight, content_weight = 0.6, 0.3, 0.1
    else:
        mf_weight, item_sim_weight, content_weight = mf_weight/sumw, item_sim_weight/sumw, content_weight/sumw

    st.sidebar.write(f"Normalized weights → MF: {mf_weight:.2f}, ItemSim: {item_sim_weight:.2f}, Content: {content_weight:.2f}")
    build_button = st.sidebar.button("(Re)build models")

    # Build models (cached by n_components)
    # @st.cache_resource(show_spinner=False)
    # def get_models_cached(user_item, n_comp):
    #     return build_models(user_item, n_components=n_comp)
    # Build models (cached by n_components)
    
    
    # @st.cache_resource(show_spinner=False)
    # def get_models_cached(_user_item, n_comp):
    #     # note: use the variable name _user_item (leading underscore tells Streamlit not to hash it)
    #     return build_models(_user_item, n_components=n_comp)
    # @st.cache_resource(show_spinner=False, hash_funcs={csr_matrix: lambda _: None})
    # def get_models_cached(user_item, n_comp):
    #     return build_models(user_item, n_components=n_comp)
    
    # Cache the wrapper. Use a leading-underscore for the csr_matrix parameter
# so Streamlit does not try to hash it.
    @st.cache_resource(show_spinner=False)
    def get_models_cached(_user_item, n_comp):
    # Note: call build_models with the unhashed _user_item variable
        return build_models(_user_item, n_components=n_comp)


    # call it the same way you already do:
    models = get_models_cached(data_objs['user_item'], n_components)


    models = get_models_cached(data_objs['user_item'], n_components)
    content_mod = build_content_model(data_objs['movies_sub'])

    st.markdown("### How to use")
    st.markdown("- Get recommendations for an existing **user id** from the dataset (choose below).")
    st.markdown("- Or enter one or more **movie titles** you like (partial match works) to get similar suggestions (cold-start).")

    col1, col2 = st.columns([1,2])

    with col1:
        st.subheader("By user id (existing in dataset)")
        uid = st.number_input("User id", min_value=int(min(data_objs['user_ids'])), max_value=int(max(data_objs['user_ids'])), value=int(data_objs['user_ids'][0]))
        topn_user = st.slider("Number of recommendations", 5, 30, 10)
        if st.button("Recommend for user"):
            with st.spinner("Computing recommendations..."):
                results, err = recommend_for_user(int(uid), data_objs, models, content_mod, top_n=topn_user, weights=(mf_weight, item_sim_weight, content_weight))
                if err:
                    st.error(err)
                else:
                    df = pd.DataFrame(results)
                    st.table(df[['title','score']])

    with col2:
        st.subheader("By liked movies (cold-start)")
        liked = st.text_input("Comma-separated liked movie titles (partial ok)", value="Toy Story")
        topn_movie = st.slider("Number of recommendations", 5, 30, 10, key="topn_movie")
        if st.button("Recommend like these"):
            liked_list = [s.strip() for s in liked.split(",") if s.strip()]
            if not liked_list:
                st.error("Enter at least one movie title.")
            else:
                with st.spinner("Computing recommendations..."):
                    results, err = recommend_for_movie_titles(liked_list, data_objs, models, content_mod, top_n=topn_movie, weights=(0.2, 0.6, 0.2))
                    if err:
                        st.error(err)
                    else:
                        df = pd.DataFrame(results)
                        st.table(df[['title','score']])

    st.markdown("---")
    st.subheader("Data sample / info")
    st.write(f"Dataset: {len(ratings)} ratings, {len(movies)} movie metadata rows.")
    st.write("Note: for stability we filter users/movies with very few ratings (configurable in code).")
    if st.checkbox("Show first movies rows"):
        st.dataframe(data_objs['movies_sub'].head(20))

    st.markdown("### Save / Load model")
    if st.button("Save models to disk (cache/ml_models.joblib)"):
        os.makedirs("cache", exist_ok=True)
        joblib.dump({'models': models, 'content_mod': content_mod, 'data_objs': data_objs}, "cache/ml_models.joblib")
        st.success("Saved to cache/ml_models.joblib")

    st.caption("Tip: deploy this file to Streamlit Cloud (https://streamlit.io) or wrap into a Flask API if you prefer REST endpoints.")

if __name__ == "__main__":
    main()
