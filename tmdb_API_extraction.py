import time
import requests
import pandas as pd
from constants import Constant as C

TMDB_API_KEY = "KEY" # Replace with your actual TMDB API key
TMDB_BASE_URL = "https://api.themoviedb.org/3/movie"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


def fetch_tmdb_movie(tmdb_id):
    if pd.isna(tmdb_id):
        return None

    url = f"{TMDB_BASE_URL}/{int(tmdb_id)}"
    params = {
        "api_key": TMDB_API_KEY,
        "append_to_response": "credits,watch/providers"}

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None
        return response.json()
    except requests.RequestException:
        return None


def extract_tmdb_features(movie_id, tmdb_id, data):
    if data is None:
        return {
            "movieId": movie_id,
            "tmdbId": tmdb_id,
            "tmdb_found": 0}

    poster_path = data.get("poster_path")
    overview = data.get("overview", "")

    return {
        "movieId": movie_id,
        "title": data.get("title"),
        "overview": data.get("overview"),
        "release_date": data.get("release_date"),
        "runtime": data.get("runtime"),
        "vote_average": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
        "popularity": data.get("popularity"),
        "poster_path": poster_path,
        "poster_url": f"{IMAGE_BASE_URL}{poster_path}" if poster_path else ""}


def build_tmdb_features(limit=None):
    links = pd.read_csv(C.CONTENT_PATH / "links.csv")
    if limit is not None:
        links = links.head(limit)
    rows = []
    for i, row in links.iterrows():
        movie_id = row["movieId"]
        tmdb_id = row["tmdbId"]
        print(f"{i + 1}/{len(links)} - movieId={movie_id}, tmdbId={tmdb_id}")
        data = fetch_tmdb_movie(tmdb_id)
        features = extract_tmdb_features(movie_id, tmdb_id, data)
        rows.append(features)
        time.sleep(0.05)
    df = pd.DataFrame(rows)
    output_path = C.CONTENT_PATH / "tmdb_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved here: {output_path}")


if __name__ == "__main__":
    build_tmdb_features()