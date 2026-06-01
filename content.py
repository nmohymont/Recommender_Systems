import re
import pandas as pd

import constants
from loaders import load_movies, load_ratings

C = constants.Constant


def extract_year(title):
    match = re.search(r"\((\d{4})\)", str(title))
    if match:
        return int(match.group(1))
    return None


def get_movies_with_years():
    movies = load_movies().copy()
    movies["year"] = movies[C.LABEL_COL].apply(extract_year)
    return movies


def get_available_genres():
    movies = load_movies()
    genres = set()

    for genre_list in movies[C.GENRES_COL].dropna():
        for genre in str(genre_list).split("|"):
            if genre and genre != "(no genres listed)":
                genres.add(genre)

    return sorted(genres)


def get_featured_movies(limit=20):
    ratings = load_ratings(use_implicit=False)
    movies = load_movies()

    stats = (
        ratings
        .groupby(C.ITEM_ID_COL)
        .agg(
            rating_count=(C.RATING_COL, "count"),
            rating_mean=(C.RATING_COL, "mean")
        )
        .reset_index()
    )

    featured = stats.merge(movies, on=C.ITEM_ID_COL, how="left")

    global_mean = ratings[C.RATING_COL].mean()
    min_votes = 50

    featured["weighted_score"] = (
        (featured["rating_count"] / (featured["rating_count"] + min_votes)) * featured["rating_mean"]
        + (min_votes / (featured["rating_count"] + min_votes)) * global_mean
    )

    featured = featured.sort_values(
        by=["weighted_score", "rating_count"],
        ascending=[False, False]
    )

    return featured.head(limit)


def filter_movies(genre=None, year_min=None, year_max=None):
    movies = get_movies_with_years()

    if genre and genre != "All":
        movies = movies[movies[C.GENRES_COL].str.contains(genre, na=False)]

    if year_min:
        movies = movies[movies["year"] >= int(year_min)]

    if year_max:
        movies = movies[movies["year"] <= int(year_max)]

    return movies