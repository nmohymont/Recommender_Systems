from flask import Flask, render_template, request
import pandas as pd
import json
from pathlib import Path

from loaders import load_ratings, load_movies, load_tmdb_features
from content import get_featured_movies
from recommander_test import HybridRecommender
from constants import Constant as C

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Gestion des utilisateurs enregistrés (JSON)
# ---------------------------------------------------------------------------
USERS_FILE = Path(__file__).parent / "data" / "registered_users.json"
USERS_FILE.parent.mkdir(parents=True, exist_ok=True)

def load_registered_users() -> list:
    """Charge les utilisateurs créés via l'interface (liste de dicts)."""
    if not USERS_FILE.exists():
        return []
    try:
        return json.loads(USERS_FILE.read_text())
    except Exception:
        return []

def save_registered_user(name: str, user_id: int, movie_ratings: dict):
    """Ajoute ou met à jour un utilisateur dans le fichier JSON."""
    users = load_registered_users()
    # Eviter les doublons sur user_id
    users = [u for u in users if u.get("user_id") != user_id]
    users.append({
        "user_id":  user_id,
        "name":     name,
        "ratings":  {str(k): v for k, v in movie_ratings.items()},
    })
    USERS_FILE.write_text(json.dumps(users, indent=2))

# ---------------------------------------------------------------------------
# Initialisation au démarrage
# ---------------------------------------------------------------------------
print("Training hybrid recommender...")
recommender = HybridRecommender(use_implicit=True, use_content=False)
recommender.fit()
print("Models trained successfully.")

print("Loading TMDB features...")
TMDB = load_tmdb_features()
print(f"  {len(TMDB)} movies with TMDB data loaded.")


def enrich_with_tmdb(records: list) -> list:
    for rec in records:
        mid = rec.get("movieId") or rec.get(C.ITEM_ID_COL)
        if mid and int(mid) in TMDB.index:
            row = TMDB.loc[int(mid)]
            rec["poster_url"]   = row.get("poster_url",   "")
            rec["overview"]     = row.get("overview",     "")
            rec["runtime"]      = row.get("runtime",      "")
            rec["vote_average"] = row.get("vote_average", "")
        else:
            rec["poster_url"]   = ""
            rec["overview"]     = ""
            rec["runtime"]      = ""
            rec["vote_average"] = ""
    return records



# Build movies list with poster URLs for the new user rating grid
_movies_df = load_movies()
ALL_MOVIES_JSON = []
for _, _row in _movies_df.iterrows():
    _mid = int(_row[C.ITEM_ID_COL])
    ALL_MOVIES_JSON.append({
        "movieId":    _mid,
        "title":      _row[C.LABEL_COL],
        "poster_url": TMDB.loc[_mid, "poster_url"] if _mid in TMDB.index else "",
    })


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    from content import get_available_genres
    featured = get_featured_movies(limit=None)
    records  = featured.to_dict(orient="records")
    records  = enrich_with_tmdb(records)
    genres   = get_available_genres()
    return render_template("home.html", featured_movies=records, genres=genres)


@app.route("/movie/<int:movie_id>")
def movie_detail(movie_id):
    from loaders import load_items
    items = load_items()

    if movie_id not in items.index:
        return "Movie not found", 404

    movie    = items.loc[movie_id]
    tmdb_row = TMDB.loc[movie_id] if movie_id in TMDB.index else None
    ratings  = load_ratings(surprise_format=False, use_implicit=False)
    mr       = ratings[ratings[C.ITEM_ID_COL] == movie_id][C.RATING_COL]

    movie_data = {
        "movieId":      movie_id,
        "title":        movie[C.LABEL_COL],
        "genres":       movie[C.GENRES_COL],
        "poster_url":   tmdb_row["poster_url"]   if tmdb_row is not None else "",
        "overview":     tmdb_row["overview"]     if tmdb_row is not None else "",
        "runtime":      tmdb_row["runtime"]      if tmdb_row is not None else "",
        "vote_average": tmdb_row["vote_average"] if tmdb_row is not None else "",
        "vote_count":   tmdb_row["vote_count"]   if tmdb_row is not None else "",
        "popularity":   tmdb_row["popularity"]   if tmdb_row is not None else "",
        "release_date": tmdb_row["release_date"] if tmdb_row is not None else "",
        "rating_mean":  round(float(mr.mean()), 2) if len(mr) > 0 else "",
        "rating_count": int(len(mr)),
    }

    return render_template("movie_detail.html", movie=movie_data)


@app.route("/individual", methods=["GET", "POST"])
def individual():
    recommendations = []
    selected_user   = None
    new_user_mode   = False
    rated_movies    = []

    if request.method == "POST":
        mode = request.form.get("mode", "existing")

        if mode == "existing":
            selected_user     = request.form.get("user_id")
            n_recommendations = 10
            recs = recommender.recommend(user_id=selected_user, n=n_recommendations)
            for _, row in recs.iterrows():
                mid      = int(row[C.ITEM_ID_COL])
                tmdb_row = TMDB.loc[mid] if mid in TMDB.index else None
                recommendations.append({
                    "movieId":      mid,
                    "title":        row[C.LABEL_COL],
                    "genres":       row[C.GENRES_COL],
                    "final_score":  round(row["final_score"], 2),
                    "svd_score":    round(row["svd_score"],   2),
                    "itr_score":    round(row["itr_score"],   2),
                    "explanation":  recommender.explain(selected_user, mid),
                    "poster_url":   tmdb_row["poster_url"]   if tmdb_row is not None else "",
                    "overview":     tmdb_row["overview"]     if tmdb_row is not None else "",
                    "runtime":      tmdb_row["runtime"]      if tmdb_row is not None else "",
                    "vote_average": tmdb_row["vote_average"] if tmdb_row is not None else "",
                })

        elif mode == "new_user":
            new_user_mode     = True
            n_recommendations = 10
            movie_ratings     = {}
            for key, val in request.form.items():
                if key.startswith("movie_"):
                    try:
                        mid = int(key.replace("movie_", ""))
                        movie_ratings[mid] = max(0.5, min(5.0, float(val)))
                    except (ValueError, TypeError):
                        pass

            if movie_ratings:
                recs      = recommender.recommend_new_user(movie_ratings, n=n_recommendations)
                movies_df = load_movies().set_index(C.ITEM_ID_COL)
                rated_movies = [
                    {
                        "movieId":    mid,
                        "title":      movies_df.loc[mid, C.LABEL_COL] if mid in movies_df.index else str(mid),
                        "rating":     int(r),
                        "poster_url": TMDB.loc[mid, "poster_url"] if mid in TMDB.index else ""
                    }
                    for mid, r in movie_ratings.items()
                ]
                user_name   = request.form.get("user_name", "").strip() or "Anonymous"
                # Assign a new unique user_id (max existing + 1)
                ratings_df  = load_ratings(surprise_format=False, use_implicit=True)
                existing_ids = [int(uid) for uid in ratings_df["userId"].unique() if int(uid) != -1]
                reg_ids      = [u["user_id"] for u in load_registered_users()]
                new_user_id  = max(existing_ids + reg_ids + [0]) + 1
                save_registered_user(user_name, new_user_id, movie_ratings)

                for _, row in recs.iterrows():
                    mid      = int(row[C.ITEM_ID_COL])
                    tmdb_row = TMDB.loc[mid] if mid in TMDB.index else None
                    recommendations.append({
                        "movieId":      mid,
                        "title":        row[C.LABEL_COL],
                        "genres":       row[C.GENRES_COL],
                        "final_score":  round(row["final_score"], 2),
                        "svd_score":    round(row["svd_score"],   2),
                        "itr_score":    round(row["itr_score"],   2),
                        "explanation":  "Based on your content profile.",
                        "poster_url":   tmdb_row["poster_url"]   if tmdb_row is not None else "",
                        "overview":     tmdb_row["overview"]     if tmdb_row is not None else "",
                        "runtime":      tmdb_row["runtime"]      if tmdb_row is not None else "",
                        "vote_average": tmdb_row["vote_average"] if tmdb_row is not None else "",
                    })

    # Build users list AFTER potential save so new user appears immediately
    ratings_for_users = load_ratings(surprise_format=False, use_implicit=True)
    registered        = {u["user_id"]: u["name"] for u in load_registered_users()}
    dataset_ids       = sorted([int(uid) for uid in ratings_for_users["userId"].unique() if int(uid) != -1])
    max_id            = max(dataset_ids) if dataset_ids else 0
    users             = [{"user_id": uid, "name": registered.get(uid, f"User {uid}")} for uid in dataset_ids]
    for uid, name in registered.items():
        if uid > max_id:
            users.append({"user_id": uid, "name": name})
    users = sorted(users, key=lambda u: u["user_id"])

    return render_template(
        "individual.html",
        users=users,
        selected_user=selected_user,
        recommendations=recommendations,
        new_user_mode=new_user_mode,
        rated_movies=rated_movies,
        all_movies=ALL_MOVIES_JSON
    )


@app.route("/group", methods=["GET", "POST"])
def group():
    group_recommendations = []
    participants          = []
    strategy              = "average"
    n_participants        = 2

    if request.method == "POST":
        n_participants = int(request.form.get("n_participants", 2))
        strategy       = request.form.get("strategy", "average")
        n_recs         = int(request.form.get("n_recommendations", 8))
        movies_df      = load_movies().set_index(C.ITEM_ID_COL)

        for i in range(n_participants):
            name     = request.form.get(f"participant_name_{i}", f"Person {i+1}")
            user_id  = request.form.get(f"participant_user_id_{i}")
            liked    = request.form.getlist(f"p{i}_liked")
            wishlist = request.form.getlist(f"p{i}_wishlist")
            participants.append({
                "name":    name,
                "user_id": int(user_id) if user_id else None,
                "liked": [
                    {"id": int(m), "title": movies_df.loc[int(m), C.LABEL_COL]
                     if int(m) in movies_df.index else str(m)}
                    for m in liked if m
                ],
                "wishlist": [
                    {"id": int(m), "title": movies_df.loc[int(m), C.LABEL_COL]
                     if int(m) in movies_df.index else str(m)}
                    for m in wishlist if m
                ],
            })

        if strategy == "surprise":
            group_recs = recommender.recommend_group_surprise(
                participants=participants,
                n=n_recs,
            )
        else:
            group_recs = recommender.recommend_group_by_movies(
                participants=participants,
                n=n_recs,
                strategy=strategy,
            )
        group_recommendations = group_recs.to_dict(orient="records")
        for rec in group_recommendations:
            rec["group_score"] = round(rec["group_score"], 2)
            mid = rec.get(C.ITEM_ID_COL) or rec.get("movieId")
            if mid and int(mid) in TMDB.index:
                rec["poster_url"]   = TMDB.loc[int(mid), "poster_url"]
                rec["overview"]     = TMDB.loc[int(mid), "overview"]
                rec["vote_average"] = TMDB.loc[int(mid), "vote_average"]
            else:
                rec["poster_url"]   = ""
                rec["overview"]     = ""
                rec["vote_average"] = ""

    ratings    = load_ratings(surprise_format=False, use_implicit=True)
    registered = {u["user_id"]: u["name"] for u in load_registered_users()}
    # Users du dataset enrichis avec les noms enregistrés
    dataset_users = [
        {"id": int(uid), "name": registered.get(int(uid), f"User {int(uid)}")}
        for uid in sorted(ratings["userId"].astype(int).unique())
        if uid != -1
    ]
    for u in dataset_users:
        if u["id"] in registered:
            u["name"] = registered[u["id"]]
    # Ajouter les utilisateurs enregistrés qui ne sont pas dans le dataset
    existing_ids = {u["id"] for u in dataset_users}
    for uid, name in registered.items():
        if uid not in existing_ids:
            dataset_users.append({"id": uid, "name": name})
    all_users = sorted(dataset_users, key=lambda u: u["id"])
    return render_template(
        "group.html",
        group_recommendations=group_recommendations,
        participants=participants,
        strategy=strategy,
        n_participants=n_participants,
        all_movies=ALL_MOVIES_JSON,
        all_users=all_users,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)