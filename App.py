"""
app.py
======
Interface Flask de MovieMatch.

Routes
------
/               → home       : films populaires
/individual     → individual : recommandations personnalisées
/group          → group      : Netflix Party
/evaluation     → evaluation : tableau de bord des modèles
"""

from flask import Flask, render_template, request
import pandas as pd
import glob

from loaders import load_ratings, load_movies, load_tmdb_features
from content import get_featured_movies
from recommander_test import HybridRecommender
from constants import Constant as C

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Initialisation au démarrage
# ---------------------------------------------------------------------------
print("Training hybrid recommender...")
recommender = HybridRecommender(use_implicit=True)
recommender.fit()
print("Models trained successfully.")

# Charger les features TMDB (posters + overviews)
print("Loading TMDB features...")
TMDB = load_tmdb_features()
print(f"  {len(TMDB)} movies with TMDB data loaded.")


def enrich_with_tmdb(records: list) -> list:
    """
    Ajoute poster_url et overview à une liste de dicts movies.
    Chaque dict doit avoir une clé 'movieId' ou C.ITEM_ID_COL.
    """
    for rec in records:
        mid = rec.get("movieId") or rec.get(C.ITEM_ID_COL)
        if mid and int(mid) in TMDB.index:
            row = TMDB.loc[int(mid)]
            rec["poster_url"] = row.get("poster_url", "")
            rec["overview"]   = row.get("overview",   "")
            rec["runtime"]    = row.get("runtime",    "")
            rec["vote_average"] = row.get("vote_average", "")
        else:
            rec["poster_url"]   = ""
            rec["overview"]     = ""
            rec["runtime"]      = ""
            rec["vote_average"] = ""
    return records


# ---------------------------------------------------------------------------
# Résultats d'évaluation
# ---------------------------------------------------------------------------
def load_latest_report():
    files = sorted(glob.glob(str(C.EVALUATION_PATH / "report_*.csv")))
    if not files:
        return []
    df = pd.read_csv(files[-1])

    family_map = {
        "baseline_mean":              ("Baseline",      "baseline"),
        "user_based_jaccard":         ("User-based",    "user"),
        "user_based_pearson_baseline":("User-based",    "user"),
        "item_based_pearson_baseline":("Item-based",    "item"),
        "content_based_ridge_v3":     ("Content-based", "content"),
        "svd":                        ("Latent Factor", "latent"),
    }

    p10_col   = "precision@10" if "precision@10" in df.columns else None
    r10_col   = "recall@10"    if "recall@10"    in df.columns else None
    best_rmse = df["rmse"].min()
    best_mae  = df["mae"].min()
    best_p10  = df[p10_col].max() if p10_col else None
    best_r10  = df[r10_col].max() if r10_col else None
    best_cov  = df["coverage"].max()  if "coverage"  in df.columns else None
    best_div  = df["diversity"].max() if "diversity" in df.columns else None

    rows = []
    for _, row in df.iterrows():
        family, family_class = family_map.get(row["model"], ("Other", "baseline"))
        rows.append({
            "model":        row["model"],
            "family":       family,
            "family_class": family_class,
            "rmse":         round(row["rmse"], 4),
            "mae":          round(row["mae"],  4),
            "precision":    round(row[p10_col], 4) if p10_col else "—",
            "recall":       round(row[r10_col], 4) if r10_col else "—",
            "coverage":     round(row["coverage"],  4) if "coverage"  in df.columns else "—",
            "diversity":    round(row["diversity"], 4) if "diversity" in df.columns else "—",
            "best_rmse":    abs(row["rmse"] - best_rmse) < 1e-6,
            "best_mae":     abs(row["mae"]  - best_mae)  < 1e-6,
            "best_p10":     bool(p10_col and abs(row[p10_col] - best_p10) < 1e-6),
            "best_r10":     bool(r10_col and abs(row[r10_col] - best_r10) < 1e-6),
            "best_cov":     bool(best_cov and abs(row.get("coverage", 0) - best_cov) < 1e-6),
            "best_div":     bool(best_div and abs(row.get("diversity", 0) - best_div) < 1e-6),
        })
    return rows


EVAL_RESULTS    = load_latest_report()
ALL_MOVIES_JSON = load_movies()[
    [C.ITEM_ID_COL, C.LABEL_COL]
].rename(columns={C.ITEM_ID_COL: "movieId", C.LABEL_COL: "title"}).to_dict(orient="records")


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
    return render_template(
        "home.html",
        featured_movies=records,
        genres=genres
    )


@app.route("/individual", methods=["GET", "POST"])
def individual():
    ratings         = load_ratings(surprise_format=False, use_implicit=True)
    users           = sorted(ratings["userId"].astype(int).unique())
    recommendations = []
    selected_user   = None
    new_user_mode   = False
    rated_movies    = []

    if request.method == "POST":
        mode = request.form.get("mode", "existing")

        if mode == "existing":
            selected_user     = request.form.get("user_id")
            n_recommendations = int(request.form.get("n_recommendations", 10))
            recs = recommender.recommend(user_id=selected_user, n=n_recommendations)
            for _, row in recs.iterrows():
                mid = int(row[C.ITEM_ID_COL])
                tmdb_row = TMDB.loc[mid] if mid in TMDB.index else None
                recommendations.append({
                    "movieId":     mid,
                    "title":       row[C.LABEL_COL],
                    "genres":      row[C.GENRES_COL],
                    "final_score": round(row["final_score"], 2),
                    "svd_score":   round(row["svd_score"],   2),
                    "item_score":  round(row["item_score"],  2),
                    "explanation": recommender.explain(selected_user, mid),
                    "poster_url":  tmdb_row["poster_url"] if tmdb_row is not None else "",
                    "overview":    tmdb_row["overview"]   if tmdb_row is not None else "",
                    "runtime":     tmdb_row["runtime"]    if tmdb_row is not None else "",
                    "vote_average":tmdb_row["vote_average"] if tmdb_row is not None else "",
                })

        elif mode == "new_user":
            new_user_mode     = True
            n_recommendations = int(request.form.get("n_recommendations", 10))
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
                        "movieId":   mid,
                        "title":     movies_df.loc[mid, C.LABEL_COL] if mid in movies_df.index else str(mid),
                        "rating":    int(r),
                        "poster_url": TMDB.loc[mid, "poster_url"] if mid in TMDB.index else ""
                    }
                    for mid, r in movie_ratings.items()
                ]
                for _, row in recs.iterrows():
                    mid = int(row[C.ITEM_ID_COL])
                    tmdb_row = TMDB.loc[mid] if mid in TMDB.index else None
                    recommendations.append({
                        "movieId":     mid,
                        "title":       row[C.LABEL_COL],
                        "genres":      row[C.GENRES_COL],
                        "final_score": round(row["final_score"], 2),
                        "svd_score":   round(row["svd_score"],   2),
                        "item_score":  round(row["item_score"],  2),
                        "explanation": "Based on your content profile.",
                        "poster_url":  tmdb_row["poster_url"] if tmdb_row is not None else "",
                        "overview":    tmdb_row["overview"]   if tmdb_row is not None else "",
                        "runtime":     tmdb_row["runtime"]    if tmdb_row is not None else "",
                        "vote_average":tmdb_row["vote_average"] if tmdb_row is not None else "",
                    })

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
            liked    = request.form.getlist(f"p{i}_liked")
            wishlist = request.form.getlist(f"p{i}_wishlist")
            participants.append({
                "name": name,
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

        group_recs = recommender.recommend_group_by_movies(
            participants=participants,
            n=n_recs,
            strategy=strategy
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

    return render_template(
        "group.html",
        group_recommendations=group_recommendations,
        participants=participants,
        strategy=strategy,
        n_participants=n_participants,
        all_movies=ALL_MOVIES_JSON
    )


@app.route("/evaluation")
def evaluation():
    return render_template(
        "evaluation.html",
        results=EVAL_RESULTS,
        model_info=recommender.get_model_info(),
        weights=recommender.get_weights()
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)