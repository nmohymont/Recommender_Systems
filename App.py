from flask import Flask, render_template, request

from loaders import load_ratings
from content import get_featured_movies
from recommander_test import HybridRecommender
from constants import Constant as C


app = Flask(__name__)

# ---------------------------------------------------------------------------
# Initialisation des modèles au démarrage (offline)
# ---------------------------------------------------------------------------
print("Training hybrid recommender...")
recommender = HybridRecommender(use_implicit=True)
recommender.fit()
print("Models trained successfully.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    featured_movies = get_featured_movies(limit=12)
    return render_template(
        "home.html",
        featured_movies=featured_movies.to_dict(orient="records")
    )


@app.route("/individual", methods=["GET", "POST"])
def individual():
    ratings = load_ratings(surprise_format=False, use_implicit=True)
    users   = sorted(ratings["userId"].astype(int).unique())

    recommendations = []
    selected_user   = None

    if request.method == "POST":
        selected_user     = request.form.get("user_id")
        n_recommendations = int(request.form.get("n_recommendations", 10))

        recs = recommender.recommend(
            user_id=selected_user,
            n=n_recommendations
        )

        for _, row in recs.iterrows():
            explanation = recommender.explain(
                selected_user,
                row[C.ITEM_ID_COL]
            )

            recommendations.append({
                "title":               row[C.LABEL_COL],
                "genres":              row[C.GENRES_COL],
                "final_score":         round(row["final_score"],         2),
                "svd_score":           round(row["svd_score"],           2),
                "user_based_score":    round(row["user_based_score"],    2),
                "item_based_score":    round(row["item_based_score"],    2),
                "content_based_score": round(row["content_based_score"], 2),
                "explanation":         explanation
            })

    return render_template(
        "individual.html",
        users=users,
        selected_user=selected_user,
        recommendations=recommendations
    )


@app.route("/group", methods=["GET", "POST"])
def group():
    ratings = load_ratings(surprise_format=False, use_implicit=True)
    users   = sorted(ratings["userId"].astype(int).unique())

    recommendations = []
    selected_users  = []
    strategy        = "average"

    if request.method == "POST":
        selected_users    = request.form.getlist("user_ids")
        n_recommendations = int(request.form.get("n_recommendations", 10))
        strategy          = request.form.get("strategy", "average")

        recs = recommender.group_recommend(
            user_ids=selected_users,
            n=n_recommendations,
            strategy=strategy
        )

        recommendations = recs.to_dict(orient="records")
        for rec in recommendations:
            rec["group_score"] = round(rec["group_score"], 2)

    return render_template(
        "group.html",
        users=users,
        selected_users=selected_users,
        recommendations=recommendations,
        strategy=strategy
    )


@app.route("/evaluation")
def evaluation():
    model_info = recommender.get_model_info()
    weights    = recommender.get_weights()
    return render_template(
        "evaluation.html",
        model_info=model_info,
        weights=weights
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )