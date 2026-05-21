# standard library imports
from collections import defaultdict

# third parties imports
import numpy as np
import random as rd
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD
from surprise.prediction_algorithms.predictions import PredictionImpossible
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# local imports
from constants import Constant as C
from loaders import load_items


def get_top_n(predictions, n):
    """Return the top-N recommendation for each user from a set of predictions.
    Source: inspired by https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    and modified by cvandekerckh for random tie breaking

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    rd.seed(0)

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        rd.shuffle(user_ratings)
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# First algorithm
class ModelBaseline1(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def estimate(self, u, i):
        return 2


# Second algorithm
class ModelBaseline2(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        rd.seed(0)

    def estimate(self, u, i):
        return rd.uniform(self.trainset.rating_scale[0], self.trainset.rating_scale[1])


# Third algorithm
class ModelBaseline3(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

        return self

    def estimate(self, u, i):
        return self.the_mean


# Fourth Model
class ModelBaseline4(SVD):
    def __init__(self, random_state=1):
        SVD.__init__(self, n_factors=100, random_state=random_state)



# Content-based model
class ContentBased(AlgoBase):
    def __init__(self, features_method, regressor_method):
        AlgoBase.__init__(self)
        self.regressor_method = regressor_method
        self.content_features = self.create_content_features(features_method)

    def create_content_features(self, features_method):
        """Content Analyzer"""
        df_items = load_items()
        if features_method is None:
            df_features = None
            
        elif features_method == "title_length": # a naive method that creates only 1 feature based on title length
            df_features = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title')

        elif features_method == "V1":
            df_features = pd.DataFrame(index=df_items.index)

            df_features["year"] = df_items[C.LABEL_COL].str.extract(r"\((\d{4})\)").astype(float)
            df_features["year"] = df_features["year"].fillna(df_features["year"].median())

            genres_dummies = df_items[C.GENRES_COL].str.get_dummies(sep="|")
            genres_dummies.index = df_items.index

            if "(no genres listed)" in genres_dummies.columns:
                genres_dummies = genres_dummies.drop(columns=["(no genres listed)"])

            df_features = df_features.join(genres_dummies)

            scaler = MinMaxScaler()
            df_features[df_features.columns] = scaler.fit_transform(df_features)

        elif features_method == "V2":
            df_features = pd.DataFrame(index=df_items.index)

            # 1. Release year
            df_features["year"] = df_items[C.LABEL_COL].str.extract(r"\((\d{4})\)").astype(float)  # feature = year extracted from the title of the movie
            df_features["year"] = df_features["year"].fillna(df_features["year"].median())  # if no year : median (not an option to have empty cells)

            # 2. Genres one-hot encoding
            genres_dummies = df_items[C.GENRES_COL].str.get_dummies(sep="|")
            genres_dummies.index = df_items.index

            if "(no genres listed)" in genres_dummies.columns:
                genres_dummies = genres_dummies.drop(columns=["(no genres listed)"])

            df_features = df_features.join(genres_dummies)

            # 3. Tags TF-IDF 
            tags = pd.read_csv(C.CONTENT_PATH / "tags.csv")
            tags["tag"] = tags["tag"].fillna("").astype(str).str.lower()

            tags_by_movie = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x))
            tags_by_movie = tags_by_movie.reindex(df_items.index).fillna("")

            tfidf = TfidfVectorizer(max_features=100, min_df=2)
            tags_matrix = tfidf.fit_transform(tags_by_movie)

            df_tags = pd.DataFrame(
                tags_matrix.toarray(),
                index=df_items.index,
                columns=[f"tag_{c}" for c in tfidf.get_feature_names_out()])

            df_features = df_features.join(df_tags)

            # 4. Genome scores
            genome_scores = pd.read_csv(C.CONTENT_PATH / "genome-scores.csv")

            genome_features = genome_scores.pivot(
                index="movieId",
                columns="tagId",
                values="relevance")

            genome_features.columns = [f"genome_{c}" for c in genome_features.columns]

            df_features = df_features.join(genome_features, how="left")

            # 5. Visual features
            visual_path = C.CONTENT_PATH / "LLVisualFeatures13K_QuantileLog.csv"

            if visual_path.exists():
                visuals = pd.read_csv(visual_path)
                visuals = visuals.rename(columns={"ML_Id": "movieId"})
                visuals = visuals.set_index("movieId")
                visuals.columns = [f"visual_{c}" for c in visuals.columns]

                df_features = df_features.join(visuals, how="left")

            # 6. Missing values
            df_features = df_features.fillna(0)

            # 7. Normalize everything
            scaler = MinMaxScaler()
            df_features[df_features.columns] = scaler.fit_transform(df_features)

        elif features_method == "V3":
            df_features = pd.DataFrame(index=df_items.index)

            # 1. Year
            df_features["year"] = df_items[C.LABEL_COL].str.extract(r"\((\d{4})\)").astype(float)
            df_features["year"] = df_features["year"].fillna(df_features["year"].median())

            # 2. Decade (one-hot encoding)
            df_features["decade"] = (df_features["year"] // 10) * 10
            decade_dummies = pd.get_dummies(df_features["decade"], prefix="decade")
            decade_dummies.index = df_features.index
            df_features = df_features.drop(columns=["decade"]).join(decade_dummies)

            # 3. Genres (one-hot encoding)
            genres_dummies = df_items[C.GENRES_COL].str.get_dummies(sep="|")
            genres_dummies.index = df_items.index

            if "(no genres listed)" in genres_dummies.columns:
                genres_dummies = genres_dummies.drop(columns=["(no genres listed)"])

            df_features = df_features.join(genres_dummies)

            # 4. Genre meta-features
            genres_clean = df_items[C.GENRES_COL].fillna("").replace("(no genres listed)", "")

            df_features["n_genres"] = genres_clean.apply(
                lambda x: 0 if x == "" else len(x.split("|")))  # number of genres (movie is specialized or hybrid)

            genre_freq = genres_dummies.mean(axis=0)
            genre_rarity = 1 - genre_freq

            df_features["genre_popularity_mean"] = genres_dummies.dot(genre_freq) / df_features["n_genres"].replace(0, np.nan)  # popularity of the genre

            # 5. Tags TF-IDF + tag meta-features
            tags_path = C.CONTENT_PATH / "tags.csv"

            if tags_path.exists():
                tags = pd.read_csv(tags_path)
                tags["tag"] = tags["tag"].fillna("").astype(str).str.lower()

                tags_by_movie = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x))
                tags_by_movie = tags_by_movie.reindex(df_items.index).fillna("")

                tfidf = TfidfVectorizer(max_features=150, min_df=2)
                tags_matrix = tfidf.fit_transform(tags_by_movie)

                df_tags = pd.DataFrame(
                    tags_matrix.toarray(),
                    index=df_items.index,
                    columns=[f"tag_{c}" for c in tfidf.get_feature_names_out()])

                df_features = df_features.join(df_tags)

                n_tags = tags.groupby("movieId").size().reindex(df_items.index).fillna(0)  # number of tags
                df_features["n_tags"] = n_tags

                tag_freq = tags["tag"].value_counts()
                tags["tag_frequency"] = tags["tag"].map(tag_freq)

                avg_tag_freq = tags.groupby("movieId")["tag_frequency"].mean()
                df_features["avg_tag_frequency"] = avg_tag_freq.reindex(df_items.index).fillna(0)

            # 6. Genome scores + aggregate genome features
            genome_path = C.CONTENT_PATH / "genome-scores.csv"

            if genome_path.exists():
                genome_scores = pd.read_csv(genome_path)

                genome_features = genome_scores.pivot(
                    index="movieId",
                    columns="tagId",
                    values="relevance")

                genome_features.columns = [f"genome_{c}" for c in genome_features.columns]

                df_features["genome_mean"] = genome_features.mean(axis=1).reindex(df_items.index)
                df_features["genome_std"] = genome_features.std(axis=1).reindex(df_items.index)
                df_features["genome_max"] = genome_features.max(axis=1).reindex(df_items.index)
                df_features["genome_min"] = genome_features.min(axis=1).reindex(df_items.index)
                df_features["genome_n_strong"] = (genome_features > 0.8).sum(axis=1).reindex(df_items.index)

                df_features = df_features.join(genome_features, how="left")

            # 7. Visual features + derived visual features
            visual_path = C.CONTENT_PATH / "LLVisualFeatures13K_QuantileLog.csv"

            if visual_path.exists():
                visuals = pd.read_csv(visual_path)
                visuals = visuals.rename(columns={"ML_Id": "movieId", "ML_ID": "movieId"})
                visuals = visuals.set_index("movieId")

                # Expected columns: f1 to f7
                if all(col in visuals.columns for col in ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]):
                    visuals["visual_fast_paced"] = visuals["f4"] / (visuals["f1"] + 1e-6)  # motion mean / average shot_length
                    visuals["visual_action_intensity"] = visuals["f4"] + visuals["f7"]  # motion mean + number of shots
                    visuals["visual_color_complexity"] = visuals["f2"] + visuals["f3"]  # color_mean + color_std
                    visuals["visual_motion_complexity"] = visuals["f4"] + visuals["f5"]  # motion_mean + motion_std
                    visuals["visual_dark_score"] = 1 - visuals["f6"]  # 1 - lighting

                visuals.columns = [f"visual_{c}" if not str(c).startswith("visual_") else c for c in visuals.columns]

                df_features = df_features.join(visuals, how="left")

            # 8. Rating-based global movie features
            ratings_path = C.EVIDENCE_PATH / C.RATINGS_FILENAME

            if ratings_path.exists():
                ratings = pd.read_csv(ratings_path)

                movie_stats = ratings.groupby(C.ITEM_ID_COL)[C.RATING_COL].agg(
                    movie_n_ratings="count",
                    movie_mean_rating="mean",
                    movie_std_rating="std")

                movie_stats["movie_std_rating"] = movie_stats["movie_std_rating"].fillna(0)

                global_mean = ratings[C.RATING_COL].mean()
                m = 10

                movie_stats["movie_bayesian_mean"] = (
                    (movie_stats["movie_n_ratings"] * movie_stats["movie_mean_rating"] + m * global_mean)
                    / (movie_stats["movie_n_ratings"] + m))

                movie_stats["movie_inverse_popularity"] = 1 / (movie_stats["movie_n_ratings"] + 1)

                df_features = df_features.join(movie_stats, how="left")

            # 10. Missing values
            df_features = df_features.fillna(0)

            # 11. Normalize everything
            scaler = MinMaxScaler()
            df_features[df_features.columns] = scaler.fit_transform(df_features)
        
                    
        else: # (implement other feature creations here)
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')
        
        return df_features
    

    def fit(self, trainset):
        """Profile Learner"""
        AlgoBase.fit(self, trainset)
        
        # Preallocate user profiles
        self.user_profile = {u: None for u in trainset.all_users()}

        if self.regressor_method == 'random_score':
            pass
        
        elif self.regressor_method == 'random_sample':
            for u in self.user_profile:
                self.user_profile[u] = [rating for _, rating in self.trainset.ur[u]]

        elif self.regressor_method == "linear_regression":

            feature_names = self.content_features.columns.tolist()

            for u in self.user_profile:  # loop over every user in the self.user_profile dictionary

                df_user = pd.DataFrame(
                    self.trainset.ur[u],
                    columns=["inner_item_id", "user_ratings"])

                df_user["item_id"] = df_user["inner_item_id"].map(
                    self.trainset.to_raw_iid)

                df_user = df_user.merge(
                    self.content_features,
                    how="left",
                    left_on="item_id",
                    right_index=True)

                X = df_user[feature_names].values
                y = df_user["user_ratings"].values

                reg = LinearRegression(fit_intercept=True)
                reg.fit(X, y)

                self.user_profile[u] = reg 
        
        elif self.regressor_method == "ridge_10":

            feature_names = self.content_features.columns.tolist()

            for u in self.user_profile:  # loop over every user in the self.user_profile dictionary

                df_user = pd.DataFrame(
                    self.trainset.ur[u],
                    columns=["inner_item_id", "user_ratings"])

                df_user["item_id"] = df_user["inner_item_id"].map(
                    self.trainset.to_raw_iid)

                df_user = df_user.merge(
                    self.content_features,
                    how="left",
                    left_on="item_id",
                    right_index=True)

                X = df_user[feature_names].values
                y = df_user["user_ratings"].values

                reg = Ridge(alpha=10)
                reg.fit(X, y)

                self.user_profile[u] = reg  

        elif self.regressor_method == "ridge_24":

            feature_names = self.content_features.columns.tolist()

            for u in self.user_profile:  # loop over every user in the self.user_profile dictionary

                df_user = pd.DataFrame(
                    self.trainset.ur[u],
                    columns=["inner_item_id", "user_ratings"])

                df_user["item_id"] = df_user["inner_item_id"].map(
                    self.trainset.to_raw_iid)

                df_user = df_user.merge(
                    self.content_features,
                    how="left",
                    left_on="item_id",
                    right_index=True)

                X = df_user[feature_names].values
                y = df_user["user_ratings"].values

                reg = Ridge(alpha=24)
                reg.fit(X, y)

                self.user_profile[u] = reg 

        elif self.regressor_method == "ridge_25":

            feature_names = self.content_features.columns.tolist()

            for u in self.user_profile:  # loop over every user in the self.user_profile dictionary

                df_user = pd.DataFrame(
                    self.trainset.ur[u],
                    columns=["inner_item_id", "user_ratings"])

                df_user["item_id"] = df_user["inner_item_id"].map(
                    self.trainset.to_raw_iid)

                df_user = df_user.merge(
                    self.content_features,
                    how="left",
                    left_on="item_id",
                    right_index=True)

                X = df_user[feature_names].values
                y = df_user["user_ratings"].values

                reg = Ridge(alpha=25)
                reg.fit(X, y)

                self.user_profile[u] = reg 

        elif self.regressor_method == "ridge_23":

            feature_names = self.content_features.columns.tolist()

            for u in self.user_profile:  # loop over every user in the self.user_profile dictionary

                df_user = pd.DataFrame(
                    self.trainset.ur[u],
                    columns=["inner_item_id", "user_ratings"])

                df_user["item_id"] = df_user["inner_item_id"].map(
                    self.trainset.to_raw_iid)

                df_user = df_user.merge(
                    self.content_features,
                    how="left",
                    left_on="item_id",
                    right_index=True)

                X = df_user[feature_names].values
                y = df_user["user_ratings"].values

                reg = Ridge(alpha=23)
                reg.fit(X, y)

                self.user_profile[u] = reg 



        elif self.regressor_method == "elasticnet":
            
            feature_names = self.content_features.columns.tolist()

            for u in self.user_profile:  # loop over every user in the self.user_profile dictionary

                df_user = pd.DataFrame(
                    self.trainset.ur[u],
                    columns=["inner_item_id", "user_ratings"])

                df_user["item_id"] = df_user["inner_item_id"].map(
                    self.trainset.to_raw_iid)

                df_user = df_user.merge(
                    self.content_features,
                    how="left",
                    left_on="item_id",
                    right_index=True)

                X = df_user[feature_names].values
                y = df_user["user_ratings"].values

                reg = ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=5000)
                reg.fit(X, y)

                self.user_profile[u] = reg 

        return self


    def estimate(self, u, i):
        """Scoring component used for item filtering"""
        # First, handle cases for unknown users and items
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')


        if self.regressor_method == 'random_score':
            rd.seed()
            score = rd.uniform(0.5,5)  # picks a random score between 0.5 and 5

        elif self.regressor_method == 'random_sample':
            rd.seed()
            score = rd.choice(self.user_profile[u])  # pick a random score in the trainset for each user
        
        elif self.regressor_method in ["linear_regression", "ridge_10", "ridge_24", "ridge_25", "ridge_26", "elasticnet"]:

            raw_item_id = self.trainset.to_raw_iid(i)  # As i is an inner item id, you first need to convert it to a raw item id

            X = self.content_features.loc[raw_item_id:raw_item_id, :].values  # find the item features

            reg = self.user_profile[u]

            score = reg.predict(X)[0]  #  select the first (and only) element in the array
            score = np.clip(score, 0.5, 5.0)  # disable predictions that are out of the rating frame

        else:
            raise NotImplementedError(
                f"Regressor method {self.regressor_method} not implemented")

        return score



from sklearn.preprocessing import normalize

class ContentKNN(AlgoBase):
    def __init__(self, features_method="V3", k=20):
        AlgoBase.__init__(self)
        self.features_method = features_method
        self.k = k

        temp_model = ContentBased(features_method=features_method, regressor_method="ridge_25")
        self.content_features = temp_model.content_features

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        self.feature_names = self.content_features.columns.tolist()
        self.item_matrix = pd.DataFrame(
            normalize(self.content_features.values),
            index=self.content_features.index,
            columns=self.feature_names
        )

        self.global_mean = np.mean([r for (_, _, r) in trainset.all_ratings()])

        self.user_items = {}
        self.user_ratings = {}
        self.user_means = {}

        for u in trainset.all_users():
            raw_items = []
            ratings = []

            for inner_i, r in trainset.ur[u]:
                raw_i = trainset.to_raw_iid(inner_i)
                if raw_i in self.item_matrix.index:
                    raw_items.append(raw_i)
                    ratings.append(r)

            self.user_items[u] = raw_items
            self.user_ratings[u] = np.array(ratings)
            self.user_means[u] = np.mean(ratings) if len(ratings) > 0 else self.global_mean

        return self

    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("Unknown user or item")

        raw_i = self.trainset.to_raw_iid(i)

        if raw_i not in self.item_matrix.index:
            return self.user_means.get(u, self.global_mean)

        rated_items = self.user_items[u]
        ratings = self.user_ratings[u]
        user_mean = self.user_means[u]

        if len(rated_items) == 0:
            return self.global_mean

        target_vec = self.item_matrix.loc[raw_i].values
        rated_matrix = self.item_matrix.loc[rated_items].values

        sims = rated_matrix @ target_vec

        # only positive similarities
        mask = sims > 0
        sims = sims[mask]
        ratings = ratings[mask]

        if len(sims) == 0:
            return user_mean

        top_idx = np.argsort(sims)[-self.k:]
        top_sims = sims[top_idx]
        top_ratings = ratings[top_idx]

        score = user_mean + np.sum(top_sims * (top_ratings - user_mean)) / (np.sum(np.abs(top_sims)) + 1e-8)

        return np.clip(score, 0.5, 5.0)


class MetaHybridModel(AlgoBase):
    def __init__(self, features_method="V_ULTIMATE"):
        AlgoBase.__init__(self)
        # On initialise le Content-Based local (Ridge) pour avoir une base solide
        self.local_model = ContentBased(features_method=features_method, regressor_method="ridge_25")
        self.global_xgb = XGBRegressor(
            n_estimators=50, 
            max_depth=5, 
            learning_rate=0.05, 
            n_jobs=-1
        )

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        # 1. Entraîner le modèle local (Ridge par utilisateur)
        self.local_model.fit(trainset)
        
        # 2. Préparer les données pour le modèle Global
        # On va créer un dataset qui contient les features du film + les stats de l'utilisateur
        # + la prédiction faite par le modèle local Ridge
        rows = []
        content_feats = self.local_model.content_features
        
        # Calculer les stats utilisateurs
        user_stats = {}
        for u in trainset.all_users():
            ratings = [r for (_, r) in trainset.ur[u]]
            user_stats[u] = {
                'u_mean': np.mean(ratings),
                'u_std': np.std(ratings) if len(ratings) > 1 else 0,
                'u_count': len(ratings)
            }

        for u, i, r in trainset.all_ratings():
            # On récupère la prédiction "locale" du Ridge
            pred_local = self.local_model.estimate(u, i)
            
            feat_row = content_feats.loc[trainset.to_raw_iid(i)].to_dict()
            feat_row.update(user_stats[u])
            feat_row['pred_ridge'] = pred_local
            feat_row['target'] = r
            rows.append(feat_row)

        df_train = pd.DataFrame(rows)
        X = df_train.drop(columns=['target'])
        y = df_train['target']
        
        self.global_xgb.fit(X, y)
        self.feature_cols = X.columns
        return self

    def estimate(self, u, i):
        # Récupérer les données de la même manière que pour le fit
        pred_local = self.local_model.estimate(u, i)
        content_row = self.local_model.content_features.loc[self.trainset.to_raw_iid(i)].to_dict()
        
        ratings = [r for (_, r) in self.trainset.ur[u]]
        user_row = {
            'u_mean': np.mean(ratings),
            'u_std': np.std(ratings) if len(ratings) > 1 else 0,
            'u_count': len(ratings),
            'pred_ridge': pred_local
        }
        
        content_row.update(user_row)
        X_eval = pd.DataFrame([content_row])[self.feature_cols]
        
        return np.clip(self.global_xgb.predict(X_eval)[0], 0.5, 5.0)