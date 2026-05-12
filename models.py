# standard library imports
from collections import defaultdict
import os

# third parties imports
import numpy as np
import pandas as pd
import random as rd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD
from surprise import PredictionImpossible
from surprise import dump

from sklearn.linear_model import LinearRegression,Lasso, ElasticNet

from sklearn.ensemble import RandomForestRegressor

from loaders import load_ratings, load_items, get_tfidf_tags_features

from constants import Constant as C

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
    def __init__(self):
        SVD.__init__(self, n_factors=100)

class ContentBased(AlgoBase):
    def __init__(self, features_method, regressor_method):
        AlgoBase.__init__(self)
        self.features_method = features_method #ligne ajoutée pour sauvegarder la features_method dans l'objet avec self.
        self.regressor_method = regressor_method
        self.content_features = self.create_content_features(features_method)

    def create_content_features(self, features_method):
        """Content Analyzer"""
        df_items = load_items()
        if features_method is None:
            df_features = None
        elif features_method == "title_length": # a naive method that creates only 1 feature based on title length
            df_features = df_items[C.LABEL_COL].apply(lambda x: len(x)).to_frame('n_character_title')

        elif features_method == 'year':
            # 1. La date (convertie en nombre)
            df_features = df_items[C.LABEL_COL].str.extract(r'\((\d{4})\)').astype(float)
            df_features.columns = ['release_year']
            df_features['release_year'] = df_features['release_year'].fillna(df_features['release_year'].median())

            # Min-Max normalization to get release_year in range(0 to 1) the same scale as categorial encoding
            min_year = df_year['release_year'].min()
            max_year = df_year['release_year'].max()
            df_year['release_year'] = (df_year['release_year'] - min_year) / (max_year - min_year)

        elif features_method == 'genres_one_hot_encoding_alone':
            #Les genres (One-Hot Encoding, déjà entre 0 et 1)
            df_features = df_items[C.GENRES_COL].str.get_dummies(sep='|')

            # a approfondir pour faire du One Hot Encoding intéressant pour regrouper les features
        elif features_method == 'genres_one_hot_encoding_features_engineering':
            #Les genres (One-Hot Encoding, déjà entre 0 et 1)
            df_features = df_items[C.GENRES_COL].str.get_dummies(sep='|')
            
            #  Feature Engineering (Top 5 Combinaisons) ---
            # On crée de nouvelles colonnes basées sur la multiplication des genres simples
            df_features['Comedy_Drama'] = df_features['Comedy'] * df_features['Drama']
            df_features['Drama_Romance'] = df_features['Drama'] * df_features['Romance']
            df_features['Comedy_Romance'] = df_features['Comedy'] * df_features['Romance']
            df_features['Comedy_Drama_Romance'] = df_features['Comedy'] * df_features['Drama'] * df_features['Romance']
            df_features['Drama_Thriller'] = df_features['Drama'] * df_features['Thriller']

        
        elif features_method =="date_and_genres":
            df_year = df_items[C.LABEL_COL].str.extract(r'\((\d{4})\)').astype(float)
            df_year.columns = ['release_year']
            df_year['release_year'] = df_year['release_year'].fillna(df_year['release_year'].median())
            
            # --- NOUVEAU : Normalisation Min-Max (entre 0 et 1) ---
            min_year = df_year['release_year'].min()
            max_year = df_year['release_year'].max()
            df_year['release_year'] = (df_year['release_year'] - min_year) / (max_year - min_year)
            
            # 2. Les genres 
            df_genres = df_items[C.GENRES_COL].str.get_dummies(sep='|')

            # Feature Engineering (Top 5 Combinaisons) ---
            # On crée de nouvelles colonnes basées sur la multiplication des genres simples
            df_genres['Comedy_Drama'] = df_genres['Comedy'] * df_genres['Drama']
            df_genres['Drama_Romance'] = df_genres['Drama'] * df_genres['Romance']
            df_genres['Comedy_Romance'] = df_genres['Comedy'] * df_genres['Romance']
            df_genres['Comedy_Drama_Romance'] = df_genres['Comedy'] * df_genres['Drama'] * df_genres['Romance']
            df_genres['Drama_Thriller'] = df_genres['Drama'] * df_genres['Thriller']

            # 3. Assemblage du tableau final
            df_features = pd.concat([df_year, df_genres], axis=1)

        elif features_method =='tfidf_tags':
            
            # Appel de la fonction externe
            df_features = get_tfidf_tags_features(C.CONTENT_PATH/C.TAGS_FILENAME)

        else: 
            pass # (implement other feature creations here)
            
            raise NotImplementedError(f'Feature method {features_method} not yet implemented')
        return df_features
    
    def prepare_user_data(self, u):
        """Prépare et retourne les features (X) et la cible (y) pour un utilisateur donné."""
        feature_names = self.content_features.columns.tolist() 
        ratings_list = self.trainset.ur[u]
        
        df_user = pd.DataFrame(ratings_list, columns=['item_id', 'user_ratings'])
        df_user['item_id'] = df_user['item_id'].map(self.trainset.to_raw_iid)
        
        df_user = df_user.merge(
            self.content_features,
            how='left',
            left_on='item_id',
            right_index=True
        )
        
        X = df_user[feature_names].values
        y = df_user['user_ratings'].values
        return X, y

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
        
        elif self.regressor_method == 'linear_regression':
            for u in self.user_profile:
                X, y = self.prepare_user_data(u)
                reg = LinearRegression(fit_intercept=True)
                reg.fit(X, y)
                self.user_profile[u] = reg
                
        elif self.regressor_method == 'lasso_regression':
            for u in self.user_profile:
                X, y = self.prepare_user_data(u)
                reg = Lasso(alpha=0.1, fit_intercept=True)
                reg.fit(X, y)
                self.user_profile[u] = reg
                
        elif self.regressor_method == 'elasticnet':
            for u in self.user_profile:
                X, y = self.prepare_user_data(u)
                reg = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True)
                reg.fit(X, y)
                self.user_profile[u] = reg

        elif self.regressor_method == 'random_forest':
            for u in self.user_profile:
                X, y = self.prepare_user_data(u)
                reg = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
                reg.fit(X, y)
                self.user_profile[u] = reg
        else:
            pass
            # (implement here the regressor fitting)  
            
        # --- AUTO-SAUVEGARDE DU MODÈLE À LA FIN DE L'ENTRAÎNEMENT ---
        dossier_destination = C.RECS_PATH
        os.makedirs(dossier_destination, exist_ok=True)
        chemin_fichier = os.path.join(dossier_destination, f"modele_{self.regressor_method}.p")
        dump.dump(chemin_fichier, algo=self)
        print(f"Modèle '{self.regressor_method}' sauvegardé avec succès sous {chemin_fichier}")
        
    def estimate(self, u, i):
        """Scoring component used for item filtering"""
        # First, handle cases for unknown users and items
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')


        if self.regressor_method == 'random_score':
            rd.seed()
            score = rd.uniform(0.5,5)

        elif self.regressor_method == 'random_sample':
            rd.seed()
            score = rd.choice(self.user_profile[u])
        
        elif self.regressor_method in ['linear_regression', 'lasso_regression','random_forest']:
            
            # 1. Convertir l'ID interne (i) en ID brut MovieLens
            raw_item_id = self.trainset.to_raw_iid(i)
            
            # 2. Récupérer les caractéristiques (features) de ce film spécifique
            # .values permet de l'extraire sous forme de tableau (array) au lieu d'un DataFrame
            X_i = self.content_features.loc[raw_item_id:raw_item_id, :].values
            
            # 3. Récupérer le modèle de régression qu'on a entraîné pour l'utilisateur 'u'
            reg = self.user_profile[u]
            
            # (Sécurité) Si pour une raison quelconque l'utilisateur n'a pas de modèle
            if reg is None:
                raise PredictionImpossible('Pas de modèle entraîné pour cet utilisateur.')
                
            # 4. Faire la prédiction
            # reg.predict() renvoie un tableau, ex: [3.45]. On veut juste le chiffre, donc on prend l'index [0]
            prediction_array = reg.predict(X_i)
            score = prediction_array[0]

        else:
            score=None
            # (implement here the regressor prediction)

        return score
