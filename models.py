# standard library imports
from collections import defaultdict

# third parties imports
import numpy as np
import random as rd
from surprise import AlgoBase
from surprise import KNNWithMeans
from surprise import SVD


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
    
    top_n = defaultdict(list) #python tool to group predictions by userID without checking id the jey exists first


    for uid, iid, true_r, est, _ in predictions:
        # _ means we do not need this value : true_r is real rating not needed as anti_testset (unspecified data),  _ is details

        top_n[uid].append((iid, est))

        # uid (userID) is the key of the list
        # iid (movieID), est is a tuple giving the movieID and the rating associated

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        #user_ratings are the tuple gathering the movies and the rating given
        # the sort is done on x[1] = rating not x[0] the movieID, but as they are in the same tuple,
        # both data are connected when sorted

        rd.shuffle(user_ratings) # shuffle to handle score ties randomly
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# First algorithm prediction is always 2
class ModelBaseline1(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def estimate(self, u, i):
        return 2


# Second algorithm : prediction is random
class ModelBaseline2(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        rd.seed(0)

    def estimate(self, u, i):
        return rd.uniform(self.trainset.rating_scale[0], self.trainset.rating_scale[1])


# Third algorithm : prediction global mean
class ModelBaseline3(AlgoBase):
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])

        return self

    def estimate(self, u, i):
        return self.the_mean


# Fourth Model : prediction is SVD 100 latent factors
class ModelBaseline4(SVD):
    def __init__(self):
        SVD.__init__(self, n_factors=100)
