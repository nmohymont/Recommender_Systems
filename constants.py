# third parties imports
from pathlib import Path


class Constant:

    DATA_PATH = Path('data/small')  # -- fill here the dataset size to use

    # Content
    CONTENT_PATH = DATA_PATH / 'content'
    # - item
    ITEMS_FILENAME = 'movies.csv'
    ITEM_ID_COL = 'movieId'
    LABEL_COL = 'title'
    GENRES_COL = 'genres'

    # Evidence
    EVIDENCE_PATH = DATA_PATH / 'evidence'
    # - ratings
    RATINGS_FILENAME = 'ratings.csv'
    USER_ID_COL = 'userId'
    RATING_COL = 'rating'
    TIMESTAMP_COL = 'timestamp'
    USER_ITEM_RATINGS = [USER_ID_COL, ITEM_ID_COL, RATING_COL]

    # Rating scale
    RATINGS_SCALE = (0.5, 5.0) # -- configure the rating scale (min, max) --
    EVALUATION_PATH = CONTENT_PATH.parent / "evaluation" # -- configure the path to export the evaluation report --
