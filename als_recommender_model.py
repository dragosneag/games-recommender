import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
import config


class ALSRecommender:
    """
    A recommender system based on the Alternating Least Squares (ALS) algorithm
    for collaborative filtering on implicit feedback data.

    This class trains on user-item interaction data (weighted by a confidence
    score) to learn latent factor representations for users and items. It can
    then predict personalized recommendations for a given user.
    """
    def __init__(self, factors=64, regularization=0.01, iterations=50):
        """
        Initializes the ALSRecommender with model hyperparameters.

        :param factors: The number of latent factors to compute.
        :param regularization: The regularization factor to use.
        :param iterations: The number of ALS iterations to run.
        """
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=config.RANDOM_STATE
        )
        self.user_item_matrix = None
        self.user_map = None
        self.item_map = None
        self.item_inv_map = None
        self.train_df = None
        self.games_df = None

    def _create_mappings(self, train_df):
        """
        Creates mappings from original IDs to sequential integer indices.

        :param train_df: DataFrame of user-item interactions for training.
        """
        self.user_map = {user_id: i for i, user_id in enumerate(train_df['user_id'].unique())}
        self.item_map = {title: i for i, title in enumerate(train_df['title'].unique())}
        # Inverse mapping is needed to convert predicted indices back to titles.
        self.item_inv_map = {i: title for title, i in self.item_map.items()}

    def _create_user_item_matrix(self, train_df):
        """
        Creates the sparse user-item interaction matrix.

        :param train_df: DataFrame of user-item interactions for training.
        """
        rows = train_df['user_id'].map(self.user_map)
        cols = train_df['title'].map(self.item_map)
        data = train_df['confidence'].astype(float)

        self.user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(self.user_map), len(self.item_map)))

    def fit(self, train_df, games_df):
        """
        Trains the ALS model on the provided interaction data.

        :param train_df: DataFrame of user-item interactions for training.
        :param games_df: DataFrame of game data.
        :return:
        """
        self.train_df = train_df
        self.games_df = games_df
        print("Training ALS model...")

        self._create_mappings(train_df)
        self._create_user_item_matrix(train_df)

        self.model.fit(self.user_item_matrix)
        print("ALS model trained successfully!")

    def predict(self, user_id, k=config.NUM_RECOMMENDATIONS):
        """
        Generates top-k recommendations and their scores for a given user.

        :param user_id: The ID of the user to generate recommendations for.
        :param k: The number of recommendations to return.
        :return: A tuple containing a list of
               recommended game titles and a
               dictionary mapping those titles
               to their recommendation scores.
        """
        if user_id not in self.user_map:
            return [], {}

        # Get the internal index for the user
        user_idx = self.user_map[user_id]

        # Get recommendations (returns item indices and scores)
        item_indices, scores = self.model.recommend(user_idx, self.user_item_matrix[user_idx], N=k)

        # Convert indices back to game titles
        recommendations = [self.item_inv_map[idx] for idx in item_indices]
        rec_scores = {self.item_inv_map[idx]: score for idx, score in zip(item_indices, scores)}

        return recommendations, rec_scores
