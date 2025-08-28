# content_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json


class ContentBasedRecommender:
    """
    A recommender system that finds similar games based on their textual content.

    This model works by transforming game metadata (tags and descriptions) into
    numerical vectors using TF-IDF. It can then calculate the cosine similarity
    between these vectors to find and recommend games with the most similar
    content.
    """

    def __init__(self):
        self.tfidf_matrix = None
        self.game_data = None
        self.indices = None

    def fit(self, games_df, metadata_path):
        """
        Processes game metadata and computes the TF-IDF feature matrix.

        :param games_df: The raw DataFrame of games.
        :param metadata_path: The file path to the games_metadata.json file.
        """
        print("Training Content-Based model...")

        # Load and merge metadata with the primary games dataframe.
        metadata_dict = {item['app_id']: item for item in
                         (json.loads(line) for line in open(metadata_path, 'r', encoding='utf-8'))}
        metadata_df = pd.DataFrame.from_dict(metadata_dict, orient='index')
        merged_df = pd.merge(games_df, metadata_df, on='app_id', how='left')

        # De-duplicate by title to create a clean catalog.
        self.game_data = merged_df.drop_duplicates(subset=['title']).reset_index(drop=True)

        # Clean and combine text features into a single "content soup".
        self.game_data['description'] = self.game_data['description'].fillna('')
        self.game_data['tags'] = self.game_data['tags'].fillna('')
        self.game_data['content_soup'] = (
                self.game_data['tags'].apply(lambda x: ' '.join(x).replace('-', '')) + ' ' +
                self.game_data['description']
        )

        # Vectorize the content soup using TF-IDF. This converts text into a
        # meaningful numerical representation based on word importance.
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.game_data['content_soup'])

        # Create a mapping from game title to its row index for fast lookups.
        self.indices = pd.Series(self.game_data.index, index=self.game_data['title'])
        print("Content-Based model trained successfully!")

    def get_similarity(self, game1_title, game2_title):
        """
        Calculates the cosine similarity between two specific games on-demand.

        :param game1_title: The title of the first game.
        :param game2_title: The title of the second game.
        :return: The cosine similarity score between 0 and 1.
        """
        if game1_title not in self.indices or game2_title not in self.indices:
            return 0

        idx1 = self.indices[game1_title]
        idx2 = self.indices[game2_title]

        return cosine_similarity(self.tfidf_matrix[idx1], self.tfidf_matrix[idx2])[0][0]

    def predict(self, game_title, k=10):
        """
        Gets top-k similar games for a given game based on content similarity.

        :param game_title: The title of the game to find recommendations for.
        :param k: The number of similar games to return.
        :return: A list of the top-k most similar game titles.
        """
        if game_title not in self.indices:
            return []

        idx = self.indices[game_title]
        game_vector = self.tfidf_matrix[idx]

        # Calculate similarity scores between the input game and all other games.
        sim_scores = cosine_similarity(game_vector, self.tfidf_matrix)

        # Sort and retrieve the top-k most similar games.
        sim_scores = list(enumerate(sim_scores[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:k + 1] # Exclude the game itself.

        game_indices = [i[0] for i in sim_scores]
        return self.game_data['title'].iloc[game_indices].tolist()
