class FilterHybrid:
    """
    A simple hybrid recommender that uses a two-stage filter approach.

    This model first generates a list of personalized candidates using a trained
    ALS model (Candidate Generation). It then filters this list, keeping only
    the items that meet a predefined quality threshold (Filtering). This
    approach preserves the personalization of the ALS model while ensuring a
    high standard of quality for the final recommendations.
    """
    def __init__(self, als_model, games_df, quality_threshold=75):
        """
        Initializes the SimpleFilterHybrid model.

        :param als_model: A trained instance of the ALSRecommender.
        :param games_df: The full DataFrame of games, containing
                         'title' and 'positive_ratio'.
        :param quality_threshold: The minimum positive_ratio for a game to
                                 be recommended.
        """
        self.als_model = als_model
        self.games_df = games_df
        self.quality_threshold = quality_threshold
        # Create a set of high-quality games for fast lookup.
        self.high_quality_set = set(
            self.games_df[self.games_df['positive_ratio'] >= self.quality_threshold]['title']
        )

    def predict(self, user_id, k=10):
        """
        Generates top-k hybrid recommendations for a given user.

        The method preserves the original ranking from the ALS model.

        :param user_id: The ID of the user to generate recommendations for.
        :param k: The number of recommendations to return.
        :return: A list of recommended game titles.
        """
        # Generate a large list of personalized candidates from the ALS model.
        als_candidates, _ = self.als_model.predict(user_id, k=100)
        if not als_candidates:
            return []

        # Filter the ranked list, keeping only high-quality games.
        final_recommendations = []
        for game in als_candidates:
            if game in self.high_quality_set:
                final_recommendations.append(game)
            # Stop once we have collected enough recommendations.
            if len(final_recommendations) == k:
                break

        return final_recommendations
