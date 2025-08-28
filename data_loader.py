import numpy as np
import pandas as pd
import config


def load_and_preprocess_data():
    """
    Loads raw game and recommendation data, preprocesses it, and returns a
    filtered DataFrame ready for model training.

    :return: A preprocessed DataFrame with columns including 'user_id',
                      'title', and 'confidence'.
    """
    # Load the raw datasets, keeping only necessary columns.
    games = pd.read_csv(config.GAMES_CSV_PATH)[['app_id', 'title']]
    recommendations = pd.read_csv(config.RECOMMENDATIONS_CSV_PATH)[
        ['user_id', 'app_id', 'hours', 'helpful', 'is_recommended']
    ]
    print("Data loaded successfully!")

    # Isolate positive recommendations for modeling user preference.
    df = recommendations[recommendations['is_recommended']].copy()

    # Engineer a confidence score. This creates a more nuanced signal than a
    # simple binary interaction, weighting interactions by user engagement.
    # np.log1p is used to handle zero values gracefully (log(x+1)).
    df['confidence'] = 1 + np.log1p(df['hours']) + np.log1p(df['helpful'])

    # Merge with game titles for human-readable item identifiers.
    df = pd.merge(df, games, on='app_id')

    # Filter out inactive users and unpopular games to reduce sparsity.
    user_counts = df['user_id'].value_counts()
    game_counts = df['title'].value_counts()

    active_users = user_counts[user_counts >= config.MIN_USER_REVIEWS].index
    popular_games = game_counts[game_counts >= config.MIN_GAME_REVIEWS].index

    df_filtered = df[df['user_id'].isin(active_users) & df['title'].isin(popular_games)]
    print(f"Interactions after filtering: {len(df_filtered)}")

    return df_filtered
