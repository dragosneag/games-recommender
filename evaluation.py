import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import config
import random
from sklearn.metrics.pairwise import cosine_similarity


def split_data(df):
    """
    Splits interaction data into training and testing sets on a per-user basis.

    This ensures that each user has a portion of their interaction history in both
    the training and testing sets, which is crucial for offline evaluation.

    :param df: The input DataFrame of user-item interactions.
    :return: A tuple containing the training DataFrame and the testing DataFrame.
    """
    train_list, test_list = [], []
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id]
        train_user, test_user = train_test_split(
            user_data,
            test_size=config.TEST_SET_SIZE,
            random_state=config.RANDOM_STATE
        )
        train_list.append(train_user)
        test_list.append(test_user)
    return pd.concat(train_list), pd.concat(test_list)


def calculate_accuracy_metrics(model, test_df, k=config.NUM_RECOMMENDATIONS, sample_size=1000):
    """
    Calculates precision@k and recall@k for a given recommender model.

    This function iterates through each user in the test set, generates
    recommendations, and compares them to the held-out items (ground truth).

    :param model: A trained recommender model object with a .predict(user_id, k) method.
    :param test_df: The DataFrame of test interactions.
    :param k: The number of recommendations to generate per user.
    :param sample_size: The number of users to sample for the calculation.
    :return: A tuple containing the average precision@k and recall@k.
    """
    test_user_items = test_df.groupby('user_id')['title'].apply(list).to_dict()

    # Sample the users to speed up evaluation
    if sample_size is not None and sample_size < len(test_user_items):
        user_sample = random.sample(list(test_user_items.keys()), sample_size)
        test_user_items_sampled = {user: test_user_items[user] for user in user_sample}
    else:
        test_user_items_sampled = test_user_items

    print(f"Evaluating Content-Based model on a sample of {len(test_user_items_sampled)} users...")

    user_precisions, user_recalls = [], []

    # Iterate through each user in our test set
    for user_id, true_items in test_user_items_sampled.items():
        # Get the top-k recommendations for this user from the ALS model
        recommendations = model.predict(user_id, k=k)
        if not recommendations:
            continue

        # Calculate how many of the recommendations were correct (hits)
        hits = len(set(recommendations) & set(true_items))

        # Calculate precision
        # Precision = (Correct Recommendations) / (Total Recommendations Made)
        precision = hits / k
        user_precisions.append(precision)

        # Calculate recall
        # Recall = (Correct Recommendations) / (Total Items the User Actually Liked)
        recall = hits / len(true_items)
        user_recalls.append(recall)

    avg_precision = sum(user_precisions) / len(user_precisions) if user_precisions else 0
    avg_recall = sum(user_recalls) / len(user_recalls) if user_recalls else 0

    return avg_precision, avg_recall


def calculate_coverage(model, train_df, k=config.NUM_RECOMMENDATIONS):
    """
    Calculates the catalog coverage of the recommender model.

    Coverage measures the percentage of unique items in the catalog that the
    model is able to recommend across all users.

    :param model: A trained recommender model object.
    :param train_df: The DataFrame of training interactions.
    :param k: The number of recommendations to generate per user.
    :return: The catalog coverage as a value between 0 and 1.
    """
    all_users = train_df['user_id'].unique()
    all_recommendations = set()

    for user_id in all_users:
        recs = model.predict(user_id, k=k)
        for rec in recs:
            all_recommendations.add(rec)

    total_items = len(train_df['title'].unique())
    return len(all_recommendations) / total_items


def calculate_personalization(model, train_df, k=config.NUM_RECOMMENDATIONS, sample_size=1000):
    """
    Calculates the personalization score of the recommender model.

    Personalization is measured as 1 minus the average cosine similarity
    between all pairs of user recommendation lists. A higher score indicates
    that users are receiving more distinct recommendations.

    :param model: A trained recommender model object.
    :param train_df: The DataFrame of training interactions.
    :param k: The number of recommendations to generate per user.
    :param sample_size: The number of users to sample for the calculation.
    :return: The personalization score, where 1.0 is maximum personalization.
    """
    all_users = train_df['user_id'].unique()
    sample_users = random.sample(list(all_users), min(len(all_users), sample_size))

    rec_lists = [model.predict(user_id, k=k) for user_id in sample_users]

    # Create a binary matrix where rows are users and columns are items.
    rec_df = pd.DataFrame(rec_lists).stack().reset_index(level=1, drop=True).to_frame('title')
    rec_df['user_index'] = rec_df.index
    binary_matrix = pd.crosstab(rec_df['user_index'], rec_df['title'])

    # Calculate pairwise cosine similarity between user recommendation vectors.
    similarity = cosine_similarity(binary_matrix)

    # Average the similarity scores from the upper triangle of the matrix.
    upper_triangle = similarity[np.triu_indices(similarity.shape[0], k=1)]
    avg_similarity = upper_triangle.mean()

    return 1 - avg_similarity
