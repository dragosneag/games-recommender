from content_model import ContentBasedRecommender
from data_loader import load_and_preprocess_data
from evaluation import split_data, calculate_coverage, \
    calculate_personalization, calculate_accuracy_metrics
from hybrid_model import FilterHybrid
import pandas as pd
import config
import pickle


def evaluate_als_model():
    """
    Loads a pre-trained ALS model and its corresponding data splits, then
    runs a full suite of evaluations (accuracy, coverage, personalization).
    """
    print("--- Evaluating PURE ALS Model ---")

    # Load the consistent dataset for evaluation.
    print("Loading pre-trained model and evaluation data...")
    with open('als_evaluation_data.pkl', 'rb') as f:
        data = pickle.load(f)

    model = data['model']
    train_df = data['train_df']
    test_df = data['test_df']

    # --- Run Evaluations ---
    print("\nCalculating Accuracy metrics...")
    avg_precision, avg_recall = calculate_accuracy_metrics(model, test_df)
    print(f"  - Average Precision@10: {avg_precision:.4f}")
    print(f"  - Average Recall@10: {avg_recall:.4f}")

    print("\nCalculating Coverage & Personalization metrics...")
    coverage = calculate_coverage(model, train_df)
    personalization = calculate_personalization(model, train_df)
    print(f"  - Catalog Coverage: {coverage:.4f}")
    print(f"  - Personalization Score: {personalization:.4f}")

    # Get and print example recommendations for a specific user
    example_user_id = train_df['user_id'].value_counts().index[0]

    recommendations = model.predict(example_user_id)
    print(f"\nRecommendations for user_id '{example_user_id}':")
    if recommendations:
        for game in recommendations:
            print(game)
    else:
        print("Could not find recommendations for this user.")


def evaluate_content_model():
    """
    Trains a Content-Based model from scratch and evaluates its accuracy.
    """
    print("--- Evaluating Content-Based Model ---")

    # Train the Content-Based Model.
    games_df_full = pd.read_csv(config.GAMES_CSV_PATH)
    content_model = ContentBasedRecommender()
    content_model.fit(games_df_full, 'data/games_metadata.json')

    # Load and split interaction data to use as ground truth.
    interaction_df = load_and_preprocess_data()
    train_df, test_df = split_data(interaction_df)

    # Evaluate the model's accuracy.
    avg_precision, avg_recall = calculate_accuracy_metrics(
        content_model, train_df, test_df, sample_size=1000
    )
    print("\n--- Content-Based Model Evaluation Results ---")
    print(f"  - Average Precision@10: {avg_precision:.4f}")
    print(f"  - Average Recall@10: {avg_recall:.4f}")

    print("\nRecommending similar games to: FlatOut 4: Total Insanity")
    print(content_model.predict("FlatOut 4: Total Insanity"))


def evaluate_hybrid_model():
    """
    Loads a pre-trained ALS model and evaluates the FilterHybrid model.
    """
    print("--- Evaluating Hybrid Model ---")

    # Load the pre-trained ALS model and data splits.
    print("Loading pre-trained model and evaluation data...")
    with open('als_evaluation_data.pkl', 'rb') as f:
        data = pickle.load(f)

    als_model = data['model']
    train_df = data['train_df']
    test_df = data['test_df']
    games_df_full = pd.read_csv(config.GAMES_CSV_PATH)

    # Initialize the hybrid model.
    hybrid_model = FilterHybrid(als_model, games_df_full)

    # Evaluate the hybrid model's accuracy.
    avg_precision, avg_recall = calculate_accuracy_metrics(hybrid_model, test_df)
    print("\n--- Hybrid Model Evaluation Results ---")
    print(f"  - Average Precision@10: {avg_precision:.4f}")
    print(f"  - Average Recall@10: {avg_recall:.4f}")

    # Get and print example recommendations for a specific user
    example_user_id = train_df['user_id'].value_counts().index[0]

    recommendations = hybrid_model.predict(example_user_id)
    print(f"\nRecommendations for user_id '{example_user_id}':")
    if recommendations:
        for game in recommendations:
            print(game)
    else:
        print("Could not find recommendations for this user.")


if __name__ == '__main__':
    # Choose which model evaluation to run by uncommenting the desired function.
    # evaluate_als_model()
    # evaluate_content_model()
    evaluate_hybrid_model()
