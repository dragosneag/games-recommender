import pandas as pd
import pickle
from data_loader import load_and_preprocess_data
from als_recommender_model import ALSRecommender
from content_model import ContentBasedRecommender
import config
from evaluation import split_data


def train_and_save_evaluation_data():
    """
    Trains the ALS model and saves it along with the corresponding train/test
    data splits to a single file.

    This function is intended to be run once to create a consistent dataset
    for reliable offline model evaluation.
    """
    print("Loading and splitting data...")
    df = load_and_preprocess_data()
    train_df, test_df = split_data(df)
    games_df_full = pd.read_csv(config.GAMES_CSV_PATH)

    print("Training ALS model...")
    als_model = ALSRecommender()
    als_model.fit(train_df, games_df_full)

    # Bundle the model and data splits into a single dictionary.
    # This ensures that the evaluation script uses the exact same data and
    # internal model mappings as the training process, preventing data mismatch errors.
    data_to_save = {
        'model': als_model,
        'train_df': train_df,
        'test_df': test_df
    }

    with open('als_evaluation_data.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    print("\nModel and evaluation data saved successfully as 'als_evaluation_data.pkl'!")


def train_and_save_all_models():
    """
    Trains both the ALS and Content-Based models on the full dataset and
    saves them together to a single file for use in the Streamlit application.
    """
    print("Starting final model training process...")

    # Load the full, preprocessed dataset for training.
    df = load_and_preprocess_data()
    games_df_full = pd.read_csv(config.GAMES_CSV_PATH)

    # Train the ALS model.
    als_model = ALSRecommender()
    als_model.fit(df, games_df_full)

    # Train the Content-Based model.
    content_model = ContentBasedRecommender()
    content_model.fit(games_df_full, 'data/games_metadata.json')

    # Save both trained models in a dictionary for easy loading in the app.
    models = {
        'als': als_model,
        'content': content_model
    }

    with open('models.pkl', 'wb') as f:
        pickle.dump(models, f)

    print("\nAll models trained and saved successfully as 'models.pkl'!")


if __name__ == '__main__':
    # By default, this script will train the models needed for the Streamlit app.
    # To generate evaluation data, you can comment out the line below and
    # uncomment the call to train_and_save_evaluation_data().
    train_and_save_all_models()
    # train_and_save_evaluation_data()
