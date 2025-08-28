import streamlit as st
import pickle
import pandas as pd
from als_recommender_model import ALSRecommender
from content_model import ContentBasedRecommender
import config
from hybrid_model import FilterHybrid

# --- Page Configuration ---
st.set_page_config(page_title="Game Recommender", page_icon="🎮", layout="wide")


# --- Caching Functions ---
@st.cache_resource
def load_models_and_data():
    """
    Loads all pre-trained models and necessary data from disk.

    This function is cached to ensure models are loaded only once per session,
    improving the application's performance.

    :return: A tuple containing the trained
           hybrid model, content model, and the
           users DataFrame.
    """
    # Load the pure ALS and Content models from the training script.
    with open('models.pkl', 'rb') as f:
        models = pickle.load(f)
    als_model = models['als']
    content_model = models['content']

    # Load supplementary dataframes.
    games_df = pd.read_csv(config.GAMES_CSV_PATH)
    users_df = pd.read_csv('data/users.csv')

    hybrid_model = FilterHybrid(als_model, games_df)

    return hybrid_model, content_model, users_df


def render_user_based_page(hybrid_model, users_df):
    """
    Renders the UI for the user-based (hybrid) recommendation mode.
    """
    st.title("Personalized Recommendations For You")

    user_ids = hybrid_model.als_model.train_df['user_id'].unique()
    selected_user_id = st.selectbox("Choose a User ID:", options=user_ids)

    if st.button("Find Games", type="primary"):
        st.subheader(f"For User: `{selected_user_id}`")

        # Display user profile stats and their most-played games for context.
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**User Profile**")
            user_stats = users_df[users_df['user_id'] == selected_user_id].iloc[0]
            st.metric("Products Owned", f"{int(user_stats['products']):,}")
            st.metric("Reviews Written", f"{int(user_stats['reviews']):,}")

        with col2:
            st.markdown("**Based on their most played games:**")
            user_history = hybrid_model.als_model.train_df[hybrid_model.als_model.train_df['user_id'] == selected_user_id]
            top_played = user_history.sort_values(by='hours', ascending=False).head(5)
            for _, row in top_played.iterrows():
                st.markdown(f"- *{row['title']}* ({row['hours']:,.0f} hours)")

        st.markdown("---")

        # Generate and display hybrid recommendations.
        with st.spinner('Generating personalized recommendations...'):
            recommendations = hybrid_model.predict(selected_user_id)
            st.subheader("We Recommend These Games For You:")
            if recommendations:
                # Display recommendations in two rows of five.
                cols = st.columns(5)
                for i, game in enumerate(recommendations[:5]):
                    cols[i].markdown(f"**{i + 1}. {game}**")

                cols = st.columns(5)
                for i, game in enumerate(recommendations[5:]):
                    cols[i].markdown(f"**{i + 6}. {game}**")
            else:
                st.warning("Could not generate recommendations for this user.")


def render_item_based_page(content_model):
    """
    Renders the UI for the item-based (content) recommendation mode.
    """
    st.title("Find Games Similar to Your Favorites")

    all_game_titles = content_model.game_data['title'].unique()
    selected_game_title = st.selectbox("Choose a Game:", options=all_game_titles)

    if st.button("Find Similar Games", type="primary"):
        st.subheader(f"Because you like *{selected_game_title}*...")

        # Display basic info about the selected game.
        game_info = content_model.game_data[content_model.game_data['title'] == selected_game_title].iloc[0]
        st.markdown(f"**Tags:** {' | '.join(game_info['tags'])}")
        st.markdown("---")

        # Generate and display content-based recommendations.
        with st.spinner('Finding similar games...'):
            recommendations = content_model.predict(selected_game_title, k=10)
            st.subheader("You might also like these games:")
            if recommendations:
                cols = st.columns(5)
                for i, game in enumerate(recommendations[:5]):
                    cols[i].markdown(f"**{i + 1}. {game}**")

                cols = st.columns(5)
                for i, game in enumerate(recommendations[5:]):
                    cols[i].markdown(f"**{i + 6}. {game}**")
            else:
                st.warning("Could not find similar games.")


def main():
    """
    Main function to run the Streamlit application.
    """
    # --- Load All Components ---
    hybrid_model, content_model, users_df = load_models_and_data()

    # --- Sidebar Navigation ---
    st.sidebar.title("🎮 Recommender App")
    app_mode = st.sidebar.radio(
        "Choose Recommender Mode",
        ("User-Based Recommendations", "Item-Based (Content) Recommendations")
    )

    # --- Page Rendering ---
    if app_mode == "User-Based Recommendations":
        render_user_based_page(hybrid_model, users_df)
    elif app_mode == "Item-Based (Content) Recommendations":
        render_item_based_page(content_model)


if __name__ == '__main__':
    main()
