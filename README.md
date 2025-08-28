# Steam Game Recommender System

This project is a comprehensive exploration of building and evaluating various recommender system algorithms. Using the [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) dataset from Kaggle, the project progresses from simple baselines to a sophisticated hybrid model, culminating in an interactive web application built with Streamlit.

## Features

-   **Interactive Web Application:** A user-friendly interface built with Streamlit to explore recommendations.
-   **Dual Recommendation Modes:**
    -   **User-Based Recommendations:** A hybrid model provides personalized game recommendations for a selected user, balancing personal taste with game quality.
    -   **Item-Based Recommendations:** A content-based model finds games similar to a selected game based on textual metadata (tags and descriptions).
-   **Multiple Models:** Implementation and evaluation of three distinct recommender models.

---

## Models Implemented

This project follows an iterative development process, building and evaluating three types of models:

1.  **Content-Based Filtering:** This model uses **TF-IDF** vectorization on game tags and descriptions to calculate **Cosine Similarity** between games. It recommends items that are textually similar to a given input item.

2.  **Collaborative Filtering (ALS):** This model employs the **Alternating Least Squares (ALS)** algorithm from the `implicit` library. It learns latent features for users and games from implicit feedback signals (a "confidence score" derived from playtime and review helpfulness) to provide highly personalized recommendations.

3.  **Simple Hybrid Model:** The final model is a two-stage "filter" hybrid. It uses the powerful **ALS model** for personalized candidate generation and then **filters the results** based on a global game quality score (`positive_ratio`) to ensure the final recommendations are both relevant and of high quality.

---

## Setup and Installation

To run this project locally, please follow these steps.

### 1. Clone the Repository
```bash
git clone https://github.com/dragosneag/games-recommender.git
cd games-recommender
```

### 2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
This project uses a `requirements.txt` file to manage all necessary libraries.
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
-   Download the dataset from [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam).
-   Unzip the file and place the CSV files (`games.csv`, `users.csv`, `recommendations.csv`) and the `games_metadata.json` file inside a `data` folder in the root of the project directory.

The final project structure should look like this:
```
your-repository-name/
├── data/
│   ├── games.csv
│   ├── users.csv
│   ├── recommendations.csv
│   └── games_metadata.json
├── app.py
├── train.py
├── recommender_model.py
├── content_model.py
├── ... (other .py files)
└── requirements.txt
```

---

## How to Run

Running the application is a two-step process:

### 1. Train the Models
First, you need to run the training script. This will train both the ALS and Content-Based models and save them to a `models.pkl` file for the app to use.
```bash
python train.py
```
This process may take a few minutes.

### 2. Launch the Streamlit App
Once the models are trained and saved, you can launch the interactive web application.
```bash
streamlit run app.py
```
This will open a new tab in your web browser with the recommender system interface.