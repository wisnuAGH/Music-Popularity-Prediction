# 🎵 Music Popularity Prediction

This project uses **machine learning** to predict the **popularity of a song** based on its audio features. In addition to building and evaluating ML models, it also includes an analysis of how each feature affects the predicted popularity score.

## 🎯 Project Goal

The main goals of the project are:
- To train and evaluate multiple regression models to predict song popularity.
- To select the best-performing model based on validation metrics.
- To generate **feature effect curves**, showing how each individual feature influences the model's predicted popularity (while other features are kept at their average values).

This project is useful for music creators, marketers, and enthusiasts who want to understand what makes a song more likely to succeed based on measurable audio characteristics.

## 🎧 Input Data

The dataset consists of songs and their audio features, including popularity scores on a scale from 0 to 100. Features are similar to those provided by:
- [Tunebat.com](https://tunebat.com/)
- [Spotify Web API](https://developer.spotify.com/documentation/web-api/)

**Important note:**  
The data used in this project was collected before the Spotify data usage policy changes that restricted access to full track popularity data.

### Example Features:
- `danceability` — how suitable a track is for dancing
- `energy` — intensity and activity level
- `tempo` — speed of the track
- `acousticness` — presence of acoustic elements
- `valence` — musical positivity
- `speechiness` — spoken word content
- and others...

## 🧠 Machine Learning Models

The following ML models are tested and compared:
- Linear Regression (Lasso, Ridge, ElasticNet)
- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- Neural Networks (Keras MLP)

Each model is evaluated based on:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

The best model is then used to generate detailed visualizations showing the predicted impact of each feature on song popularity.

## 🛠️ Preprocessing & Feature Engineering
- Duplicate removal
- Outlier detection and filtering
- Feature scaling (Min-Max normalization)
- Feature selection using Lasso and RFE
- Hyperparameter tuning (grid search)

## 📦 Project Structure
music-popularity-prediction/

│

├── data/ # Raw data (CSV files)

├── src/ # Python scripts: preprocessing, models, feature analysis

├── notebooks/ # Exploratory notebooks

├── results/ # Model outputs and plots

├── main.py # End-to-end training script

├── requirements.txt # Project dependencies

└── README.md

## 📌 Planned Extensions

- Export of trained models (`.h5`, `.pkl`)
- Flask/FastAPI endpoint for popularity prediction
- Real-time analysis using Spotify API (given user input)

## 📋 Requirements

Install required libraries with:
> pip install -r requirements.txt


## 👤 Author

Created by `Mikołaj Wiśniewski` as a personal project to explore music data and build interpretable machine learning models in Python.

---

**Disclaimer:** This project is for educational and research purposes only. No commercial usage of Spotify data is involved.
