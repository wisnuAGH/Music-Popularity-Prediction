"""
-------------------------------------------------
File Name: main.py
Author: Mikołaj Wiśniewski
Created: 2025-06-09
Version: 1.0

Description: main file

Dependencies:
- Python 3.10+
- Required libraries: sklearn
-------------------------------------------------
"""
import joblib
from sklearn.model_selection import train_test_split

from src.data_preprocessing import (
    load_and_clean_data,
    remove_outliers,
    standardize_data,
    split_features_target
)
from src.feature_selection import select_top_features_rfe
from src.models import build_ann
from src.hyperparameter_tuning import tune_hyperparameters_ann
from src.utils import evaluate_model

# === 1. Load and preprocess data ===
data = load_and_clean_data("data/data.csv")
data = remove_outliers(data)  # usuwa odstające wartości tylko raz
data, scaler = standardize_data(data)  # tylko jedna standaryzacja

# === 2. Split features and target ===
X, y = split_features_target(data)

# === 3. Feature selection (optional: RFE) ===
X_selected = select_top_features_rfe(X, y, n_features=5)

# === 4. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# === 5. Hyperparameter tuning (returns best config) ===
best_params = tune_hyperparameters_ann(X_train, y_train)

# === 6. Build and train final model ===
model = build_ann(input_dim=X_selected.shape[1],
                  layers=[best_params['layer1'],
                          best_params['layer2'],
                          best_params['layer3']],
                  learning_rate=best_params['lr'])

model.fit(X_train, y_train,
          epochs=best_params['epochs'],
          batch_size=best_params['batch_size'],
          validation_split=0.2)

# === 7. Evaluate model ===
evaluate_model(model, X_test, y_test)

# === 8. Evaluate save ===
model.save("model/popularity_model.h5")
joblib.dump(scaler, "model/scaler.pkl")
