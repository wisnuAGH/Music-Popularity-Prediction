"""
-------------------------------------------------
File Name: main.py
Author: Mikołaj Wiśniewski
Created: 2025-06-09
Version: 2.0

Description: main file

Dependencies:
- Python 3.10+
- Required libraries: sklearn
-------------------------------------------------
"""
import joblib
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
from src.visualization import plot_learning_curves

# === 1. Load and preprocess data ===
data = load_and_clean_data("data/data.csv")
data = remove_outliers(data)  # usuwa odstające wartości tylko raz
data, scaler = standardize_data(data)  # tylko jedna standaryzacja

# === 2. Split features and target ===
X, y = split_features_target(data)

# === 3. Feature selection (optional: RFE) ===
X_selected = X  # lepsze wyniki bez selekcji

# === 4. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# === 5. Hyperparameter tuning (returns best config) ===
# pass

# === 6. Build and train final model ===
model = build_ann(input_dim=X_selected.shape[1])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)

history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping, reduce_lr])

# === 7. Evaluate model ===
evaluate_model(model, X_test, y_test)

# === 8. Visualize learning curves ===
plot_learning_curves(history, save_path="results/learning_curves.png")

# === 9. Evaluate save ===
model.save("model/popularity_model.keras")
joblib.dump(scaler, "model/scaler.pkl")
