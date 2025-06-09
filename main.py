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

from src.data_preprocessing import load_and_clean_data, standardize_data, split_features_target
from src.models import build_ann
from sklearn.model_selection import train_test_split

data = load_and_clean_data("data/dane_3.csv")
data, scaler = standardize_data(data)
X, y = split_features_target(data)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = build_ann(input_dim=X.shape[1], layers=[256, 64, 32], learning_rate=0.001)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
