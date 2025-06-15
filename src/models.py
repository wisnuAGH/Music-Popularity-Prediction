"""
-------------------------------------------------
File Name: models.py
Author: Mikołaj Wiśniewski
Created: 2025-06-09
Version: 1.0

Description: Best model -> Neural network model + Lasso regression.

Dependencies:
- Python 3.10+
- Required libraries: tensorflow
-------------------------------------------------
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam


def build_ann(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(1)  # regresja, bez aktywacji
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    return model

