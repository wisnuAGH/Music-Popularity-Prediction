"""
-------------------------------------------------
File Name: models.py
Author: Mikołaj Wiśniewski
Created: 2025-06-09
Version: 1.0

Description: Neural network model + Lasso regression.

Dependencies:
- Python 3.10+
- Required libraries: tensorflow
-------------------------------------------------
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def build_ann(input_dim, layers, learning_rate):
    model = Sequential()
    model.add(Dense(layers[0], activation='relu', input_shape=(input_dim,)))
    for units in layers[1:]:
        if units:
            model.add(Dense(units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model
