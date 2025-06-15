from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

def tune_hyperparameters_ann(X, y):
    # Best hyperparameters base on notebook tests
    return {
        'layer1': 256,
        'layer2': 64,
        'layer3': 16,
        'lr': 0.0005,
        'epochs': 100,
        'batch_size': 32
    }
