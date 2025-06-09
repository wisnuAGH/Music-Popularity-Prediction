"""
-------------------------------------------------
File Name: data_preprocessing.py
Author: Mikołaj Wiśniewski
Created: 2025-06-09
Version: 1.0

Description: Preprocessing input data
1. drop duplicates
2. drop useless columns
3. standardize
4. split features X Y

Dependencies:
- Python 3.10+
- Required libraries: sklearn, pandas
-------------------------------------------------
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_and_clean_data(path):
    data = pd.read_csv(path)
    data = data.drop(columns=['streaming_platform', 'key', 'mode'], errors='ignore')
    data = data.drop_duplicates()
    return data


def standardize_data(data):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns), scaler


def split_features_target(data, target_col='popularity'):
    X = data.drop(columns=[target_col])
    y = data[target_col]
    return X, y
