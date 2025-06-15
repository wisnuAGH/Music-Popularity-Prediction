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


def load_data(path):
    data = pd.read_csv(path)
    return data


def filter_rap_genre(data):
    return data[data['playlist_genre'].str.lower() == 'rap']


def clean_data(data):
    data = data.drop(columns=[
        'track_name',
        'track_artist',
        'playlist_genre',
        'playlist_subgenre',
        'key',
        'mode',
        'loudness'  # cecha silnie skorelowana
    ], errors='ignore')
    data = data.drop_duplicates()
    return data


def standardize_data(data):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns), scaler


def split_features_target(data, target_col='track_popularity'):
    X = data.drop(columns=[target_col])
    y = data[target_col]
    print("\nData length:", len(X))
    return X, y


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes outliers from a DataFrame using the IQR (interquartile range) method.
    Applies only to numeric columns.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    df_filtered = df.copy()

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_filtered = df_filtered[
            (df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)
        ]

    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered
