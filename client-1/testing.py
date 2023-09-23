import os

import matplotlib.pyplot as plt
import flwr as fl

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import warnings

import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

TRAIN_SIZE = 0.75
FLIGHT_NUMBER = "0d02a8"

def read_data():
    folder_path = os.path.join('..','dataset','processed')

    dfs = []

    for filename in os.listdir(folder_path):
        if FLIGHT_NUMBER in filename and filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    columns_to_drop = ['signature_label', 'icao24', 'trip_number', 'Unnamed: 0']

    y = combined_df['signature_label']
    X = combined_df.drop(columns_to_drop, axis=1)

    return X, y

def get_train_test_split(X, y):

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          train_size=TRAIN_SIZE)

    return X_train, X_valid, y_train, y_valid

if __name__ == "__main__":
    X, y = read_data()

    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Create LogisticRegression Model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print(model.get_params())