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

SERVER_ADDR = "127.0.0.1:12000"
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


# def get_rf_parameters(rf_model):
#     # rf_params = rf_model.get_params()
#     # print(rf_params)
    
#     rf_params = {'bootstrap': True, 'max_features':'sqrt', 'n_estimators':100}
#     params_list = []
    
#     for param, value in rf_params.items():
#         arr = np.array([value])
#         params_list.append(arr)
#     print(params_list)
#     print(type(params_list))
#     print(type(params_list[0]))

#     return params_list

def get_rf_parameters(rf_model):
    print(rf_model.estimator_)
    rf_model.estimators_ = np.array([None] * rf_model.n_estimators, dtype=object)

    return rf_model

    # if hasattr(rf_model, 'estimators_'):
    #     print(rf_model.estimators_)
    #     trees = [tree for tree in rf_model.estimators_ if tree is not None]
    #     return [trees]

    # rf_params = {'bootstrap': True, 'n_estimators': 100}
    # print(rf_params)
    # params_list = []
    # for param, value in rf_params.items():
    #     arr = np.array([value])
    #     params_list.append(arr)
    # print(params_list)
    # print(type(params_list))
    # print(type(params_list[0]))
    # return params_list


if __name__ == "__main__":
    X, y = read_data()

    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)

    class FlightClient(fl.client.NumPyClient):
        
        def get_parameters(self, config):
            return get_rf_parameters(model)

        def fit(self, parameters, config):
            model.fit(X_train_scaled, y_train)
            return get_rf_parameters(model), len(X_train_scaled), {}

        # def evaluate(self, parameters, config):
        #     loss = log_loss(y_test, model.predict_proba(X_test_scaled))
        #     accuracy = model.score(X_test_scaled, y_test)
        #     return loss, len(X_test_scaled), {"accuracy": accuracy}

    fl.client.start_numpy_client(server_address=SERVER_ADDR,
                                 client=FlightClient())