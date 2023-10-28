import os
import flwr as fl
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


SERVER_ADDR = "0.0.0.0:9092"
TRAIN_SIZE = 0.75

def read_data():
    folder_path = os.path.join('.', 'data')
    constant_value = 2
    dfs = []

    for filename in os.listdir(folder_path):
        if filename.endswith('all_trips_m2.csv'):
            file_path = os.path.join(folder_path, filename)
            print(file_path)
            df = pd.read_csv(file_path)

            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # columns_to_drop = ['Unnamed: 0', 'mean_velocity']

    y = combined_df['mean_geoaltitude']

    X = combined_df['window']

    return X, y

if __name__ == "__main__":
    X, y = read_data()

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          train_size=TRAIN_SIZE)

    # Define a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(4),
        tf.keras.layers.Dense(1)
        ])

    # Compile the model
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=tf.keras.metrics.MeanSquaredError())

    class MyClient(fl.client.NumPyClient):
        def __init__(self) -> None:
            self.num_of_round = 0

        def get_parameters(self, config):
            return model.get_weights()

        # def set_parameters(self, parameters):
        #     print(parameters)
        #     model.set_weights(parameters)

        def fit(self, parameters, config):
            print(f"ROUND - {self.num_of_round}")
            self.num_of_round = self.num_of_round + 1
            print('-' * 3)
            print(f"Param - {parameters}")
            print('-' * 3)

            model.set_weights(parameters)
            model.fit(X_train, y_train, epochs=1)
            return model.get_weights(), len(X), {}

        def evaluate(self, parameters, config):
            print("evaluate")
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(X_valid, y_valid)
            return loss, len(X_valid), {"accuracy": accuracy}

    fl.client.start_numpy_client(server_address=SERVER_ADDR, 
                                 client=MyClient())