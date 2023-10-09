import os
import flwr as fl
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras import Sequential
from keras.layers import Lambda, LSTM, Dense


from keras.models import Sequential, save_model, load_model
from keras.layers import *
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam 



SERVER_ADDR = "0.0.0.0:9092"
TRAIN_SIZE = 0.75
NUM_OF_SAMPLES = 0
INPUT_SEQ = 4

def read_data():
    global NUM_OF_SAMPLES
    folder_path = os.path.join('.', 'data', 'geoaltitude')

    dfs = []

    for filename in os.listdir(folder_path):
        # if filename.endswith('one_trip.csv'): #Change to all_trip,
        if filename.startswith('testing_flights0d02a8_0.csv'): #Change to all_trip, 
            file_path = os.path.join(folder_path, filename)
            print(file_path)
            df = pd.read_csv(file_path)

            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Drop NaN
    combined_df = combined_df.dropna(subset=['mean_geoaltitude'])

    df_np = combined_df['mean_geoaltitude'].to_numpy()
    df_X = []
    df_y = []
    
    for i in range(len(df_np) - INPUT_SEQ):
        row = [a for a in df_np[i:i + INPUT_SEQ]]
        df_X.append(row)
        label = df_np[i + INPUT_SEQ]
        df_y.append(label)
        
    return np.array(df_X), np.array(df_y)


if __name__ == "__main__":
    X, y = read_data()

    ################ LSTM #################
    
    model = Sequential()

    n_features = 1              # number of input variables used for forecast (here, only 1 i.e. temperature)
    
    model.add(InputLayer((INPUT_SEQ, n_features)))
    model.add(LSTM(10, return_sequences = True))
    # model1.add(LSTM(50))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1))

    model.summary()

    model.compile(loss = MeanSquaredError(), 
                    optimizer = Adam(learning_rate = 10), 
                    metrics = RootMeanSquaredError())

    ################ LSTM #################


    class MyClient(fl.client.NumPyClient):
        def __init__(self) -> None:
            self.num_of_round = 0

        def get_parameters(self, config):
            print("params")
            return model.get_weights()

        def fit(self, parameters, config):
            # print(f"ROUND - {self.num_of_round}")
            # self.num_of_round = self.num_of_round + 1
            # print('-' * 3)
            # print(f"Param - {parameters}")
            # print('-' * 3)
            print("fit")

            model.set_weights(parameters)
            model.fit(X, y,
                epochs = 100, )
            return model.get_weights(), len(X), {}

        # def evaluate(self, parameters, config):
        #     model.set_weights(parameters)
        #     loss, accuracy = model.evaluate(X_valid, y_valid)
        #     return loss, len(X_valid), {"accuracy": accuracy}

    fl.client.start_numpy_client(server_address=SERVER_ADDR, 
                                 client=MyClient())