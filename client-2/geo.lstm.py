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
#SERVER_ADDR = "192.0.0.101:9092"
TRAIN_SIZE = 0.75
NUM_OF_SAMPLES = 0
INPUT_SEQ = 4
COLUMN_NAME = 'geoaltitude'

def read_data():
    global NUM_OF_SAMPLES
    folder_path = os.path.join('.', 'data', 'geoaltitude')

    dfs = []

    for filename in os.listdir(folder_path):
        #if filename.endswith('_0.csv'): #ONE TRIP,
        if filename.startswith('adc0fb.csv'): #Change to all_trip, 
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Drop NaN
    combined_df = combined_df.dropna(subset=[COLUMN_NAME])

    df_np = combined_df[COLUMN_NAME].to_numpy()
    df_X = []
    df_y = []
    
    for i in range(len(df_np) - INPUT_SEQ):
        row = [a for a in df_np[i:i + INPUT_SEQ]]
        df_X.append(row)
        label = df_np[i + INPUT_SEQ]
        df_y.append(label)
    
    print(f"SHAPE - {len(df_X)}")
    return np.array(df_X), np.array(df_y)


if __name__ == "__main__":
    X, y = read_data()

    ################ LSTM #################
    
    model = Sequential()

    n_features = 1
    
    model.add(InputLayer((INPUT_SEQ, n_features)))
    model.add(LSTM(10, return_sequences = True))
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
            print(f"Round - {self.num_of_round}")
            self.num_of_round = self.num_of_round + 1
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