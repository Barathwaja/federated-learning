import os
import flwr as fl
import numpy as np
import pandas as pd

import tensorflow as tf
import argparse


parser = argparse.ArgumentParser(description="A simple command-line")

# Add arguments
parser.add_argument('--ip', 
                    help='Provide the IP address', 
                    default="0.0.0.0", 
                    required=False)
parser.add_argument('--port', 
                    help='Provide the Port address', 
                    default="9092", 
                    required=False)
parser.add_argument('--input_seq', 
                    help='Provide the Time-Series Window Size', 
                    default=4, 
                    required=False)

args = parser.parse_args()

SERVER_ADDR = f'{args.ip}:{args.port}'
INPUT_SEQ = args.input_seq
COLUMN_NAME = 'geoaltitude'
CUTOFF_DT = pd.to_datetime('2022-02-26 00:00:00')
FLIGHT_ICAO = 'adc0fb'

def read_uni_dataset(dataf):

    dataf = dataf.dropna(subset=[COLUMN_NAME])

    df_np = dataf[COLUMN_NAME].to_numpy()
    df_X = []
    df_y = []
    
    for i in range(len(df_np) - INPUT_SEQ):
        row = [a for a in df_np[i:i + INPUT_SEQ]]
        df_X.append(row)
        label = df_np[i + INPUT_SEQ]
        df_y.append(label)
    
    print(f"SHAPE - {len(df_X)}")
    return np.array(df_X), np.array(df_y)


def convert_to_train_test():
    folder_path = os.path.join('.', 'data', 'geoaltitude')
    temp_store = pd.DataFrame()

    for filename in os.listdir(folder_path):
        #if filename.endswith('_0.csv'): #ONE TRIP,
        if filename.startswith(f'{FLIGHT_ICAO}.csv'): #Change to all_trip, 
            file_path = os.path.join(folder_path, filename)
            # print(f"File Path - {file_path}")
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            temp_store = df
    
    # print(temp_store.head())
    train_df = temp_store[temp_store['datetime'] < CUTOFF_DT]
    test_df = temp_store[temp_store['datetime'] >= CUTOFF_DT]
    return train_df, test_df


if __name__ == "__main__":
    train, test = convert_to_train_test()

    X_train, y_train = read_uni_dataset(train)
    X_test, y_test = read_uni_dataset(test)

    # ################ LSTM #################

    tf.random.set_seed(0)

    inputs = tf.keras.layers.Input(shape=(X_train[0].shape[0]))
    layer_inp = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)

    layer_inp = tf.keras.layers.LSTM(10, activation="relu")(layer_inp)
    output = tf.keras.layers.Dense(1)(layer_inp)

    model = tf.keras.Model(inputs=inputs,
                            outputs=output,
                            name="model_lstm")
    
    model.summary()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.2),
                    loss="mse",
                    metrics=["mape"])

    # ################ LSTM #################


    class MyClient(fl.client.NumPyClient):
        def __init__(self) -> None:
            self.num_of_round = 0

        def get_parameters(self, config):
            print("params")
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            model.fit(X_train, y_train, epochs = 10)
            return model.get_weights(), len(X_train), {}

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, mape_ = model.evaluate(X_test, y_test)
            return loss, len(X_test), {"mape": mape_}


    fl.client.start_numpy_client(server_address=SERVER_ADDR, 
                                 client=MyClient())