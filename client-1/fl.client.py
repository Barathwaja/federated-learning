import os

import flwr as fl
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import LSTM, Dense

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SERVER_ADDR = "0.0.0.0:9092"
TRAIN_SIZE = 0.75
# FLIGHT_NUMBER = "0d02a8"

def read_data():
    folder_path = os.path.join('.', 'data')

    dfs = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    
    print("NA Exists:", combined_df.isnull().values.any())
    
    combined_df = combined_df.dropna()
    combined_df = combined_df.reset_index(drop=True)
    
    print("After cleaning:", combined_df.isnull().values.any())

    # combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])

    # # Ignoring YEAR as it's constant

    # combined_df['month'] = combined_df['datetime'].dt.month
    # combined_df['day'] = combined_df['datetime'].dt.day
    # combined_df['minute'] = combined_df['datetime'].dt.minute
    # combined_df['hour'] = combined_df['datetime'].dt.hour

    columns_to_drop = ['row_id', 'datetime', 'flight_trip', 'label', 'Unnamed: 0']

    y = combined_df['label']
    X = combined_df.drop(columns_to_drop, axis=1)

    return X, y

def get_train_test_split(X, y):

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          train_size=TRAIN_SIZE)

    return X_train, X_valid, y_train, y_valid


if __name__ == "__main__":
    X, y = read_data()

    print()
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    print(X_train.head())
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_input_shape = X_train_scaled.shape[1]
    print(X_train_scaled.shape)
    print("Shape - ", X_input_shape)

    model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(X_input_shape, )),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # # LSTM with back sequence
    # model = tf.keras.models.Sequential()
    # # Add an LSTM layer with 64 units and return_sequences=True for intermediate layers
    # model.add(LSTM(64, return_sequences=True, input_shape=(1, 15, 8, 6717)))

    # # Add another LSTM layer with 32 units
    # model.add(LSTM(32))

    # # Add the output layer for binary classification with sigmoid activation
    # model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Nadam(), 
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    class FlightClient(fl.client.NumPyClient):
        def __init__(self) -> None:
            self.num_of_round = 0

        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            self.num_of_round = self.num_of_round + 1
            model.set_weights(parameters)
            history = model.fit(X_train_scaled, y_train, 
                      epochs=10, 
                      shuffle=False,
                      batch_size=32)
            
            print(history.history.keys())

            # summarize history for accuracy
            # plt.plot(history.history['accuracy'])
            # plt.plot(self.num_of_round)
            # plt.title('model accuracy')
            # plt.ylabel('accuracy')
            # plt.xlabel('num of rounds')
            # plt.legend(['train', 'rounds'], loc='upper left')
            # plt.savefig('./line_plot-accuracy.png')
            # plt.show()

            # summarize history for loss
            # plt.plot(history.history['loss'])
            # plt.plot(self.num_of_round)
            # plt.title('model loss')
            # plt.ylabel('loss')
            # plt.xlabel('num_of_rounds')
            # plt.legend(['train', 'test'], loc='upper left')
            # plt.savefig('./line_plot-loss.png')
            # # plt.show()
            return model.get_weights(), len(X_train_scaled), {}

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(X_test_scaled, y_test)
            return loss, len(X_test_scaled), {"accuracy": accuracy}


    # Start Flower client
    fl.client.start_numpy_client(server_address=SERVER_ADDR, 
                                 client=FlightClient())