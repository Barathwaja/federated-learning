import os

import flwr as fl
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

SERVER_ADDR = "0.0.0.0:9092"
TRAIN_SIZE = 0.75
FLIGHT_NUMBER = "0d02a8"

folder_path = './data'


def get_train_test_split(client1: str):
    client1_df = pd.DataFrame()

    all_files = os.listdir(folder_path)

    # files_not_in_trip_numbers = [file for file in all_files if not any((flight_icao_num + "_" + str(trip) + ".csv") in file for trip in trip_numbers)]
    # files_in_trip_numbers = [file for file in all_files if any((flight_icao_num + "_" + str(trip) + ".csv") in file for trip in trip_numbers)]

    files_in_trip_numbers = [file for file in all_files if client1 in file]

    # print(files_in_trip_numbers)
    # print(files_not_in_trip_numbers)

    files_in_trip_numbers = sorted(files_in_trip_numbers, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    # Client-1
    for client1_file in files_in_trip_numbers:
        file_path = os.path.join(folder_path, client1_file)

        print("TRAIN", file_path)

        df = pd.read_csv(file_path)
        client1_df = pd.concat([client1_df, df], ignore_index=True)

    client1_df = client1_df.sort_values(by='time', ascending=True)

    drop_columns = ['Unnamed: 0', 'icao24']
    client1_df = client1_df.drop(drop_columns, axis=1)

    y_client1 = client1_df['signature_label']
    X_client1 = client1_df.drop('signature_label', axis=1)

    return X_client1, y_client1


if __name__ == "__main__":
    X, y = get_train_test_split(client1='a007c6')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    # X_test_scaled = scaler.transform(X_test)
    
    X_input_shape = X_train_scaled.shape[1]
    print("Shape - ", X_input_shape)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_input_shape, )),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(loss='binary_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=3), 
                  metrics=[tf.keras.metrics.Accuracy()])

    class FlightClient(fl.client.NumPyClient):
        def __init__(self) -> None:
            self.num_of_round = 0

        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            self.num_of_round = self.num_of_round + 1

            print(self.num_of_round)
            model.set_weights(parameters)
            history = model.fit(X_train_scaled, y, 
                      epochs=10, 
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

            # # summarize history for loss
            # plt.plot(history.history['loss'])
            # plt.plot(self.num_of_round)
            # plt.title('model loss')
            # plt.ylabel('loss')
            # plt.xlabel('num_of_rounds')
            # plt.legend(['train', 'test'], loc='upper left')
            # plt.savefig('./line_plot-loss.png')
            # # plt.show()
            return model.get_weights(), len(X_train_scaled), {}

        # def evaluate(self, parameters, config):
        #     model.set_weights(parameters)
        #     loss, accuracy = model.evaluate(X_test_scaled, y_test)
        #     return loss, len(X_test_scaled), {"accuracy": accuracy}


    # Start Flower client
    fl.client.start_numpy_client(server_address=SERVER_ADDR, 
                                 client=FlightClient())