import os
import flwr as fl
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from tslearn.clustering import TimeSeriesKMeans

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
        
    # return np.array(df_X), np.array(df_y)
    return df_X, df_y


if __name__ == "__main__":
    X, y = read_data()

    ################ kMeans #################

    km = TimeSeriesKMeans(n_clusters=1, 
                        verbose=False, 
                        metric='euclidean',
                        random_state=42)
    
    km.fit(X)

    ################ kMeans #################

    # print(f'Cluster - {km.cluster_centers_.shape}')
    # print(f'distance - {km.inertia_}')

    class kMeansClient(fl.client.NumPyClient):
        def __init__(self) -> None:
            self.num_of_round = 0

        def get_parameters(self, config):
            if hasattr(km, "cluster_centers_"):
                return np.array(km.cluster_centers_)

        def fit(self, parameters, config):
            empty_3d = np.empty((2, 2, 3), dtype=complex)
            if hasattr(km, "cluster_centers_"):
                print(parameters)
                # km.cluster_centers_ = parameters
            km.fit(X)
            return [], 0, {}

        # def evaluate(self, parameters, config):
        #     model.set_weights(parameters)
        #     loss, accuracy = model.evaluate(X_valid, y_valid)
        #     return loss, len(X_valid), {"accuracy": accuracy}

    fl.client.start_numpy_client(server_address=SERVER_ADDR, 
                                 client=kMeansClient())