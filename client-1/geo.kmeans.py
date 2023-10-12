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
        if filename.endswith('_0.csv'): #Change to all_trip,
        #if filename.startswith('testing_flightsa7124d_0.csv'): #Change to all_trip, 
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

    
    model = TimeSeriesKMeans(n_clusters=2,
                             verbose=False,
                             metric='euclidean', 
                             random_state=2)

    ################ kMeans #################

    # print(f'Cluster - {model.cluster_centers_.shape}')
    # print(f'distance - {model.inertia_}')

    class kMeansClient(fl.client.NumPyClient):
        def __init__(self, model) -> None:
            self.model = model

        def get_parameters(self, config):
            params = []

            if hasattr(self.model, "cluster_centers_"):
                params.append(
                    self.model.cluster_centers_
                )
            return params
        
        def fit(self, parameters, config):
            
            if hasattr(self.model, "cluster_centers_"):
                my_array = np.array(parameters)
                
                print("ARR")
                print(my_array[0].shape)
                print(my_array.shape)
                print(self.model.cluster_centers_.shape)
                
                if hasattr(my_array, "__array__"):
                    # self.model.cluster_centers_ = my_array
                    self.model.init = my_array[0]
                    # self.model.fit(X)
            
            
            self.model.fit(X)
            print(self.model.__dict__)

            return self.get_parameters(self.model), len(X), {}


        # def evaluate(self, parameters, config):
        #     model.set_weights(parameters)
        #     loss, accuracy = model.evaluate(X_valid, y_valid)
        #     return loss, len(X_valid), {"accuracy": accuracy}

    fl.client.start_numpy_client(server_address=SERVER_ADDR, 
                                 client=kMeansClient(model))