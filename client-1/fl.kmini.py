import os
import flwr as fl
import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans

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
                    type=str,
                    required=False)
parser.add_argument('--clusters', 
                    help='Provide the Clusters', 
                    default=2, 
                    required=False)

args = parser.parse_args()

SERVER_ADDR = f'{args.ip}:{args.port}'
INPUT_SEQ = args.input_seq
NUM_CLUSTERS = args.clusters
COLUMN_NAME = 'geoaltitude'
CUTOFF_DT = pd.to_datetime('2022-02-26 00:00:00')
FLIGHT_ICAO = 'ad564c'

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
    return np.array(df_X), np.array(df_y)


def convert_to_train_test():
    folder_path = os.path.join('.', 'data', 'geoaltitude')
    temp_store = pd.DataFrame()

    for filename in os.listdir(folder_path):
        #if filename.endswith('_0.csv'): #ONE TRIP,
        if filename.startswith(f'{FLIGHT_ICAO}.csv'):
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

    ################ kMeans #################

    model = MiniBatchKMeans(n_clusters=1,
                             verbose=False,
                             random_state=0)
    
    ################ kMeans #################

    class kMeansClient(fl.client.NumPyClient):
        def __init__(self, model) -> None:
            self.model = model

        def get_parameters(self, config):
            compute_param = []

            if hasattr(self.model, "cluster_centers_"):
                compute_param.append(
                    self.model.cluster_centers_
                )
            return compute_param
        
        def fit(self, parameters, config):
            
            if hasattr(self.model, "cluster_centers_"):
                get_compute_param = np.array(parameters)
                
                if hasattr(get_compute_param, "__array__"):
                    self.model.init = get_compute_param[0]

            model.fit(X_train, y_train) #y_train = Ignored

            return self.get_parameters(self.model), len(X_train), {}


        def evaluate(self, parameters, config):
            if hasattr(self.model, "cluster_centers_"):
                get_compute_param = np.array(parameters)
                
                if hasattr(get_compute_param, "__array__"):
                    self.model.init = get_compute_param[0]
            
            get_cluster_labels_ = model.predict(X_test)
            absolute_percentage_errors = np.abs((y_test - model.cluster_centers_[get_cluster_labels_]) / y_test)
            mape_ = np.mean(absolute_percentage_errors) * 100

            return float(0), len(X_test), {"mape": mape_}


    fl.client.start_numpy_client(server_address=SERVER_ADDR, 
                                 client=kMeansClient(model))