import os
import flwr as fl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import argparse
from tslearn.clustering import TimeSeriesKMeans
import time

print("--------------------")

start_time = time.time()

print(f"START TIME - {start_time}")


parser = argparse.ArgumentParser(description="A simple command-line")

# Add arguments
parser.add_argument('--input_seq', 
                    help='Provide the Time-Series Window Size', 
                    default=4, 
                    required=False)

parser.add_argument('--out', 
                    help='Provide the Graph Name', 
                    default='out.png', 
                    required=False)

parser.add_argument('--num_clusters', 
                    help='Provide the Number of Clusters', 
                    default=2, 
                    type=int,
                    required=False)

parser.add_argument('--folder', 
                    help='Provide the Dataset folder', 
                    default='one-trip', 
                    type=str,
                    required=False)

args = parser.parse_args()

INPUT_SEQ = args.input_seq
NUM_CLUSTERS = args.num_clusters
FOLDER_LOC = args.folder
COLUMN_NAME = 'geoaltitude'
CUTOFF_DT = pd.to_datetime('2022-02-26 00:00:00')
OUTPUT_NAME = args.out

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
    folder_path = os.path.join('.', 'data', 'geoaltitude', f'{FOLDER_LOC}')
    print(folder_path)
    temp_store = pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
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

    model = TimeSeriesKMeans(n_clusters=NUM_CLUSTERS,
                             verbose=False,
                             random_state=0)
    
    model.fit(X_train, y_train)

    get_cluster_labels_ = model.predict(X_test)
    absolute_percentage_errors = np.abs((y_test - model.cluster_centers_[get_cluster_labels_]) / y_test)
    mape_ = np.mean(absolute_percentage_errors) * 100

    print(f"Eval MAPE - {mape_}")
    
    # test_data = np.concatenate((X_test, model.predict(X_test).reshape(-1, 1)), axis=1)

    # # # Create a line plot to visualize the clusters for training and test data
    # # sns.set()
    # # plt.figure(figsize=(20, 6))

    # # # Iterate through the clusters and plot the time series data for test data
    # # for cluster_id in range(NUM_CLUSTERS):
    # #     cluster_data = test_data[test_data[:, -1] == cluster_id, :-1]
    # #     for ts in cluster_data:
    # #         plt.plot(ts, alpha=0.6, linestyle='--', label=f'Test Cluster {cluster_id}')

    # # plt.xlabel('Time')
    # # plt.ylabel('Value')
    # # plt.title('Time Series Clustering')
    # # plt.legend()
    # # plt.savefig(f'./output/{OUTPUT_NAME}', dpi=300)
    
    ################ kMeans #################

    end_time = time.time()
    print("--------------------")
    print(f"END TIME - {end_time}")

    print(f"ELAPSED TIME -  {end_time - start_time}")