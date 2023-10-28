import os
import flwr as fl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import argparse
import time

print("--------------------")

start_time = time.time()

print(f"START TIME - {start_time}")

parser = argparse.ArgumentParser(description="A simple command-line")


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
parser.add_argument('--epochs', 
                    help='Provide the Epochs Name', 
                    default=20, 
                    type=int,
                    required=False)
parser.add_argument('--out', 
                    help='Provide the Graph Name', 
                    default='out.png', 
                    required=False)
parser.add_argument('--folder', 
                    help='Provide the Dataset folder', 
                    default='one-trip', 
                    type=str,
                    required=False)


args = parser.parse_args()

SERVER_ADDR = f'{args.ip}:{args.port}'
INPUT_SEQ = args.input_seq
FOLDER_LOC = args.folder
EPOCHS = args.epochs
COLUMN_NAME = 'geoaltitude'
CUTOFF_DT = pd.to_datetime('2022-02-26 00:00:00')
OUTPUT_NAME = args.out


temp_loss = []
temp_mape = []


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
    folder_path = os.path.join('.', 'data', 'geoaltitude', f'{FOLDER_LOC}')
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

    num_of_round = 0
    ################## LSTM #################

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
                  loss="mae",
                  metrics=["mape"])

    ################## LSTM #################


    class MyClient(fl.client.NumPyClient):
        def __init__(self, model, num_of_round) -> None:
            self.num_of_round = num_of_round
            self.model = model

        def get_parameters(self, config):
            return self.model.get_weights()

        def fit(self, parameters, config):
            self.num_of_round = self.num_of_round + 1
            print(f"Rounds - {self.num_of_round}")
            self.model.set_weights(parameters)
            history = self.model.fit(X_train, y_train, epochs = EPOCHS)

            # fit_loss = history.history['loss']
            # fit_mape = history.history['mape']

            # sns.set(style="whitegrid")
            # plt.figure(figsize=(10, 5))

            # plt.plot(fit_loss)
            # plt.plot(self.num_of_round)
            # plt.title('Training Loss v/s Num of Rounds')
            # plt.ylabel('Loss (MAE)')
            # plt.xlabel('Number of Rounds')
            # plt.savefig(f'./output/loss_{OUTPUT_NAME}', dpi=300)
            # plt.show(block=False)

            # plt.plot(fit_mape)
            # plt.plot(self.num_of_round)
            # plt.title('Training Loss v/s Num of Rounds')
            # plt.ylabel('Metrics (MAPE)')
            # plt.xlabel('Number of Rounds')
            # plt.savefig(f'./output/mape_{OUTPUT_NAME}', dpi=300)
            # plt.show(block=False)

            return self.model.get_weights(), len(X_train), {}

        def evaluate(self, parameters, config):
            self.model.set_weights(parameters)
            eval_loss, eval_mape = self.model.evaluate(X_test, y_test)
            
            temp_loss.append(eval_loss)
            temp_mape.append(eval_mape)

            print(f"Eval Loss: {eval_loss} and Eval MAPE: {eval_mape}")
            return eval_loss, len(X_test), {"mape": eval_mape}


    fl.client.start_numpy_client(server_address=SERVER_ADDR, 
                                 client=MyClient(model, num_of_round))
    
    print(f'Losses - {temp_loss}')
    print(f'Mape - {temp_mape}')

    end_time = time.time()
    print("--------------------")
    print(f"END TIME - {end_time}")

    print(f"ELAPSED TIME -  {end_time - start_time}")