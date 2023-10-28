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

# Add arguments
parser.add_argument('--input_seq', 
                    help='Provide the Time-Series Window Size', 
                    default=4, 
                    required=False)

parser.add_argument('--out', 
                    help='Provide the Graph Name', 
                    default='out.png', 
                    required=False)

parser.add_argument('--epochs', 
                    help='Provide the Epochs Name', 
                    default=20, 
                    type=int,
                    required=False)

parser.add_argument('--folder', 
                    help='Provide the Dataset folder', 
                    default='one-trip', 
                    type=str,
                    required=False)

args = parser.parse_args()

INPUT_SEQ = args.input_seq
EPOCHS = args.epochs
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
    
    print(f"SHAPE - {len(df_X)}")
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
    
    train_df = temp_store[temp_store['datetime'] < CUTOFF_DT]
    test_df = temp_store[temp_store['datetime'] >= CUTOFF_DT]
    return train_df, test_df


if __name__ == "__main__":
    train, test = convert_to_train_test()

    X_train, y_train = read_uni_dataset(train)
    X_test, y_test = read_uni_dataset(test)

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
    
    history = model.fit(X_train, y_train, epochs = EPOCHS)

    eval_loss, eval_mape = model.evaluate(X_test, y_test)
    
    # fit_loss = history.history['loss']
    # fit_mape = history.history['mape']

    # # Create a line plot using Seaborn
    # sns.set(style="whitegrid")
    # plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
    # sns.lineplot(x=range(1, len(fit_loss) + 1), y=fit_loss, label='Training Loss')
    # sns.lineplot(x=range(1, len(fit_mape) + 1), y=fit_mape, label='Training MAPE')

    # plt.title('Training Loss (MAE) and MAPE')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss / MAPE')
    # plt.legend()
    # plt.savefig(f'./output/{OUTPUT_NAME}', dpi=300)

    ################## LSTM #################

    print(f"Eval Loss: {eval_loss} and Eval MAPE: {eval_mape}")

    end_time = time.time()
    print("--------------------")
    print(f"END TIME - {end_time}")

    print(f"ELAPSED TIME -  {end_time - start_time}")
