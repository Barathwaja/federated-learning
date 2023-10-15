import flwr as fl
import socket
# from fed_knn import FedKnn
# sfrom fedkmeans import FedKMeans
#from fedavg_modified import FedAvg
from fedavg import FedAvg
import time


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
                    default=socket.gethostbyname((socket.gethostname())), 
                    required=False)
parser.add_argument('--port', 
                    help='Provide the Port address', 
                    default="9092", 
                    required=False)
parser.add_argument('--num_rounds', 
                    help='Provide the Time-Series Window Size', 
                    default=3, 
                    required=False)

args = parser.parse_args()

SERVER_ADDR = f'{args.ip}:{args.port}'
NUM_ROUNDS = args.num_rounds



SERVER_ADDR = f'{args.ip}:{args.port}'

print(f"server addr - {SERVER_ADDR}")

start_time = time.time()

strategy = FedAvg(
    min_available_clients=2,
    min_fit_clients=2
)

print("--------------------")
print(f"START TIME - {start_time}")


fl.server.start_server(
    server_address=SERVER_ADDR,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)

end_time = time.time()
print("--------------------")
print(f"END TIME - {end_time}")

print(f"ELAPSED TIME -  {end_time - start_time}")