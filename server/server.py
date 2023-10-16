import flwr as fl
import socket
import argparse
# from fedkmeans import FedKMeans
from fedavg import FedAvg
import time


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
                    type=int,
                    required=False)
parser.add_argument('--fit_clients', 
                    help='Provide the Fit Clients', 
                    default=2, 
                    type=int,
                    required=False)

args = parser.parse_args()

SERVER_ADDR = f'{args.ip}:{args.port}'
NUM_ROUNDS = args.num_rounds
SERVER_ADDR = f'{args.ip}:{args.port}'
NUM_CLIENTS = args.fit_clients

print(f"server addr - {SERVER_ADDR}")

start_time = time.time()

strategy = FedAvg(
    min_available_clients=NUM_CLIENTS,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS
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