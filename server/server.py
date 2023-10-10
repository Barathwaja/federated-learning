import flwr as fl
import sys
import socket
# from fed_knn import FedKnn
# from fedkmeans import FedKMeans
# from fedavg_modified import FedAvg
from fedavg import FedAvg



IP_ADDR = socket.gethostbyname((socket.gethostname()))
PORT = 9092

SERVER_ADDR = f'{IP_ADDR}:{PORT}'

strategy = FedAvg(
    min_available_clients=2,
    min_fit_clients=2
)

# server = fl.driver.start_driver(
#     # server_address="0.0.0.0:9091",
#     config=fl.server.ServerConfig(num_rounds=2),
#     strategy=strategy,
# )

fl.server.start_server(
    server_address=SERVER_ADDR,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)