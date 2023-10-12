import flwr as fl
import sys
import socket
# from fed_knn import FedKnn
# sfrom fedkmeans import FedKMeans
#from fedavg_modified import FedAvg
from fedavg import FedAvg
import time


IP_ADDR = socket.gethostbyname((socket.gethostname()))
PORT = 9092

SERVER_ADDR = "0.0.0.0:9092"
#SERVER_ADDR = f'{IP_ADDR}:{PORT}'

print(f"server addr - {SERVER_ADDR}")

start_time = time.time()

strategy = FedAvg(
    min_available_clients=2,
    min_fit_clients=2
)

# server = fl.driver.start_driver(
#     # server_address="0.0.0.0:9091",
#     config=fl.server.ServerConfig(num_rounds=2),
#     strategy=strategy,
# )

print("--------------------")
print(f"START TIME - {start_time}")


fl.server.start_server(
    server_address=SERVER_ADDR,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)

end_time = time.time()
print("--------------------")
print(f"END TIME - {end_time}")

print(f"ELAPSED TIME -  {end_time - start_time}")