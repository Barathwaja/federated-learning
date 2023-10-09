import flwr as fl
import sys
from fed_knn import FedKnn
from fedkmeans import FedKMeans

from fedavg_modified import FedAvg
# from fedavg import FedAvg

# strategy = FedKnn(
#     k_neighbor=2,
#     random_state=2,
#     k_metrics='euclidean'
# )

strategy = FedAvg(
    min_available_clients=2,
    min_fit_clients=2
)

server = fl.driver.start_driver(
    # server_address="0.0.0.0:9091",
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=strategy,
)

# fl.server.start_server(
#     server_address="0.0.0.0:9091",
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=strategy,
# )