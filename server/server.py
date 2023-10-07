import flwr as fl
import sys
from fed_knn import FedKnn

# strategy = FedKnn(
#     k_neighbor=2,
#     random_state=2,
#     k_metrics='euclidean'
# )

strategy = fl.server.strategy.FedAvg(
    min_available_clients=2,
    min_fit_clients=2
)

server = fl.driver.start_driver(
    # server_address="0.0.0.0:9091",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=None
)