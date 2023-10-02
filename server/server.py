import flwr as fl
import sys

strategy = fl.server.strategy.FedAvg(
    min_available_clients=2,
    min_fit_clients=2
)

server = fl.driver.start_driver(
    # server_address="0.0.0.0:9091",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy
)