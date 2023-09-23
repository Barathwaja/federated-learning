import flwr as fl
# import tensorflow as tf


strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2
    )

fl.server.start_server(
    server_address="127.0.0.1:12000",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)