import flwr as fl
import matplotlib.pyplot as plt
import sys

# Fed Averaging
# strategy = fl.server.strategy.FedAvg(
#     min_fit_clients=2
#     )

strategy = fl.server.strategy.FedAvg(
    min_available_clients=4,
    min_fit_clients=4,
    min_evaluate_clients=4
)

server = fl.driver.start_driver(
    # server_address="0.0.0.0:9091",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)

# server = fl.server.start_server(
#     server_address="0.0.0.0:9092",
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=strategy
# )

print(server.losses_distributed)


x_values = [point[0] for point in server.losses_distributed]
y_values = [point[1] for point in server.losses_distributed]

# Create the plot
plt.plot(x_values, y_values, marker='o', linestyle='-')

# Add labels and title
plt.xlabel('Num. of rounds')
plt.ylabel('Loss')
plt.title('Plot of Rounds vs Loss - FedAvg')

# Save the plot as a PNG file
plt.savefig('./line_plot-avg.png')

# Show the plot
# plt.show()
sys.exit()