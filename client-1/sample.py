import tensorflow as tf
import numpy as np

# Create a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Display the initial model weights
initial_weights = model.get_weights()
print("Initial Model Weights:")
for layer_weights in initial_weights:
    print(layer_weights.shape)

# Create a new set of weights (for demonstration purposes)
new_weights = []
for layer_weights in initial_weights:
    new_weights.append(np.random.randn(*layer_weights.shape))

# Set the model weights to the new_weights
model.set_weights(new_weights)

# Verify that the model weights have been updated
updated_weights = model.get_weights()
print("\nUpdated Model Weights:")
for layer_weights in updated_weights:
    print(layer_weights.shape)
