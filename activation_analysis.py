import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -------------------- LOAD DATA --------------------
X = np.load("X_features.npy")

# -------------------- BUILD MODEL (FUNCTIONAL API) --------------------
inputs = tf.keras.Input(shape=(X.shape[1],))
x = tf.keras.layers.Dense(256, activation="relu")(inputs)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# -------------------- ACTIVATION MODEL --------------------
activation_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.layers[1].output
)

# -------------------- GET ACTIVATIONS --------------------
activations = activation_model.predict(X[:50]).flatten()

# -------------------- FIXED ACTIVATION DISTRIBUTION PLOT --------------------
plt.figure(figsize=(7, 4))

# Optional: clip extreme values for better visibility
activations_clipped = np.clip(activations, 0, 50)

plt.hist(
    activations_clipped,
    bins=50,
    log=True,              # 🔹 KEY FIX: log scale
    color="steelblue",
    edgecolor="black"
)

plt.title("Activation Distribution (Dense ReLU Layer)")
plt.xlabel("Activation Value")
plt.ylabel("Log Frequency")
plt.tight_layout()
plt.show()
