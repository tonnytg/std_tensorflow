import tensorflow as tf
import numpy as np

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,)),
])

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

model.compile(optimizer=custom_optimizer, loss='mean_squared_error')

model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)

# Salve o modelo usando SavedModel
model.save(
    "model_savedmodel",
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

print("Finished training the model and saved as model_savedmodel")
