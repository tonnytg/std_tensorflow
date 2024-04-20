import tensorflow as tf
import numpy as np

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([l0])
model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
              loss='mean_squared_error',
              metrics=['mse'])

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
model.save("model.h5")
print("Finished training the model and save as model.h5")