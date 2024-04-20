import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i, c in enumerate(celsius_q):
    print("{} degrees Celsius = {} Fahrenheit".format(c, fahrenheit_a[i]))

l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([l0, l1, l2])

model.compile(optimizer=tf.keras.optimizers.Adam(0.1),
              loss='mean_squared_error',
              metrics=['mse'])

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=True)
print("Finished training the model")

if ((model.predict([100.0])) - 212.0) < 0.1:
    print("Model is correct")
else:
    print("Model is incorrect")
    print("Model predicts: {}".format(model.predict([100.0])))
print(model.predict([100.0]))

print("These are the l0 variables: {}".format(l0.get_weights()))
print("These are the l1 variables: {}".format(l1.get_weights()))
print("These are the l2 variables: {}".format(l2.get_weights()))

plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnitude")
plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'])
plt.savefig('loss.png')