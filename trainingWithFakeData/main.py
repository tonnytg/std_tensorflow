import tensorflow as tf
import numpy as np

# Gera dados de exemplo
x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
y_train = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)

# Define o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compila o modelo
model.compile(optimizer='sgd', loss='mean_squared_error')

# Treina o modelo
print("Treinando o modelo...")
history = model.fit(x_train, y_train, epochs=500, verbose=0)

# Faz previsões
print("Previsão para x=6.0:")
print(model.predict([6.0]))
