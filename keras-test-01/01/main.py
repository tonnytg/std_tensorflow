import tensorflow as tf

# Carregar o conjunto de dados MNIST
mnist = tf.keras.datasets.mnist

# Dividir o conjunto de dados em treinamento e teste
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar os dados (transformar os valores de pixel de 0-255 para 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Construir o modelo sequencial
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),  # Transforma a matriz 28x28 em um vetor 1D
  tf.keras.layers.Dense(128, activation='relu'),  # Camada densa com 128 neurônios e ativação ReLU
  tf.keras.layers.Dropout(0.2),                   # Camada de dropout para reduzir overfitting
  tf.keras.layers.Dense(10, activation='softmax') # Camada de saída com 10 neurônios para as classes de dígitos, usando softmax
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train, y_train, epochs=5)  # Treina por 5 épocas

# Avaliar o modelo no conjunto de teste
model.evaluate(x_test, y_test)
