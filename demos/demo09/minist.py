import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Carregar o dataset MNIST
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalizar os valores das imagens de 0-255 para 0-1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Visualizar alguns exemplos do dataset
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()

# Construir o modelo
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),      # Entrada de 28x28 pixels
    layers.Dense(128, activation='relu'),      # Camada oculta com 128 neurônios
    layers.Dense(10, activation='softmax')     # Camada de saída com 10 neurônios (10 classes)
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
model.fit(train_images, train_labels, epochs=5)

# Avaliar a precisão do modelo
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nAcurácia no teste:', test_acc)

# Fazer previsões
predictions = model.predict(test_images)

# Visualizar algumas previsões
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"Pred: {predictions[i].argmax()}, True: {test_labels[i]}")
plt.show()
