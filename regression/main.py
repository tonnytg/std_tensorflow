# Importar TensorFlow
import tensorflow as tf

# Criar variáveis para os parâmetros do modelo
W = tf.Variable(tf.random.normal([1]), name="weight")
b = tf.Variable(tf.random.normal([1]), name="bias")

# Placeholder para os dados de entrada
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Construir o modelo de regressão linear
def linear_regression(X):
    return X * W + b

