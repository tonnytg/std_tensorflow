import os
import tensorflow as tf

# Configuração para minimizar mensagens de aviso do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Definição de tensores de entrada
X = tf.constant([[2, 2, 2], [1, 1, 1]], dtype=tf.float32, name="X")
Y = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name="Y")

# Operação de adição
addition = tf.add(X, Y, name="addition")

# Execução da operação (no modo eager do TensorFlow 2.x)
result = addition.numpy()

# Impressão do resultado
print(result)
