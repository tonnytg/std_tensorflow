import tensorflow as tf

# Carregue o modelo
modelo_carregado = tf.keras.models.load_model("model_savedmodel")

# Faça previsões com o modelo carregado
resultado = modelo_carregado.predict([100.0])
print("100 degrees Celsius =", resultado[0][0], "Fahrenheit")
