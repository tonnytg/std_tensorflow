import tensorflow as tf

def loadModel(path):
    return tf.keras.models.load_model(path)

model = loadModel("model.h5")
print("Loaded model from disk")

result = model.predict([100.0])
print("100 degrees Celsius = {} Fahrenheit".format(result[0][0]))