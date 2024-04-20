### Demo06

This demo use Celsius and Fahrenheit to train a model to create a formula like `F = C * 1.8 + 32`

This model will return somethin like `F = C * 1.8 + 29.xxxxx` as weight and `32` as bias.

Now we will save this model to easy reload it in the future applications. We use `model.save("model.h5")` to save the model.
Now you can reload the model with `model = tf.keras.models.load_model("model.h5")`

The model will be saved as HDF5 file.
HD5F is a file format designed to store large amounts of data. It is a binary file format. It is more flexible than the CSV format.

Let test:

`python createModel.py`

Than:

`python useModel.py`
