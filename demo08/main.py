import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the CSV file
data = pd.read_csv('file.csv')

# Convert the 'region' column into numerical values using one-hot encoding
data = pd.get_dummies(data, columns=['region'])

# Split the data into features (X) and labels (y)
X = data.drop('crime', axis=1).values
y = data['crime'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Loss on the test set: {loss}')

# Make predictions
predictions = model.predict(X_test)

print(predictions)
