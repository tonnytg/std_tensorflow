import tensorflow as tf

# Create a function to calculate the product and print the result
def calculate_product(x, y):
    product = tf.multiply(x, y)
    print(f"The product of {x.numpy()} and {y.numpy()} is {product.numpy()}")

# Create variables to hold the values of the numbers we want to test
x = tf.Variable(35, name='x')
y = tf.Variable(5, name='y')

# Calculate and print the product within the default graph
calculate_product(x, y)

# Calculate and print the division of x by y
division_result = tf.divide(x, y)
print(f"The division of {x.numpy()} by {y.numpy()} is {division_result.numpy()}")

# Retest the product calculation
calculate_product(x, y)
