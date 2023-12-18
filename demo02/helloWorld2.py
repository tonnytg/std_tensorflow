
# // write in Python 3.6 to calculate number pars
# and use tensorflow to say is it even or odd

import tensorflow as tf

# // create a variable to hold the value of the number
# // we want to test
x = tf.Variable(35, name='x')
y = tf.Variable(5, name='y')

# // create a variable that will perform the multiplication
product = tf.Variable(0, name='product')

# // initialize the variables
model = tf.global_variables_initializer()

# // create a session and run the model
with tf.Session() as session:
    session.run(model)
    session.run(product.assign(tf.multiply(x, y)))
    print(session.run(product))

# // close the session
session.close()
