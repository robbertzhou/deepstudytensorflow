import numpy as np
import tensorflow as tf

tf.compat.v1.disable_v2_behavior
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
x_vals = np.array([1.,3.,5.,7.,9.])
x_data = tf.compat.v1.placeholder(tf.float32)
m_const = tf.constant(3.)
my_product = tf.compat.v1.multiply(x_data,m_const)
for x_val in x_vals:
    print(sess.run(my_product,feed_dict={x_data:x_val}))

