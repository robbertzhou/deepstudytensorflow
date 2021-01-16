# encoding:utf-8
#两个矩阵乘法

import numpy as np
from tensorflow import compat as cp
import tensorflow as tf

cp.v1.disable_v2_behavior()
cp.v1.disable_eager_execution()

sess = cp.v1.Session()
my_array = np.array([[1.,3.,5.,7.,9.,1.],
                     [-2.,0.,2.,4.,6.,1.],
                     [-6.,-3.,0.,3.,6.,2.]])

x_vals = np.array([my_array,my_array+1])
print(x_vals)
x_data = cp.v1.placeholder(tf.float32,shape=[None,None])

m1 = tf.constant([[1.0],[0.],[-1.],[2.],[4.],[1.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

prod1 = cp.v1.matmul(x_data,m1)
prod2 = tf.matmul(prod1,m2)
add1 = tf.add(prod2,a1)

for x_val in x_vals:
    print(sess.run(add1,feed_dict={x_data:x_val}))