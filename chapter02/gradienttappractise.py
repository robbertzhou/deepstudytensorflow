import tensorflow as tf
x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = x * x * 4.0
  print("eew")
dy_dx = g.gradient(y, x) # y’ = 2*x = 2*3 = 6
print(dy_dx)
print(x)