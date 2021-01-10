import tensorflow as tf
import numpy as np
# import tensorflow.compat.v1 as tt
# tt.disable_v2_behavior()

arr_list = np.arange(0,100).astype(np.float32)
shape = arr_list.shape
dataset = tf.data.Dataset.from_tensor_slices(arr_list)
dataset_iterator = dataset.batch(10)

def model(xs):
    outputs = tf.multiply(xs,1)
    return outputs

for it in dataset_iterator:
    logits = model(it)
    print(logits)
