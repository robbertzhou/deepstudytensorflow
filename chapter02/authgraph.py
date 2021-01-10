import tensorflow as tf


def check_fun(input_num):
    if input_num > 1:
        return input_num * input_num
    else:
        return 1 - input_num


print("output is:",check_fun(tf.constant(2.0)),"and",check_fun(tf.constant(0.3)))


@tf.function
def get_fun(input_num):
    print("input is:",input_num)
    return input_num + input_num

print("result is:",get_fun(tf.constant(1.0)))