import tensorflow as tf

'''
控制流
'''
@tf.function
def control_flow(x):
    if(x > 2):
        return x + 10
    else:
        return 0

'''
普通的python脚本转换成tf流
'''
@tf.function
def matrix(c1,c2):
    res = tf.matmul(c1,c2)
    return res

if __name__ == "__main__":
    res = control_flow(10)
    print("控制流计算结果：{}".format(res))
    # c1 = tf.constant([[1,2,4],[3,4,5]])
    # c2 = tf.constant([[2,3],[3,6],[8,9]])
    # res = matrix(c1,c2)
    # print("矩阵结果：{}".format(res))