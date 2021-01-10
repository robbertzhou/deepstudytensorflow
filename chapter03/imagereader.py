#  cifar10数据集10分类数据集，32X32大小的RGB3通道图片，50000张用于训练，10000张用于测试
import numpy as np
import pickle
from matplotlib import pyplot as plt
import cv2


def unpickle(file):  # CIFAR-10官方给出的使用方法
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='iso-8859-1')
    return dict


# 加载训练集
file = '数据文件/cifar-10-batches-py/data_batch_'  # 文件的路径，只加载了10000张图片
x_train = np.empty(shape=[0, 3072])
y_train = []
for ii in range(5):
    file1 = file + str(ii + 1)
    dict_train_batch1 = unpickle(file1)  # 将data_batch文件读入到数据结构(字典)中
    data_train_batch1 = dict_train_batch1.get('data')  # 字典中取data
    labels1 = dict_train_batch1.get('labels')  # 字典中取labels
    x_train = np.append(x_train, data_train_batch1, axis=0)
    y_train = np.append(y_train, labels1)

# 加载测试集
file = '数据文件/cifar-10-batches-py/test_batch'
dict_test = unpickle(file)
x_test = dict_test.get("data")
y_test = dict_test.get("labels")

image_m = np.reshape(x_test[16], (3, 32, 32))

r = image_m[0, :, :]
g = image_m[1, :, :]
b = image_m[2, :, :]
img23 = cv2.merge([r, g, b])

plt.figure()
plt.imshow(img23)
plt.show()