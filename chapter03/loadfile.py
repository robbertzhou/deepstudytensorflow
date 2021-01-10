import base64
from PIL import Image


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

up = unpickle("I:\\testdata\\cifar-10-python\\cifar-10-batches-py\\data_batch_3")

files = up.get(b'filenames')

im = Image.frombuffer('L', (32,32), up.get(b'data')[1], 'raw', 'L', 0, 3)
im.save('result.png')
#     data = f.read()
#     base64.b64encode()
# base64.b64encode(data)  # 图片转字节
# base64.b64decode(data)  # 字节转图片