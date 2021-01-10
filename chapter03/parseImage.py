import tensorflow as tf
import glob
from PIL import Image

print(tf.__version__)

##将 图片 生成TFRecord文件
##filenames = tf.io.match_filenames_once('./*.jpg')
filenames = glob.iglob('result.png')
output_tfrecord_file = './jpg.tfrecords'

with tf.io.TFRecordWriter(output_tfrecord_file) as writer :
    for filename in filenames:
        img = Image.open(filename)
        img_raw = img.tobytes()
        label = 1
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }
            )
        )
        writer.write(record=example.SerializeToString())