# filename_queue = tf.train.string_input_producer([filename])# create a queue
#  
#   reader = tf.TFRecordReader()
#   _, serialized_example = reader.read(filename_queue)#return file_name and file
#   features = tf.parse_single_example(serialized_example,
#                     features={
#                       'label': tf.FixedLenFeature([], tf.int64),
#                       'img_raw' : tf.FixedLenFeature([], tf.string),
#                     })#return image and label
#  
#   img = tf.decode_raw(features['img_raw'], tf.uint8)
#   img = tf.reshape(img, [512, 80, 3]) #reshape image to 512*80*3
#   img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #throw img tensor
#   label = tf.cast(features['label'], tf.int32) #throw label tensor
#   return img, label

import tensorflow as tf

image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def parse_tf_example(example_proto):
    # 解析出来
    parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)

    # 预处理
    x_train = tf.image.decode_jpeg(parsed_example['image_raw'], channels=3)
    x_train = tf.image.resize(x_train, (416, 416))
    x_train /= 255.

    lebel = parsed_example['label']
    y_train = lebel

    return x_train, y_train


dataset = tf.data.TFRecordDataset("jpb.tfrecords")
image,label = parse_tf_example(dataset)
# features = tf.io.parse_single_example(dataset,
#                                     features={
#                                         "image_raw":tf.io.FixedLenFeature([],tf.string),
#                                         "image_num":tf.io.FixedLenFeature([],tf.int64),
# "height":tf.io.FixedLenFeature([],tf.int64),
# "width":tf.io.FixedLenFeature([],tf.int64)
#                                     }
#                                       )
print(dataset)
