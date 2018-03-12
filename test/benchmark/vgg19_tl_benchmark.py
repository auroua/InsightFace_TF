import tensorflow as tf
from nets.vgg19 import get_vgg19
import numpy as np


if __name__ == '__main__':
    sess = tf.Session()
    x = tf.placeholder(name="inputs_x", shape=[None, 224, 224, 3], dtype=tf.float32)
    y = tf.placeholder(name='inputs_y', shape=[None, 1000], dtype=tf.float32)
    network = get_vgg19(x, sess, pretrained=False)
    outputs_y = network.outputs
    probs = tf.nn.softmax(outputs_y, name="prob")
    loss = tf.reduce_mean(tf.subtract(probs, y))

    while True:
        batch_size = 128
        datasets_x = np.random.randn(batch_size, 224, 224, 3).astype(np.float32)
        datasets_y = np.random.randn(batch_size, 1000).astype(np.float32)
        feed_dict = {x: datasets_x, y: datasets_y}
        loss_val = sess.run(loss, feed_dict=feed_dict)
        print('batch size %d, loss value is %.2f' % (batch_size, loss_val))
