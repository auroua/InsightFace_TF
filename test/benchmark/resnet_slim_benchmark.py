import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import numpy as np


slim = tf.contrib.slim
resnet = nets.resnet_v1

if __name__ == '__main__':
    output_shape = 85164
    batch_size = 64
    image = tf.placeholder(name='input_x', shape=[None, 224, 224, 3], dtype=tf.float32)
    labels = tf.placeholder(name='input_label', shape=[None, output_shape], dtype=tf.float32)
    with slim.arg_scope(nets.resnet_utils.resnet_arg_scope()):
        resnet_50, end_points = resnet.resnet_v1_50(inputs=image, num_classes=output_shape, scope='resnet_v1_50')
        prob = tf.squeeze(resnet_50, axis=[1, 2])
    probabilities = tf.reduce_mean(tf.nn.softmax(prob, dim=-1))
    losses = tf.norm(tf.subtract(probabilities, labels))
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(losses)
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    while True:
        datasets = np.random.randn(batch_size, 224, 224, 3).astype(np.float32)
        datasets_labels = np.random.randn(batch_size, output_shape).astype(np.float32)
        losses_val, _ = sess.run([losses, train_op], feed_dict={image: datasets, labels: datasets_labels})
        print(losses_val)