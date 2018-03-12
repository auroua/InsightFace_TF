import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import numpy as np

slim = tf.contrib.slim

if __name__ == '__main__':
    output_shape = 1000
    batch_size = 128
    image = tf.placeholder(name='input_x', shape=[None, 224, 224, 3], dtype=tf.float32)
    labels = tf.placeholder(name='input_label', shape=[None, output_shape], dtype=tf.float32)
    with slim.arg_scope(nets.vgg.vgg_arg_scope()):
        vgg_19, end_points = nets.vgg.vgg_19(inputs=image, num_classes=output_shape, scope='vgg_19')
    probabilities = tf.reduce_mean(tf.nn.softmax(vgg_19, dim=-1))
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