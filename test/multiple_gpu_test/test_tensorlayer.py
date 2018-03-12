import tensorflow as tf
import tensorlayer as tl


if __name__ == '__main__':
    with tf.name_scope('test'):
            x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
            network = tl.layers.InputLayer(x, name='input')
            network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
            network = tl.layers.DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu1')
            network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
            network = tl.layers.DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu2')
            network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
            network = tl.layers.DenseLayer(network, n_units=10, act=tf.identity, name='output')
        # y = network
    network.print_layers()
    sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True))
    tl.layers.initialize_global_variables(sess)
    with sess:
        network.print_params()
