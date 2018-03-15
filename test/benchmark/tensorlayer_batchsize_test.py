import tensorflow as tf
import tensorlayer as tl
import os


def inference(x):
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=True)
    # define the network
    network = tl.layers.InputLayer(x, name='input')
    network = tl.layers.Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), padding='SAME', act=None,
                               W_init=w_init_method, name='conv1_1')
    network = tl.layers.BatchNormLayer(network, act=tf.identity, is_train=True, name='bn1')
    network = tl.layers.PReluLayer(network, name='prelu1')
    network = tl.layers.Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), padding='SAME', act=None,
                               W_init=w_init_method, name='conv1_2')
    network = tl.layers.BatchNormLayer(network, act=tf.identity, is_train=True, name='bn2')
    network = tl.layers.PReluLayer(network, name='prelu2')
    network = tl.layers.Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(2, 2), padding='SAME', act=None,
                               W_init=w_init_method, name='conv1_3')
    network = tl.layers.BatchNormLayer(network, act=tf.identity, is_train=True, name='bn3')
    network = tl.layers.PReluLayer(network, name='prelu3')

    network = tl.layers.Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), padding='SAME', act=None,
                               W_init=w_init_method, name='conv2_1')
    network = tl.layers.BatchNormLayer(network, act=tf.identity, is_train=True, name='bn4')
    network = tl.layers.PReluLayer(network, name='prelu4')

    network = tl.layers.Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), padding='SAME', act=None,
                               W_init=w_init_method, name='conv2_2')
    network = tl.layers.BatchNormLayer(network, act=tf.identity, is_train=True, name='bn5')
    network = tl.layers.PReluLayer(network, name='prelu5')
    network = tl.layers.Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(2, 2), padding='SAME', act=None,
                               W_init=w_init_method, name='conv2_3')
    network = tl.layers.BatchNormLayer(network, act=tf.identity, is_train=True, name='bn6')
    network = tl.layers.PReluLayer(network, name='prelu6')

    network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), padding='SAME', act=None,
                               W_init=w_init_method, name='conv3_1')
    network = tl.layers.BatchNormLayer(network, act=tf.identity, is_train=True, name='bn7')
    network = tl.layers.PReluLayer(network, name='prelu7')
    network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), padding='SAME', act=None,
                               W_init=w_init_method, name='conv3_2')
    network = tl.layers.BatchNormLayer(network, act=tf.identity, is_train=True, name='bn8')
    network = tl.layers.PReluLayer(network, name='prelu8')
    network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(2, 2), padding='SAME', act=None,
                               W_init=w_init_method, name='conv3_3')
    network = tl.layers.BatchNormLayer(network, act=tf.identity, is_train=True, name='bn9')
    network = tl.layers.PReluLayer(network, name='prelu9')

    network = tl.layers.FlattenLayer(network, name='flatten')
    network = tl.layers.DenseLayer(network, 10)

    return network.outputs


if __name__ == '__main__':
    # without bn prelu     8000< max batch size <9000
    # with bn only         5000< max batch size <6000
    # with prelu only      3000< max batch size <4000
    # with bn and prelu    2000< max batch size <3000
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    batch_size = 2000
    n_epoch = 10
    # prepare data
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
    # define placeholder
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
    y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

    output = inference(x)
    cost = tl.cost.cross_entropy(output, y_, 'cost')
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

    sess = tf.Session()
    tl.layers.initialize_global_variables(sess)

    correct_prediction = tf.equal(tf.argmax(output, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for epoch in range(n_epoch):
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            _, err, ac = sess.run([train_op, cost, acc], feed_dict=feed_dict)
            train_loss += err
            train_acc += ac
            n_batch += 1
        print("epoch %d, train acc: %f" % (epoch, (train_acc / n_batch)))