import tensorflow as tf
import tensorlayer as tl
import os

Layer = tl.layers.Layer
D_TYPE = tf.float32

class DenseLayer(Layer):
    """
    The :class:`DenseLayer` class is a fully connected layer.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    n_units : int
        The number of units of the layer.
    act : activation function
        The function that is applied to the layer activations.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer or None
        The initializer for initializing the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weights tf.get_variable.
    b_init_args : dictionary
        The arguments for the biases tf.get_variable.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.DenseLayer(
    ...                 network,
    ...                 n_units=800,
    ...                 act = tf.nn.relu,
    ...                 W_init=tf.truncated_normal_initializer(stddev=0.1),
    ...                 name ='relu_layer'
    ...                 )

    >>> Without TensorLayer, you can do as follow.
    >>> W = tf.Variable(
    ...     tf.random_uniform([n_in, n_units], -1.0, 1.0), name='W')
    >>> b = tf.Variable(tf.zeros(shape=[n_units]), name='b')
    >>> y = tf.nn.relu(tf.matmul(inputs, W) + b)

    Notes
    -----
    If the input to this layer has more than two axes, it need to flatten the
    input by using :class:`FlattenLayer` in this case.
    """

    def __init__(
            self,
            layer=None,
            n_units=100,
            act=tf.identity,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args={},
            b_init_args={},
            name='dense_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units
        print("  [TL] DenseLayer  %s: %d %s" % (self.name, self.n_units, act.__name__))
        with tf.variable_scope(name) as vs:
            with tf.device('/cpu:0'):
                W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, dtype=D_TYPE, **W_init_args)
            if b_init is not None:
                try:
                    with tf.device('/cpu:0'):
                        b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, dtype=D_TYPE, **b_init_args)
                except:  # If initializer is a constant, do not specify shape.
                    with tf.device('/cpu:0'):
                        b = tf.get_variable(name='b', initializer=b_init, dtype=D_TYPE, **b_init_args)
                self.outputs = act(tf.matmul(self.inputs, W) + b)
            else:
                self.outputs = act(tf.matmul(self.inputs, W))

        # Hint : list(), dict() is pass by value (shallow), without them, it is
        # pass by reference.
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init is not None:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])


def inference(x):
    network = tl.layers.InputLayer(x, name='input')
    network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
    network = DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = DenseLayer(network, n_units=800, act=tf.nn.relu, name='relu2')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
    network = DenseLayer(network, n_units=10, act=tf.identity, name='output')
    y = network.outputs
    return y


def load_data():
    X_train, y_train, X_val, y_val, X_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1, 784), path='/home/aurora/workspaces/data')
    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_val.shape', X_val.shape)
    print('y_val.shape', y_val.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))
    return X_train, y_train


def tower_losses(inputs, labels):
    logit = inference(inputs)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels, name='cross_entropy')
    return loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []

  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, g1 in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(0.01,
                                        global_step,
                                        10000,
                                        0.99,
                                        staircase=True)
        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(lr)
        tower_grads = []
        x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(1):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        tl.layers.set_name_reuse(True)
                        # Dequeues one batch for the GPU
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        loss = tower_losses(x, y_)
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        train_op = tf.group(apply_gradient_op, variables_averages_op)
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True))
        sess.run(init)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train()

