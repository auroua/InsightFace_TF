import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Layer, list_remove_repeat


D_TYPE = tf.float32


class ElementwiseLayer(Layer):
    """
    The :class:`ElementwiseLayer` class combines multiple :class:`Layer` which have the same output shapes by a given elemwise-wise operation.

    Parameters
    ----------
    layer : a list of :class:`Layer` instances
        The `Layer` class feeding into this layer.
    combine_fn : a TensorFlow elemwise-merge function
        e.g. AND is ``tf.minimum`` ;  OR is ``tf.maximum`` ; ADD is ``tf.add`` ; MUL is ``tf.multiply`` and so on.
        See `TensorFlow Math API <https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#math>`_ .
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
        self,
        layer = [],
        combine_fn = tf.minimum,
        name ='elementwise_layer',
        act = None,
    ):
        Layer.__init__(self, name=name)

        if act:
            print("  [TL] ElementwiseLayer %s: size:%s fn:%s, act:%s" % (
            self.name, layer[0].outputs.get_shape(), combine_fn.__name__, act.__name__))
        else:
            print("  [TL] ElementwiseLayer %s: size:%s fn:%s" % (
            self.name, layer[0].outputs.get_shape(), combine_fn.__name__))

        self.outputs = layer[0].outputs
        # print(self.outputs._shape, type(self.outputs._shape))
        for l in layer[1:]:
            # assert str(self.outputs.get_shape()) == str(l.outputs.get_shape()), "Hint: the input shapes should be the same. %s != %s" %  (self.outputs.get_shape() , str(l.outputs.get_shape()))
            self.outputs = combine_fn(self.outputs, l.outputs, name=name)
        if act:
            self.outputs = act(self.outputs)
        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)


class BatchNormLayer(Layer):
    """
    The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Batch normalization on fully-connected or convolutional maps.

    Parameters
    -----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    decay : float, default is 0.9.
        A decay factor for ExponentialMovingAverage, use larger value for large dataset.
    epsilon : float
        A small float number to avoid dividing by 0.
    act : activation function.
    is_train : boolean
        Whether train or inference.
    beta_init : beta initializer
        The initializer for initializing beta
    gamma_init : gamma initializer
        The initializer for initializing gamma
    dtype : tf.float32 (default) or tf.float16
    name : a string or None
        An optional name to attach to this layer.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_
    """

    def __init__(
            self,
            layer=None,
            decay=0.9,
            epsilon=2e-5,
            act=tf.identity,
            is_train=False,
            fix_gamma=True,
            beta_init=tf.zeros_initializer,
            gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),  # tf.ones_initializer,
            # dtype = tf.float32,
            trainable=None,
            name='batchnorm_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] BatchNormLayer %s: decay:%f epsilon:%f act:%s is_train:%s" % (self.name, decay, epsilon, act.__name__, is_train))
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        from tensorflow.python.training import moving_averages
        from tensorflow.python.ops import control_flow_ops

        with tf.variable_scope(name) as vs:
            axis = list(range(len(x_shape) - 1))

            ## 1. beta, gamma
            if tf.__version__ > '0.12.1' and beta_init == tf.zeros_initializer:
                beta_init = beta_init()
            with tf.device('/cpu:0'):
                beta = tf.get_variable('beta', shape=params_shape, initializer=beta_init, dtype=tf.float32, trainable=is_train)  #, restore=restore)

                gamma = tf.get_variable(
                    'gamma',
                    shape=params_shape,
                    initializer=gamma_init,
                    dtype=tf.float32,
                    trainable=fix_gamma,
                )  #restore=restore)

            ## 2.
            if tf.__version__ > '0.12.1':
                moving_mean_init = tf.zeros_initializer()
            else:
                moving_mean_init = tf.zeros_initializer
            with tf.device('/cpu:0'):
                moving_mean = tf.get_variable('moving_mean', params_shape, initializer=moving_mean_init, dtype=tf.float32, trainable=False)  #   restore=restore)
                moving_variance = tf.get_variable(
                    'moving_variance',
                    params_shape,
                    initializer=tf.constant_initializer(1.),
                    dtype=tf.float32,
                    trainable=False,
                )  #   restore=restore)

            ## 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)
            try:  # TF12
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay, zero_debias=False)  # if zero_debias=True, has bias
                update_moving_variance = moving_averages.assign_moving_average(
                    moving_variance, variance, decay, zero_debias=False)  # if zero_debias=True, has bias
                # print("TF12 moving")
            except Exception as e:  # TF11
                update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, decay)
                # print("TF11 moving")

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            # if is_train:
            #     mean, var = mean_var_with_update()
            #     self.outputs = act(tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon))
            # else:
            #     self.outputs = act(tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma, epsilon))

            def train_outputs():
                mean, var = mean_var_with_update()
                return act(tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon))

            def test_outputs():
                return act(tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma, epsilon))

            self.outputs = tf.cond(trainable,
                                   lambda: train_outputs(),
                                   lambda: test_outputs())

            variables = [beta, gamma, moving_mean, moving_variance]
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        self.all_params.extend(variables)


def Conv2d(
        net,
        n_filter=32,
        filter_size=(3, 3),
        strides=(1, 1),
        act=None,
        padding='SAME',
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(value=0.0),
        W_init_args={},
        b_init_args={},
        use_cudnn_on_gpu=None,
        data_format=None,
        name='conv2d',
):
    """Wrapper for :class:`Conv2dLayer`, if you don't understand how to use :class:`Conv2dLayer`, this function may be easier.

    Parameters
    ----------
    net : TensorLayer layer.
    n_filter : number of filter.
    filter_size : tuple (height, width) for filter size.
    strides : tuple (height, width) for strides.
    act : None or activation function.
    others : see :class:`Conv2dLayer`.

    Examples
    --------
    >>> w_init = tf.truncated_normal_initializer(stddev=0.01)
    >>> b_init = tf.constant_initializer(value=0.0)
    >>> inputs = InputLayer(x, name='inputs')
    >>> conv1 = Conv2d(inputs, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_1')
    >>> conv1 = Conv2d(conv1, 64, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_2')
    >>> pool1 = MaxPool2d(conv1, (2, 2), padding='SAME', name='pool1')
    >>> conv2 = Conv2d(pool1, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_1')
    >>> conv2 = Conv2d(conv2, 128, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_2')
    >>> pool2 = MaxPool2d(conv2, (2, 2), padding='SAME', name='pool2')
    """
    assert len(strides) == 2, "len(strides) should be 2, Conv2d and Conv2dLayer are different."
    if act is None:
        act = tf.identity

    try:
        pre_channel = int(net.outputs.get_shape()[-1])
    except:  # if pre_channel is ?, it happens when using Spatial Transformer Net
        pre_channel = 1
        print("[warnings] unknow input channels, set to 1")
    net = Conv2dLayer(
        net,
        act=act,
        shape=[filter_size[0], filter_size[1], pre_channel, n_filter],  # 32 features for each 5x5 patch
        strides=[1, strides[0], strides[1], 1],
        padding=padding,
        W_init=W_init,
        W_init_args=W_init_args,
        b_init=b_init,
        b_init_args=b_init_args,
        use_cudnn_on_gpu=use_cudnn_on_gpu,
        data_format=data_format,
        name=name)
    return net


class Conv2dLayer(Layer):
    """
    The :class:`Conv2dLayer` class is a 2D CNN layer, see `tf.nn.conv2d <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#conv2d>`_.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    act : activation function
        The function that is applied to the layer activations.
    shape : list of shape
        shape of the filters, [filter_height, filter_width, in_channels, out_channels].
    strides : a list of ints.
        The stride of the sliding window for each dimension of input.\n
        It Must be in the same order as the dimension specified with format.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    W_init : weights initializer
        The initializer for initializing the weight matrix.
    b_init : biases initializer or None
        The initializer for initializing the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weights tf.get_variable().
    b_init_args : dictionary
        The arguments for the biases tf.get_variable().
    use_cudnn_on_gpu : bool, default is None.
    data_format : string "NHWC" or "NCHW", default is "NHWC"
    name : a string or None
        An optional name to attach to this layer.

    Notes
    ------
    - shape = [h, w, the number of output channel of previous layer, the number of output channels]
    - the number of output channel of a layer is its last dimension.

    Examples
    --------
    >>> x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    >>> network = tl.layers.InputLayer(x, name='input_layer')
    >>> network = tl.layers.Conv2dLayer(network,
    ...                   act = tf.nn.relu,
    ...                   shape = [5, 5, 1, 32],  # 32 features for each 5x5 patch
    ...                   strides=[1, 1, 1, 1],
    ...                   padding='SAME',
    ...                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
    ...                   W_init_args={},
    ...                   b_init = tf.constant_initializer(value=0.0),
    ...                   b_init_args = {},
    ...                   name ='cnn_layer1')     # output: (?, 28, 28, 32)
    >>> network = tl.layers.PoolLayer(network,
    ...                   ksize=[1, 2, 2, 1],
    ...                   strides=[1, 2, 2, 1],
    ...                   padding='SAME',
    ...                   pool = tf.nn.max_pool,
    ...                   name ='pool_layer1',)   # output: (?, 14, 14, 32)

    >>> Without TensorLayer, you can implement 2d convolution as follow.
    >>> W = tf.Variable(W_init(shape=[5, 5, 1, 32], ), name='W_conv')
    >>> b = tf.Variable(b_init(shape=[32], ), name='b_conv')
    >>> outputs = tf.nn.relu( tf.nn.conv2d(inputs, W,
    ...                       strides=[1, 1, 1, 1],
    ...                       padding='SAME') + b )
    """

    def __init__(
            self,
            layer=None,
            act=tf.identity,
            shape=[5, 5, 1, 100],
            strides=[1, 1, 1, 1],
            padding='SAME',
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args={},
            b_init_args={},
            use_cudnn_on_gpu=None,
            data_format=None,
            name='cnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] Conv2dLayer %s: shape:%s strides:%s pad:%s act:%s" % (self.name, str(shape), str(strides), padding, act.__name__))

        with tf.variable_scope(name) as vs:
            with tf.device('/cpu:0'):
                W = tf.get_variable(name='W_conv2d', shape=shape, initializer=W_init, dtype=D_TYPE, **W_init_args)
            if b_init:
                with tf.device('/cpu:0'):
                    b = tf.get_variable(name='b_conv2d', shape=(shape[-1]), initializer=b_init, dtype=D_TYPE, **b_init_args)
                self.outputs = act(
                    tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format) + b)
            else:
                self.outputs = act(tf.nn.conv2d(self.inputs, W, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu, data_format=data_format))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])


## Special activation
class PReluLayer(Layer):
    """
    The :class:`PReluLayer` class is Parametric Rectified Linear layer.

    Parameters
    ----------
    x : A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
        `int16`, or `int8`.
    channel_shared : `bool`. Single weight is shared by all channels
    a_init : alpha initializer, default zero constant.
        The initializer for initializing the alphas.
    a_init_args : dictionary
        The arguments for the weights initializer.
    name : A name for this activation op (optional).

    References
    -----------
    - `Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification <http://arxiv.org/pdf/1502.01852v1.pdf>`_
    """

    def __init__(
            self,
            layer=None,
            channel_shared=False,
            a_init=tf.constant_initializer(value=0.0),
            a_init_args={},
            # restore = True,
            name="prelu_layer"):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] PReluLayer %s: channel_shared:%s" % (self.name, channel_shared))
        if channel_shared:
            w_shape = (1, )
        else:
            w_shape = int(self.inputs.get_shape()[-1])

        # with tf.name_scope(name) as scope:
        with tf.variable_scope(name) as vs:
            with tf.device('/cpu:0'):
                alphas = tf.get_variable(name='alphas', shape=w_shape, initializer=a_init, dtype=D_TYPE, **a_init_args)
            try:  ## TF 1.0
                self.outputs = tf.nn.relu(self.inputs) + tf.multiply(alphas, (self.inputs - tf.abs(self.inputs))) * 0.5
            except:  ## TF 0.12
                self.outputs = tf.nn.relu(self.inputs) + tf.mul(alphas, (self.inputs - tf.abs(self.inputs))) * 0.5

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend([self.outputs])
        self.all_params.extend([alphas])


## Dense layer
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