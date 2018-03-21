import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib.layers.python.layers import utils
import collections
from tensorlayer.layers import Layer, list_remove_repeat


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


def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return tl.layers.MaxPool2d(inputs, [1, 1], strides=(factor, factor), name=scope)


def conv2d_same(inputs, num_outputs, kernel_size, strides, rate=1, w_init=None, scope=None, trainable=None):
    '''
    Reference slim resnet
    :param inputs:
    :param num_outputs:
    :param kernel_size:
    :param strides:
    :param rate:
    :param scope:
    :return:
    '''
    if strides == 1:
        if rate == 1:
            nets = tl.layers.Conv2d(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size), b_init=None,
                                   strides=(strides, strides), W_init=w_init, act=None, padding='SAME', name=scope,
                                    use_cudnn_on_gpu=True)
            nets.outputs = tf.layers.batch_normalization(inputs=nets.outputs,
                                                         momentum=0.9,
                                                         training=trainable,
                                                         renorm=True,
                                                         renorm_clipping={'rmax':3, 'rmin':0.3333,
                                                                          'dmax':5},
                                                         renorm_momentum=0.9,
                                                         name=scope+'_bn/BatchNorm')
        else:
            nets = tl.layers.AtrousConv2dLayer(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size),
                                               rate=rate, act=None, W_init=w_init, padding='SAME', name=scope)
            nets.outputs = tf.layers.batch_normalization(inputs=nets.outputs,
                                                         momentum=0.9,
                                                         training=trainable,
                                                         renorm=True,
                                                         renorm_clipping={'rmax':3, 'rmin':0.3333,
                                                                          'dmax':5},
                                                         renorm_momentum=0.9,
                                                         name=scope+'_bn/BatchNorm')
        return nets
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tl.layers.PadLayer(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], name='padding_%s' % scope)
        if rate == 1:
            nets = tl.layers.Conv2d(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size), b_init=None,
                                    strides=(strides, strides), W_init=w_init, act=None, padding='VALID', name=scope,
                                    use_cudnn_on_gpu=True)
            nets.outputs = tf.layers.batch_normalization(inputs=nets.outputs,
                                                         momentum=0.9,
                                                         training=trainable,
                                                         renorm=True,
                                                         renorm_clipping={'rmax':3, 'rmin':0.3333,
                                                                          'dmax':5},
                                                         renorm_momentum=0.9,
                                                         name=scope+'_bn/BatchNorm')
        else:
            nets = tl.layers.AtrousConv2dLayer(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size), b_init=None,
                                              rate=rate, act=None, W_init=w_init, padding='SAME', name=scope)
            nets.outputs = tf.layers.batch_normalization(inputs=nets.outputs,
                                                         momentum=0.9,
                                                         training=trainable,
                                                         renorm=True,
                                                         renorm_clipping={'rmax':3, 'rmin':0.3333,
                                                                          'dmax':5},
                                                         renorm_momentum=0.9,
                                                         name=scope+'_bn/BatchNorm')
        return nets


def bottleneck_IR(inputs, depth, depth_bottleneck, stride, rate=1, w_init=None, scope=None, trainable=None):
    with tf.variable_scope(scope, 'bottleneck_v1') as sc:
        depth_in = utils.last_dimension(inputs.outputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = tl.layers.Conv2d(inputs, depth, filter_size=(1, 1), strides=(stride, stride), act=None,
                                        W_init=w_init, b_init=None, name='shortcut_conv', use_cudnn_on_gpu=True)
            shortcut.outputs = tf.layers.batch_normalization(inputs=shortcut.outputs,
                                                             momentum=0.9,
                                                             training=trainable,
                                                             renorm=True,
                                                             renorm_clipping={'rmax': 3, 'rmin': 0.3333,
                                                                              'dmax': 5},
                                                             renorm_momentum=0.9,
                                                             name='shortcut_bn/BatchNorm')
        # bottleneck layer 1
        inputs.outputs = tf.layers.batch_normalization(inputs=inputs.outputs,
                                                         momentum=0.9,
                                                         training=trainable,
                                                         renorm=True,
                                                         renorm_clipping={'rmax': 3, 'rmin': 0.3333,
                                                                          'dmax': 5},
                                                         renorm_momentum=0.9,
                                                         name='conv1_bn1')
        residual = tl.layers.Conv2d(inputs, depth_bottleneck, filter_size=(3, 3), strides=(1, 1), act=None, b_init=None,
                                    W_init=w_init, name='conv1', use_cudnn_on_gpu=True)
        residual.outputs = tf.layers.batch_normalization(inputs=residual.outputs,
                                                         momentum=0.9,
                                                         training=trainable,
                                                         renorm=True,
                                                         renorm_clipping={'rmax': 3, 'rmin': 0.3333,
                                                                          'dmax': 5},
                                                         renorm_momentum=0.9,
                                                         name='conv1_bn2')
        # bottleneck prelu
        residual = tl.layers.PReluLayer(residual)
        # bottleneck layer 2
        residual = conv2d_same(residual, depth, kernel_size=3, strides=stride, rate=rate, w_init=w_init, scope='conv2', trainable=trainable)
        output = ElementwiseLayer(layer=[shortcut, residual],
                                  combine_fn=tf.add,
                                  name='combine_layer',
                                  act=None)
        return output


def resnet(inputs, bottle_neck, blocks, w_init=None, trainable=None, scope=None):
    with tf.variable_scope(scope):
        # inputs = tf.subtract(inputs, 127.5)
        # inputs = tf.multiply(inputs, 0.0078125)
        net_inputs = tl.layers.InputLayer(inputs, name='input_layer')
        if bottle_neck:
            net = tl.layers.Conv2d(net_inputs, n_filter=64, filter_size=(3, 3), strides=(1, 1),
                                   act=None, W_init=w_init, b_init=None, name='conv1', use_cudnn_on_gpu=True)
            net.outputs = tf.layers.batch_normalization(inputs=net.outputs,
                                                             momentum=0.9,
                                                             training=trainable,
                                                             renorm=True,
                                                             renorm_clipping={'rmax': 3, 'rmin': 0.3333,
                                                                              'dmax': 5},
                                                             renorm_momentum=0.9,
                                                             name='bn0')
            net = tl.layers.PReluLayer(net, name='prelu0')
        else:
            raise ValueError('The standard resnet must support the bottleneck layer')
        for block in blocks:
            with tf.variable_scope(block.scope):
                for i, var in enumerate(block.args):
                    with tf.variable_scope('unit_%d' % (i+1)):
                        net = block.unit_fn(net, depth=var['depth'], depth_bottleneck=var['depth_bottleneck'],
                                            w_init=w_init, stride=var['stride'], rate=var['rate'], scope=None,
                                            trainable=trainable)
        net.outputs = tf.layers.batch_normalization(inputs=net.outputs,
                                                    momentum=0.9,
                                                    training=trainable,
                                                    renorm=True,
                                                    renorm_clipping={'rmax': 3, 'rmin': 0.3333,
                                                                     'dmax': 5},
                                                    renorm_momentum=0.9,
                                                    name='E_BN1')
        net = tl.layers.DropoutLayer(net, keep=0.4, name='E_Dropout')
        net_shape = net.outputs.get_shape()
        net = tl.layers.ReshapeLayer(net, shape=[-1, net_shape[1]*net_shape[2]*net_shape[3]], name='E_Reshapelayer')
        net = tl.layers.DenseLayer(net, n_units=512, W_init=w_init, name='E_DenseLayer')
        net.outputs = tf.layers.batch_normalization(inputs=net.outputs,
                                                    momentum=0.9,
                                                    training=trainable,
                                                    renorm=True,
                                                    renorm_clipping={'rmax': 3, 'rmin': 0.3333,
                                                                     'dmax': 5},
                                                    renorm_momentum=0.9,
                                                    name='E_BN2')
        return net


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

    Its parts are:
      scope: The scope of the `Block`.
      unit_fn: The ResNet unit function which takes as input a `Tensor` and
        returns another `Tensor` with the output of the ResNet unit.
      args: A list of length equal to the number of units in the `Block`. The list
        contains one (depth, depth_bottleneck, stride) tuple for each unit in the
        block to serve as argument to unit_fn.
    """


def resnetse_v1_block(scope, base_depth, num_units, stride, rate=1, unit_fn=None):
  """Helper function for creating a resnet_v1 bottleneck block.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
    A resnet_v1 bottleneck block.
  """
  return Block(scope, unit_fn, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride,
      'rate': rate
  }] + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1,
      'rate': rate
  }] * (num_units - 1))


def get_resnet(inputs, num_layers, type=None, w_init=None, trainable=None, sess=None):
    if type == 'ir':
        unit_fn = bottleneck_IR
    # elif type == 'se_ir':
    #     unit_fn = bottleneck_IR_SE
    else:
        raise ValueError('the input fn is unknown')

    if num_layers == 50:
        blocks = [
            resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block2', base_depth=128, num_units=4, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block3', base_depth=256, num_units=14, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block4', base_depth=512, num_units=3, stride=2, rate=1, unit_fn=unit_fn)
        ]
    elif num_layers == 101:
        blocks = [
            resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block2', base_depth=128, num_units=13, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block3', base_depth=256, num_units=30, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block4', base_depth=512, num_units=3, stride=2, rate=1, unit_fn=unit_fn)
        ]
    elif num_layers == 152:
        blocks = [
            resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block2', base_depth=128, num_units=8, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block3', base_depth=256, num_units=36, stride=2, rate=1, unit_fn=unit_fn),
            resnetse_v1_block('block4', base_depth=512, num_units=3, stride=2, rate=1, unit_fn=unit_fn)
        ]
    else:
        raise ValueError('Resnet layer %d is not supported now.' % num_layers)
    net = resnet(inputs=inputs,
                 bottle_neck=True,
                 blocks=blocks,
                 w_init=w_init,
                 trainable=trainable,
                 scope='resnet_v1_%d' % num_layers)
    return net


if __name__ == '__main__':
        x = tf.placeholder(dtype=tf.float32, shape=[None, 112, 112, 3], name='input_place')
        sess = tf.Session()
        # w_init = tf.truncated_normal_initializer(mean=10, stddev=5e-2)
        w_init = tf.contrib.layers.xavier_initializer(uniform=False)
        # test resnetse
        nets = get_resnet(x, 50, type='ir', w_init=w_init, sess=sess)
        tl.layers.initialize_global_variables(sess)

        for p in tl.layers.get_variables_with_name('W_conv2d', True, True):
            print(p.op.name)
        print('##############'*30)
        with sess:
            nets.print_params()
