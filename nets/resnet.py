import tensorflow as tf
import tensorlayer as tl
from tensorflow.contrib.layers.python.layers import utils
import collections
from tensorlayer.layers import Layer, list_remove_repeat
from nets_utils import get_variables_in_checkpoint_file


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


def conv2d_same(inputs, num_outputs, kernel_size, strides, rate=1, scope=None):
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
                                   strides=(strides, strides), act=None, padding='SAME', name=scope)
            nets = tl.layers.BatchNormLayer(nets, act=tf.nn.relu, is_train=True, name=scope+'_bn/BatchNorm')
        else:
            nets = tl.layers.AtrousConv2dLayer(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size),
                                               rate=rate, act=None, padding='SAME', name=scope)
            nets = tl.layers.BatchNormLayer(nets, act=tf.nn.relu, is_train=True, name=scope+'_bn/BatchNorm')
        return nets
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tl.layers.PadLayer(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], name='padding_%s' % scope)
        if rate == 1:
            nets = tl.layers.Conv2d(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size), b_init=None,
                                    strides=(strides, strides), act=None, padding='VALID', name=scope)
            nets = tl.layers.BatchNormLayer(nets, act=tf.nn.relu, is_train=True, name=scope+'_bn/BatchNorm')
        else:
            nets = tl.layers.AtrousConv2dLayer(inputs, n_filter=num_outputs, filter_size=(kernel_size, kernel_size), b_init=None,
                                              rate=rate, act=None, padding='SAME', name=scope)
            nets = tl.layers.BatchNormLayer(nets, act=tf.nn.relu, is_train=True, name=scope+'_bn/BatchNorm')
        return nets


def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v1') as sc:
        depth_in = utils.last_dimension(inputs.outputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = tl.layers.Conv2d(inputs, depth, filter_size=(1, 1), strides=(stride, stride), act=None,
                                        b_init=None, name='shortcut_conv')
            shortcut = tl.layers.BatchNormLayer(shortcut, act=tf.identity, is_train=True, name='shortcut_bn/BatchNorm')
        # bottleneck layer 1
        residual = tl.layers.Conv2d(inputs, depth_bottleneck, filter_size=(1, 1), strides=(1, 1), act=None, b_init=None,
                                    name='conv1')
        residual = tl.layers.BatchNormLayer(residual, act=tf.nn.relu, is_train=True, name='conv1_bn/BatchNorm')

        # bottleneck layer 2
        residual = conv2d_same(residual, depth_bottleneck, kernel_size=3, strides= stride, rate=rate, scope='conv2')

        # bottleneck layer 3
        residual = tl.layers.Conv2d(residual, depth, filter_size=(1, 1), strides=(1, 1), act=None, b_init=None,
                                    name='conv3')
        residual = tl.layers.BatchNormLayer(residual, act=tf.identity, is_train=True, name='conv3_bn/BatchNorm')

        output = ElementwiseLayer(layer=[shortcut, residual],
                                  combine_fn=tf.add,
                                  name='combine_layer',
                                  act=tf.nn.relu)
        return output


def bottleneck_SE(inputs, depth, depth_bottleneck, stride, rate=1, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v1') as sc:
        depth_in = utils.last_dimension(inputs.outputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = tl.layers.Conv2d(inputs, depth, filter_size=(1, 1), strides=(stride, stride), act=None,
                                        b_init=None, name='shortcut_conv')
            shortcut = tl.layers.BatchNormLayer(shortcut, act=tf.identity, is_train=True, name='shortcut_bn/BatchNorm')
        # bottleneck layer 1
        residual = tl.layers.Conv2d(inputs, depth_bottleneck, filter_size=(1, 1), strides=(1, 1), act=None, b_init=None,
                                    name='conv1')
        residual = tl.layers.BatchNormLayer(residual, act=tf.nn.relu, is_train=True, name='conv1_bn/BatchNorm')

        # bottleneck layer 2
        residual = conv2d_same(residual, depth_bottleneck, kernel_size=3, strides= stride, rate=rate, scope='conv2')

        # bottleneck layer 3
        residual = tl.layers.Conv2d(residual, depth, filter_size=(1, 1), strides=(1, 1), act=None, b_init=None,
                                    name='conv3')
        residual = tl.layers.BatchNormLayer(residual, act=tf.identity, is_train=True, name='conv3_bn/BatchNorm')

        # squeeze
        squeeze = tl.layers.InputLayer(tf.reduce_mean(residual.outputs, axis=[1, 2]), name='squeeze_layer')
        # excitation
        excitation1 = tl.layers.DenseLayer(squeeze, n_units=int(depth/16.0), act=tf.nn.relu, name='excitation_1')
        excitation2 = tl.layers.DenseLayer(excitation1, n_units=depth, act=tf.nn.sigmoid, name='excitation_2')
        # scale
        scale = tl.layers.ReshapeLayer(excitation2, shape=[tf.shape(excitation2.outputs)[0], 1, 1, depth], name='excitation_reshape')

        residual_se = ElementwiseLayer(layer=[residual, scale],
                                    combine_fn=tf.multiply,
                                    name='scale_layer',
                                    act=None)

        output = ElementwiseLayer(layer=[shortcut, residual_se],
                                  combine_fn=tf.add,
                                  name='combine_layer',
                                  act=tf.nn.relu)
        return output


def bottleneck_Xt(inputs, depth, stride, cardinality, cardinality_dim, rate=1, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v1') as sc:
        depth_in = utils.last_dimension(inputs.outputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = tl.layers.Conv2d(inputs, depth, filter_size=(1, 1), strides=(stride, stride), act=None,
                                        b_init=None, name='shortcut_conv')
            shortcut = tl.layers.BatchNormLayer(shortcut, act=tf.identity, is_train=True, name='shortcut_bn/BatchNorm')

        cardinality_layers = []
        for i in range(cardinality):
            # bottleneck layer 1
            residual = tl.layers.Conv2d(inputs, cardinality_dim, filter_size=(1, 1), strides=(1, 1), act=None, b_init=None,
                                        name='cardinality_%d/conv1' % i)
            residual = tl.layers.BatchNormLayer(residual, act=tf.nn.relu, is_train=True, name='cardinality_%d/conv1_bn/BatchNorm' % i)

            # bottleneck layer 2
            residual = conv2d_same(residual, cardinality_dim, kernel_size=3, strides= stride, rate=rate, scope='cardinality_%d/conv2' % i)

            # bottleneck layer 3
            residual = tl.layers.Conv2d(residual, depth, filter_size=(1, 1), strides=(1, 1), act=None, b_init=None,
                                        name='cardinality_%d/conv3' % i)
            residual = tl.layers.BatchNormLayer(residual, act=tf.identity, is_train=True, name='cardinality_%d/conv3_bn/BatchNorm' % i)
            cardinality_layers.append(residual)

        residual_total = ElementwiseLayer(layer=cardinality_layers,
                                          combine_fn=tf.add,
                                          name='cardinality_cmobine',
                                          act=None)
        with tf.control_dependencies([residual_total.outputs]):
            output = ElementwiseLayer(layer=[shortcut, residual_total],
                                          combine_fn=tf.add,
                                          name='combine_layer',
                                          act=tf.nn.relu)
        return output


def resnet(inputs, bottle_neck, blocks, num_classes=1000, scope=None, type=None):
    # mean_rgb_var = tf.Variable()
    with tf.variable_scope(scope):
        mean_rgb_var = tf.Variable(dtype=tf.float32, name='mean_rgb', trainable=False, initial_value=[128.0, 128.0, 128.0])
        rgb_mean_dims = tf.reshape(mean_rgb_var, shape=[1, 1, 1, 3])
        inputs = tf.subtract(inputs, rgb_mean_dims)
        net_inputs = tl.layers.InputLayer(inputs, name='input_layer')
        if bottle_neck:
            net = conv2d_same(net_inputs, 64, 7, strides=2, rate=1, scope='conv1')
            net = tl.layers.MaxPool2d(net, (2, 2), padding='SAME', name='pool1')
        else:
            raise ValueError('The standard resnet must support the bottleneck layer')
        for block in blocks:
            with tf.variable_scope(block.scope):
                for i, var in enumerate(block.args):
                    with tf.variable_scope('unit_%d' % (i+1)):
                        if type=='resnext':
                            net = block.unit_fn(net, depth=var['depth'], stride=var['stride'],
                                                cardinality=var['cardinality'],
                                                cardinality_dim=var['cardinality_dim'], rate=1, scope=None)
                        else:
                            net = block.unit_fn(net, depth=var['depth'], depth_bottleneck=var['depth_bottleneck'],
                                                stride=var['stride'], rate=var['rate'], scope=None)
        net.outputs = tf.reduce_mean(net.outputs, [1, 2], keep_dims=True)
        net = tl.layers.Conv2d(net, num_classes, filter_size=(1, 1), strides=(1, 1), act=None, name='logits')
        net.outputs = tf.squeeze(net.outputs, [1, 2], name='SpatialSqueeze')
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


def resnet_v1_block(scope, base_depth, num_units, stride, rate=1):
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
  return Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1,
      'rate': rate
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride,
      'rate': rate
  }])


def resnetse_v1_block(scope, base_depth, num_units, stride, rate=1):
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
  return Block(scope, bottleneck_SE, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1,
      'rate': rate
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride,
      'rate': rate
  }])


def resnext_v1_block(scope, base_depth, num_units, stride, cardinality, cardinality_dim, rate=1):
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
  return Block(scope, bottleneck_Xt, [{
      'depth': base_depth * 4,
      'stride': 1,
      'rate': rate,
      'cardinality': cardinality,
      'cardinality_dim': cardinality_dim
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'stride': stride,
      'rate': 1,
      'cardinality': cardinality,
      'cardinality_dim': cardinality_dim
  }])


def get_resnet(inputs, num_classes, num_layers, type='resnet', sess=None, pretrained=True):
    '''
    :param inputs: inputs is an tensorflow placeholder
    :param num_classes:
    :param num_layers:
    :param type: choose weather using se xt or just oridinary resnet  ['resnet', 'resnetse', 'resnext']
    :return:
    '''
    if type == 'resnet':
        if num_layers == 50:
            blocks = [
                resnet_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1),
                resnet_v1_block('block2', base_depth=128, num_units=4, stride=2, rate=1),
                resnet_v1_block('block3', base_depth=256, num_units=6, stride=2, rate=1),
                resnet_v1_block('block4', base_depth=512, num_units=3, stride=1, rate=1)
            ]
            ckpt_file_path = '../model_weights/resnet_v1_50.ckpt'
        elif num_layers == 101:
            blocks = [
                resnet_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1),
                resnet_v1_block('block2', base_depth=128, num_units=4, stride=2, rate=1),
                resnet_v1_block('block3', base_depth=256, num_units=23, stride=2, rate=1),
                resnet_v1_block('block4', base_depth=512, num_units=3, stride=1, rate=1)
            ]
            ckpt_file_path = '../model_weights/resnet_v1_101.ckpt'
        elif num_layers == 152:
            blocks = [
                resnet_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1),
                resnet_v1_block('block2', base_depth=128, num_units=8, stride=2, rate=1),
                resnet_v1_block('block3', base_depth=256, num_units=36, stride=2, rate=1),
                resnet_v1_block('block4', base_depth=512, num_units=3, stride=1, rate=1)
            ]
            ckpt_file_path = '../model_weights/resnet_v1_152.ckpt'
        else:
            raise ValueError('Resnet layer %d is not supported now.' % num_layers)
        net = resnet(inputs=inputs,
                     bottle_neck=True,
                     blocks=blocks,
                     num_classes=num_classes,
                     scope='resnet_v1_%d' % num_layers,
                     type=type)
        if pretrained:
            var_ckpt = get_variables_in_checkpoint_file(ckpt_file_path)
            vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            vars_dict = {}
            for var in vars:
                var_name = var.op.name
                var_name_new = var_name
                if '_bn' in var_name:
                    var_name_new = var_name_new.replace('_bn', '')
                if 'W_conv2d' in var_name:
                    var_name_new = var_name_new.replace('W_conv2d', 'weights')
                if 'b_conv2d' in var_name:
                    var_name_new = var_name_new.replace('b_conv2d', 'biases')
                if 'shortcut_conv' in var_name:
                    var_name_new = var_name_new.replace('shortcut_conv', 'shortcut')
                print(var_name_new)
                if var_name_new in var_ckpt:
                    vars_dict[var_name_new] = var
            tl.layers.initialize_global_variables(sess)
            if len(vars_dict.keys()) > 0:
                saver = tf.train.Saver(vars_dict)
                saver.restore(sess, ckpt_file_path)
            return net
        else:
            return net
    elif type == 'resnetse':
        if pretrained:
            raise ValueError('resnetse have no pretrained model!')
        if num_layers == 50:
            blocks = [
                resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1),
                resnetse_v1_block('block2', base_depth=128, num_units=4, stride=2, rate=1),
                resnetse_v1_block('block3', base_depth=256, num_units=6, stride=2, rate=1),
                resnetse_v1_block('block4', base_depth=512, num_units=3, stride=1, rate=1)
            ]
        elif num_layers == 101:
            blocks = [
                resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1),
                resnetse_v1_block('block2', base_depth=128, num_units=4, stride=2, rate=1),
                resnetse_v1_block('block3', base_depth=256, num_units=23, stride=2, rate=1),
                resnetse_v1_block('block4', base_depth=512, num_units=3, stride=1, rate=1)
            ]
        elif num_layers == 152:
            blocks = [
                resnetse_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1),
                resnetse_v1_block('block2', base_depth=128, num_units=8, stride=2, rate=1),
                resnetse_v1_block('block3', base_depth=256, num_units=36, stride=2, rate=1),
                resnetse_v1_block('block4', base_depth=512, num_units=3, stride=1, rate=1)
            ]
        else:
            raise ValueError('Resnet layer %d is not supported now.' % num_layers)
        net = resnet(inputs=inputs,
                     bottle_neck=True,
                     blocks=blocks,
                     num_classes=num_classes,
                     scope='resnet_v1_%d' % num_layers,
                     type=type)
        return net
    elif type == 'resnext':
        if pretrained:
            raise ValueError('resnetse have no pretrained model!')
        if num_layers == 50:
            blocks = [
                resnext_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, cardinality=32,
                                 cardinality_dim=4),
                resnext_v1_block('block2', base_depth=128, num_units=4, stride=2, rate=1, cardinality=32,
                                 cardinality_dim=4),
                resnext_v1_block('block3', base_depth=256, num_units=6, stride=2, rate=1, cardinality=32,
                                 cardinality_dim=4),
                resnext_v1_block('block4', base_depth=512, num_units=3, stride=1, rate=1, cardinality=32,
                                 cardinality_dim=4)
            ]
        elif num_layers == 101:
            blocks = [
                resnext_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, cardinality=32,
                                 cardinality_dim=4),
                resnext_v1_block('block2', base_depth=128, num_units=4, stride=2, rate=1, cardinality=32,
                                 cardinality_dim=4),
                resnext_v1_block('block3', base_depth=256, num_units=23, stride=2, rate=1, cardinality=32,
                                 cardinality_dim=4),
                resnext_v1_block('block4', base_depth=512, num_units=3, stride=1, rate=1, cardinality=32,
                                 cardinality_dim=4)
            ]
        elif num_layers == 152:
            blocks = [
                resnext_v1_block('block1', base_depth=64, num_units=3, stride=2, rate=1, cardinality=32,
                                 cardinality_dim=4),
                resnext_v1_block('block2', base_depth=128, num_units=8, stride=2, rate=1, cardinality=32,
                                 cardinality_dim=4),
                resnext_v1_block('block3', base_depth=256, num_units=36, stride=2, rate=1, cardinality=32,
                                 cardinality_dim=4),
                resnext_v1_block('block4', base_depth=512, num_units=3, stride=1, rate=1, cardinality=32,
                                 cardinality_dim=4)
            ]
        else:
            raise ValueError('Resnet layer %d is not supported now.' % num_layers)
        net = resnet(inputs=inputs,
                     bottle_neck=True,
                     blocks=blocks,
                     num_classes=num_classes,
                     scope='resnet_v1_%d' % num_layers,
                     type=type)
        return net
    else:
        raise ValueError('resnet type %s does not support now!' % type)


if __name__ == '__main__':
        x = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='input_place')
        sess = tf.Session()
        # # test resnet
        # nets = get_resnet(x, 1000, 50, type='resnet', sess=sess, pretrained=True)
        # with sess:
        #     nets.print_params()

        # # test resnetse
        # nets = get_resnet(x, 1000, 50, type='resnetse', sess=sess, pretrained=False)
        # tl.layers.initialize_global_variables(sess)
        # with sess:
        #     nets.print_params()

        # test resnext
        nets = get_resnet(x, 1000, 50, type='resnext', sess=sess, pretrained=False)
        tl.layers.initialize_global_variables(sess)
        with sess:
            nets.print_params()