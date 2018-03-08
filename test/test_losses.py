import tensorflow as tf
import numpy as np
from losses.face_losses import cosineface_losses
import mxnet as mx
import math


def arcface_loss_val(embedding, labels, weights, out_num, s=64., m=0.5):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m * s
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
    return output


def test_arcface_losses(np_embedding, np_weights):
    tf_embedding = tf.constant(np_embedding, name='embedding', dtype=tf.float32)
    labels = tf.constant([1, 3, 2, 1, 1], name='input_labels', dtype=tf.int64)
    print(labels)
    tf_weights = tf.constant(np_weights, name='weights')
    output = arcface_loss_val(embedding=tf_embedding, labels=labels, out_num=10, weights=tf_weights)
    print(output)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    results1 = sess.run(output)
    print(results1)
    return results1


def test_cosineface_losses():
    np_embedding = np.random.randn(5, 512).astype(dtype=np.float32)
    tf_embedding = tf.constant(np_embedding, name='embedding', dtype=tf.float32)
    labels = tf.constant([1, 3, 2, 1, 1], name='input_labels', dtype=tf.int64)
    output = cosineface_losses(embedding=tf_embedding, labels=labels, out_num=10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(output))


def test_mxnet_losses(np_embedding, np_weights):
    labels = np.array([1, 3, 2, 1, 1]).astype(dtype=np.float32)
    return mxnet_arcface_val(np_embedding, labels, np_weights)


def mxnet_arcface_val(embedding, gt_label, weights):
    s = 64
    m = 0.5
    _weight = mx.symbol.Variable("fc7_weight", shape=(10, 512), lr_mult=1.0)
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    _embedding = mx.symbol.Variable('mx_embedding', shape=(5, 512), lr_mult=1.0)
    nembedding = mx.symbol.L2Normalization(_embedding, mode='instance', name='fc1n')*s
    fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True, num_hidden=10, name='fc7')

    _labels = mx.symbol.Variable('labels', shape=(5, ), lr_mult=1.0)
    zy = mx.sym.pick(fc7, _labels, axis=1)
    cos_t = zy/s

    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = math.sin(math.pi - m) * m
    # threshold = 0.0
    threshold = math.cos(math.pi - m)

    cond_v = cos_t - threshold
    cond = mx.symbol.Activation(data=cond_v, act_type='relu')

    body = cos_t * cos_t
    body = 1.0 - body
    sin_t = mx.sym.sqrt(body)
    new_zy = cos_t * cos_m
    b = sin_t * sin_m
    new_zy = new_zy - b
    new_zy = new_zy * s

    zy_keep = zy - s * mm
    new_zy = mx.sym.where(cond, new_zy, zy_keep)

    diff = new_zy - zy
    diff = mx.sym.expand_dims(diff, 1)
    gt_one_hot = mx.sym.one_hot(_labels, depth = 10, on_value = 1.0, off_value = 0.0)
    body = mx.sym.broadcast_mul(gt_one_hot, diff)
    fc7 = fc7+body
    executor = fc7.bind(mx.cpu(), {'fc7_weight': mx.nd.array(weights.T), 'mx_embedding': mx.nd.array(embedding),
                                   'labels': mx.nd.array(gt_label)})
    output = executor.forward()
    print(output)
    return output


if __name__ == '__main__':
    np_embedding = np.random.randn(5, 512).astype(dtype=np.float32)
    np_weights = np.random.randn(512, 10).astype(dtype=np.float32)
    # test arcface_losses output
    result1 = test_arcface_losses(np_embedding, np_weights)
    # print('########'*30)
    print('################')
    result2 = test_mxnet_losses(np_embedding, np_weights)
    print(len(result2[0]))
    print(type(result1))
    print(type(result2[0].asnumpy()))
    print(np.mean(result1 - result2[0].asnumpy()))   # 1.26362e-07