import tensorflow as tf
import numpy as np
from losses.face_losses import arcface_loss, cosineface_losses
import mxnet as mx
import math


def test_arcface_losses():
    np_embedding = np.random.randn(5, 512).astype(dtype=np.float32)
    tf_embedding = tf.constant(np_embedding, name='embedding', dtype=tf.float32)
    labels = tf.constant([[1], [3], [2], [1], [1]], name='input_labels', dtype=tf.int64)
    output = arcface_loss(embedding=tf_embedding, labels=labels, out_num=10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(output))


def test_cosineface_losses():
    np_embedding = np.random.randn(5, 512).astype(dtype=np.float32)
    tf_embedding = tf.constant(np_embedding, name='embedding', dtype=tf.float32)
    labels = tf.constant([[1], [3], [2], [1], [1]], name='input_labels', dtype=tf.int64)
    output = cosineface_losses(embedding=tf_embedding, labels=labels, out_num=10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(output))


def mxnet_arcface_val(embedding, gt_label):
    s = 64
    m = 0.5
    _weight = mx.symbol.Variable("fc7_weight", shape=(10, 512), lr_mult=1.0)
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=10, name='fc7')
    zy = mx.sym.pick(fc7, gt_label, axis=1)
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
    gt_one_hot = mx.sym.one_hot(gt_label, depth = 10, on_value = 1.0, off_value = 0.0)
    body = mx.sym.broadcast_mul(gt_one_hot, diff)
    fc7 = fc7+body
    return fc7


if __name__ == '__main__':
    # test arcface_losses output
    test_arcface_losses()
    print('########'*30)
    # test cosine face losses output
    # test_cosineface_losses()