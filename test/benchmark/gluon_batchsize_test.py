from mxnet import gluon
import mxnet as mx
from mxnet import ndarray as nd
import utils_final as utils
import mxnet.gluon.nn as nn
from mxnet import init
import os
from mxnet import initializer
from mxnet.gluon.block import HybridBlock


def prelu():
    pass


def inference():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(nn.Conv2D(channels=64, kernel_size=3, padding=1))
        net.add(nn.BatchNorm(axis=1, center=True, scale=True))
        # net.add(mx.sym.LeakyReLU(data=net, act_type='prelu', name='prelu1'))
        net.add(nn.Conv2D(channels=64, kernel_size=3, padding=1))
        net.add(nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(nn.Conv2D(channels=64, kernel_size=3, padding=1, strides=2))
        net.add(nn.BatchNorm(axis=1, center=True, scale=True))

        net.add(nn.Conv2D(channels=128, kernel_size=3, padding=1))
        net.add(nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(nn.Conv2D(channels=128, kernel_size=3, padding=1))
        net.add(nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(nn.Conv2D(channels=128, kernel_size=3, padding=1, strides=2))
        net.add(nn.BatchNorm(axis=1, center=True, scale=True))

        net.add(nn.Conv2D(channels=256, kernel_size=3, padding=1))
        net.add(nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(nn.Conv2D(channels=256, kernel_size=3, padding=1))
        net.add(nn.BatchNorm(axis=1, center=True, scale=True))
        net.add(nn.Conv2D(channels=256, kernel_size=3, padding=1, strides=2))
        net.add(nn.BatchNorm(axis=1, center=True, scale=True))

        net.add(nn.Flatten())
        net.add(nn.Dense(10))
    return net


if __name__ == '__main__':
    # without prelu and bn    7000< max batch size <8000
    # with bn only            3000< max batch size <4000
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    batch_size = 3000
    train_data, test_data = utils.load_data_mnist(batch_size=batch_size)
    ctx = utils.try_gpu()
    net = inference()
    print(net)
    net.initialize(ctx=ctx, init=init.Xavier())
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
    utils.train(train_data, test_data, net, softmax_cross_entropy, trainer, ctx, num_epochs=10)


