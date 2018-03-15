import mxnet as mx
import mxnet.ndarray as nd
import os


if __name__ == '__main__':
    # without bn and prelu  max batchsize (40000, 50000)
    # with bn max batchsize (20000, 30000)
    # with prelu batchsize (20000, 30000)
    # with bn and prelu max batchsize (10000, 20000)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    batch_size = 10000
    mnist = mx.test_utils.get_mnist()
    print(mnist['train_data'].shape)
    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)

    # inference
    data = mx.sym.var('data')
    # first conv layer
    net = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=64)
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, name='_bn1')
    net = mx.sym.LeakyReLU(data=net, act_type='prelu', name='_preul1')
    net = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=64)
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, name='_bn2')
    net = mx.sym.LeakyReLU(data=net, act_type='prelu', name='_preul2')
    net = mx.sym.Convolution(data=data, kernel=(3, 3), stride=(2, 2), num_filter=64)
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, name='_bn3')
    net = mx.sym.LeakyReLU(data=net, act_type='prelu', name='_preul3')

    net = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=128)
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, name='_bn4')
    net = mx.sym.LeakyReLU(data=net, act_type='prelu', name='_preul4')
    net = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=128)
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, name='_bn5')
    net = mx.sym.LeakyReLU(data=net, act_type='prelu', name='_preul5')
    net = mx.sym.Convolution(data=data, kernel=(3, 3), stride=(2, 2), num_filter=128)
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, name='_bn6')
    net = mx.sym.LeakyReLU(data=net, act_type='prelu', name='_preul6')

    net = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=256)
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, name='_bn7')
    net = mx.sym.LeakyReLU(data=net, act_type='prelu', name='_preul7')
    net = mx.sym.Convolution(data=data, kernel=(3, 3), num_filter=256)
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, name='_bn8')
    net = mx.sym.LeakyReLU(data=net, act_type='prelu', name='_preul8')
    net = mx.sym.Convolution(data=data, kernel=(3, 3), stride=(2, 2), num_filter=256)
    net = mx.sym.BatchNorm(data=net, fix_gamma=False, eps=2e-5, name='_bn9')
    net = mx.sym.LeakyReLU(data=net, act_type='prelu', name='_preul9')

    flatten = mx.sym.flatten(data=net)
    # MNIST has 10 classes
    fc3 = mx.sym.FullyConnected(data=flatten, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

    import logging

    logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
    # create a trainable module on GPU
    mlp_model = mx.mod.Module(symbol=mlp, context=mx.gpu())
    mlp_model.fit(train_iter,  # train data
                  optimizer='sgd',  # use SGD to train
                  optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
                  eval_metric='acc',  # report accuracy during training
                  batch_end_callback=mx.callback.Speedometer(batch_size, 100),
                  # output progress for each 100 data batches
                  num_epoch=10)  # train for at most 10 dataset passes