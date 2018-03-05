import tensorflow as tf
from vgg16 import get_vgg16
from vgg19 import get_vgg19


def get_model(inputs, sess, type, pretrained=True):
    if type == 'vgg16':
        return get_vgg16(inputs, sess, pretrained)
    elif type == 'vgg19':
        return get_vgg19(inputs, sess, pretrained)


if __name__ == '__main__':
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='inpust')
    with tf.Session(config=tfconfig) as sess:
        network = get_model(x, sess, type='vgg19', pretrained=True)
        network.print_params()
        network.print_layers()