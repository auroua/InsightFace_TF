from resnet import get_resnet
import tensorflow as tf
from nets_utils import get_tensor_static_val
import numpy as np


def resnet_diff_test(layers_num):
    ckpt_file_path = '../model_weights/resnet_v1_'+str(layers_num)+'.ckpt'
    x = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='input_place')
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=tfconfig)
    nets = get_resnet(x, 1000, layers_num, sess)
    ckpt_static = get_tensor_static_val(ckpt_file_path, all_tensors=True, all_tensor_names=True)

    print('###########'*30)
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    total_count = 0
    mean_avg = 0.0
    median_avg = 0.0
    std_avg = 0.0

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

        if var_name_new in ckpt_static:
            print(var_name_new, end=',    ')
            total_count += 1
            ckpt_s = ckpt_static[var_name_new]
            var_val = sess.run(var)
            mean_diff = np.mean(var_val) - ckpt_s.mean
            mean_avg += mean_diff
            median_diff = np.median(var_val) - ckpt_s.median
            median_avg += median_diff
            std_diff = np.std(var_val) - ckpt_s.std
            std_avg += std_diff
            print('mean_diff: ', mean_diff, 'median_diff: ', median_diff, 'std_diff: ', std_diff)

    print('total_mean_diff', mean_avg/total_count, 'total_mean_diff', median_avg/total_count,
          'total_std_diff', std_avg/total_count)


if __name__ == '__main__':
    with tf.device('/device:GPU:1'):
        resnet_diff_test(50)
