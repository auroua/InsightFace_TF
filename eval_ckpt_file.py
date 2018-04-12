import tensorflow as tf
import argparse
from data.eval_data_reader import load_bin
from losses.face_losses import arcface_loss
from nets.L_Resnet_E_IR_GBN import get_resnet
# from nets.L_Resnet_E_IR import get_resnet
import tensorlayer as tl
from verification import ver_test


def get_args():
    parser = argparse.ArgumentParser(description='input information')
    parser.add_argument('--ckpt_file', default='/home/aurora/workspaces12/PycharmProjects/InsightFace_TF/output/ckpt/InsightFace_iter_',
    # parser.add_argument('--ckpt_file', default='/home/aurora/workspaces12/PycharmProjects/InsightFace_TF/output/ckpt2/InsightFace_iter_best_',
                       type=str, help='the ckpt file path')
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--net_depth', default=50, help='resnet depth, default is 50')
    parser.add_argument('--num_output', default=85164, help='the image size')
    parser.add_argument('--batch_size', default=32, help='batch size to train network')
    # parser.add_argument('--ckpt_index_list', default=['10000.ckpt', '20000.ckpt', '30000.ckpt', '40000.ckpt', '50000.ckpt'
    #                                                   , '60000.ckpt', '70000.ckpt', '80000.ckpt', '90000.ckpt', '100000.ckpt',
    #                                                   '110000.ckpt', '120000.ckpt', '130000.ckpt', '140000.ckpt',
    #                                                   '150000.ckpt'], help='ckpt file indexes')
    # parser.add_argument('--ckpt_index_list', default=['180000.ckpt', '190000.ckpt', '200000.ckpt', '210000.ckpt', '220000.ckpt'], help='ckpt file indexes')
    # parser.add_argument('--ckpt_index_list', default=['270000.ckpt', '280000.ckpt', '290000.ckpt', '300000.ckpt', '310000.ckpt'], help='ckpt file indexes')
    parser.add_argument('--ckpt_index_list', default=['730000.ckpt'], help='ckpt file indexes')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    ver_list = []
    ver_name_list = []
    for db in args.eval_datasets:
        print('begin db %s convert.' % db)
        data_set = load_bin(db, args.image_size, args)
        ver_list.append(data_set)
        ver_name_list.append(db)

    images = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
    labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
    trainable = tf.placeholder(name='trainable_bn', dtype =tf.bool)

    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    net = get_resnet(images, args.net_depth, type='ir', w_init=w_init_method, trainable=trainable)
    embedding_tensor = net.outputs
    # 3.2 get arcface loss
    logit = arcface_loss(embedding=net.outputs, labels=labels, w_init=w_init_method, out_num=args.num_output)

    sess = tf.Session()
    saver = tf.train.Saver()

    result_index = []
    for file_index in args.ckpt_index_list:
        path = args.ckpt_file + file_index
        saver.restore(sess, path)
        print('ckpt file %s restored!' % file_index)

        feed_dict_test = {trainable: False}
        feed_dict_test.update(tl.utils.dict_to_one(net.all_drop))
        results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=sess,
                           embedding_tensor=embedding_tensor, batch_size=args.batch_size, feed_dict=feed_dict_test,
                           input_placeholder=images)
        result_index.append(results)
    print(result_index)
