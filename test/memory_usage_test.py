import mxnet as mx
import argparse
import PIL.Image
import io
import numpy as np
import cv2
import tensorflow as tf
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='data path information'
    )
    parser.add_argument('--bin_path', default='../datasets/faces_ms1m_112x112/train.rec', type=str,
                        help='path to the binary image file')
    parser.add_argument('--idx_path', default='../datasets/faces_ms1m_112x112/train.idx', type=str,
                        help='path to the image index path')
    parser.add_argument('--tfrecords_file_path', default='../datasets/tfrecords', type=str,
                        help='path to the output of tfrecords file path')
    args = parser.parse_args()
    return args


def mx2tfrecords_mem_test(imgidx, imgrec, args):
    output_path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        print(type(img))
        print(img)
        print(sys.getsizeof(img))
        print('#####################')
        img_mx = mx.image.imdecode(img)
        print(type(img_mx))
        print(sys.getsizeof(img_mx))
        print(img_mx.size)
        print(img_mx.dtype)
        print(img_mx.context)
        print(img_mx.stype)
        print(img_mx)
        print('#####################')
        img_mx_np = img_mx.asnumpy()
        print(type(img_mx_np))
        print(sys.getsizeof(img_mx_np))
        print('#####################')
        back_mx_ndarray = mx.nd.array(img_mx_np)
        print(type(back_mx_ndarray))
        print(sys.getsizeof(back_mx_ndarray))
        encoded_jpg_io = io.BytesIO(img)
        print(sys.getsizeof(encoded_jpg_io))
        image = PIL.Image.open(encoded_jpg_io)
        np_img = np.array(image)
        img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        print(sys.getsizeof(img))
        print('#####################')
        img_raw = img.tobytes()
        print(sys.getsizeof(img))
        print('#####################')
    writer.close()


def mx2tfrecords(imgidx, imgrec, args):
    output_path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        # encoded_jpg_io = io.BytesIO(img)
        # image = PIL.Image.open(encoded_jpg_io)
        # np_img = np.array(image)
        # img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        # img_raw = img.tobytes()
        # images = tf.image.decode_jpeg(img)
        # images = tf.reshape(images, shape=(112, 112, 3))
        # r, g, b = tf.split(images, num_or_size_splits=3, axis=-1)
        # images = tf.concat([b, g, r], axis=-1)
        # sess = tf.Session()
        # np_images = sess.run(images)
        # print(images.shape)
        # print(type(np_images))
        # print(sys.getsizeof(np_images))
        # cv2.imshow('test', np_images)
        # cv2.waitKey(0)
        label = int(header.label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    writer.close()


if __name__ == '__main__':
    # define parameters
    id2range = {}
    data_shape = (3, 112, 112)
    args = parse_args()
    imgrec = mx.recordio.MXIndexedRecordIO(args.idx_path, args.bin_path, 'r')
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    print(header.label)
    imgidx = list(range(1, int(header.label[0])))
    seq_identity = range(int(header.label[0]), int(header.label[1]))
    for identity in seq_identity:
        s = imgrec.read_idx(identity)
        header, _ = mx.recordio.unpack(s)
        a, b = int(header.label[0]), int(header.label[1])
        id2range[identity] = (a, b)
    print('id2range', len(id2range))

    # generate tfrecords
    mx2tfrecords_mem_test(imgidx, imgrec, args)




