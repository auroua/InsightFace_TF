## Insight Face in TensorFlow

#### Tasks
* ~~mxnet dataset to tfrecords~~
* ~~backbone network architectures [vgg16, vgg19, resnet]~~
* ~~backbone network architectures [resnet-se, resnext]~~
* ~~LResNet50E-IR~~
* ~~LResNet100E-IR~~
* ~~Additive Angular Margin Loss~~
* ~~CosineFace Loss~~
* ~~train network code~~
* evaluate code

#### Training Logs
```
epoch 0, total_step 20, total loss is 107.34 , inference loss is 80.60, weight deacy loss is 26.74, training accuracy is 0.000000, time 38.373 samples/sec
epoch 0, total_step 40, total loss is 109.65 , inference loss is 77.31, weight deacy loss is 32.34, training accuracy is 0.000000, time 38.281 samples/sec
epoch 0, total_step 60, total loss is 114.86 , inference loss is 82.29, weight deacy loss is 32.57, training accuracy is 0.000000, time 37.687 samples/sec
epoch 0, total_step 80, total loss is 104.92 , inference loss is 72.77, weight deacy loss is 32.15, training accuracy is 0.000000, time 38.402 samples/sec
epoch 0, total_step 100, total loss is 101.66 , inference loss is 69.99, weight deacy loss is 31.67, training accuracy is 0.000000, time 38.235 samples/sec
epoch 0, total_step 120, total loss is 101.70 , inference loss is 70.54, weight deacy loss is 31.16, training accuracy is 0.000000, time 37.822 samples/sec
epoch 0, total_step 140, total loss is 102.23 , inference loss is 71.61, weight deacy loss is 30.63, training accuracy is 0.000000, time 38.308 samples/sec
epoch 0, total_step 160, total loss is 103.26 , inference loss is 73.17, weight deacy loss is 30.08, training accuracy is 0.000000, time 38.054 samples/sec
epoch 0, total_step 180, total loss is 98.61 , inference loss is 69.07, weight deacy loss is 29.54, training accuracy is 0.000000, time 38.198 samples/sec
epoch 0, total_step 200, total loss is 95.20 , inference loss is 66.16, weight deacy loss is 29.04, training accuracy is 0.000000, time 38.217 samples/sec
```

#### Requirements
1. TensorFlow 1.4 1.6
2. TensorLayer 1.7


#### pretrained model download link
* [resnet_v1_50](download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)
* [resnet_v1_101](download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz)
* [resnet_v1_152](download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz)
* [vgg16](http://www.cs.toronto.edu/~frossard/post/vgg16/)
* [vgg19](https://github.com/machrisaa/tensorflow-vgg)


#### References
1. [InsightFace mxnet](https://github.com/deepinsight/insightface)
2. [InsightFace : Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
3. [tensorlayer_vgg16](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_vgg16.py)
4. [tensorlayer_vgg19](https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_vgg19.py)
5. [tf_slim](https://github.com/tensorflow/models/tree/master/research/slim)
6. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
7. [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
8. [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)