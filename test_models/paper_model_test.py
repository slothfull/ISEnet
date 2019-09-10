#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@time:2019/5/10 上午10:08
@author:bigmelon
按照训练的形式测试数据 逐个batch送入数据 直接使用np_dataset中的test数据
"""
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from test_models.testingdata import TestingData
from sklearn.metrics import classification_report, confusion_matrix

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


aa = 'ch01/i17/res1/art'
bb = '/highest'
# bb = ''
data_dir = '/share/donghao/data2/exp2/np_dataset'
batch_size = 8
print('[i] Configuring the training data...')
td = None
try:
    td = TestingData(data_dir, batch_size)
    print('[i] training samples:             ', td.num_train)
    print('[i] testing samples:              ', td.num_test)
    print('[i] classes:                      ', td.num_classes)
    print('[i] ecg_chip size:                ', f'({td.sample_width}, {td.sample_length})')
except (AttributeError, RuntimeError) as e:
    exit('[!] Unable to load training data: ' + str(e))

#              194 or 195 达到了0.8878的test_acc -> 在highest中存放了195的checkpoint
#              188 or 189 达到了0.8862的test_acc -> 在highest中存放了189的checkpoint
#              275 or 276 达到了0.8829的test_acc -> 在highest中存放了276的checkpoint
#              336 or 337 达到了0.8770的test_acc -> 在highest中存放了337的checkpoint
#              347 or 348 达到了0.8804的test_acc -> 在highest中存放了348的checkpoint
with tf.Session(config=config) as sess:
    ch_dir = f'/share/donghao/demo10/exp2/trained_models/{aa}/sesenet'+f'{bb}'
    metagraph_file = f'/share/donghao/demo10/exp2/trained_models/{aa}/sesenet' + f'{bb}' + '/e199.ckpt.meta'
    state = tf.train.get_checkpoint_state(checkpoint_dir=ch_dir, latest_filename=None)
    ckpt_paths = state.all_model_checkpoint_paths
    saver = tf.train.import_meta_graph(metagraph_file)  # import metagraph
    saver.restore(sess, ckpt_paths[-1])  # restore
    xs = sess.graph.get_tensor_by_name('inputs/x_input:0')  # (?,1,512,2)
    ys = sess.graph.get_tensor_by_name('inputs/y_input:0')  # (?,5)
    y_predict = sess.graph.get_tensor_by_name('compute_accuracy/ArgMax:0')  # (?,) [0, 0, 1, 3, 4]

    # Train ->
    y_lst = []
    predict_lst = []
    td.train_iter(process='train', num_epoch=1)
    description = '[i] Train {:>2}/{}'.format(1, 1)  # epoch_No/total_epoch_No -> e+1/200
    for _ in tqdm(iterable=td.train_tqdm_iter, total=td.num_train_iter, desc=description, unit='batches'):
        x, y = sess.run(td.train_sample)  # array(?,1,512,2) array(?,5)
        train_dict = {xs: x, ys: y}
        predict = sess.run(y_predict, feed_dict=train_dict)  # np_array (?,)
        predict = list(predict)
        y = list(np.argmax(y, axis=1))
        # add up
        predict_lst.extend(predict)
        y_lst.extend(y)
    print(classification_report(y_lst, predict_lst, [0, 1, 2, 3, 4]))
    print(confusion_matrix(y_lst, predict_lst, [0, 1, 2, 3, 4]))
    y_lst = []
    predict_lst = []
    # Test ->
    td.test_iter(process='test_models', num_epoch=1)
    description = '[i] Test {:>2}/{}'.format(1, 1)
    for _ in tqdm(iterable=td.test_tqdm_iter, total=td.num_test_iter, desc=description, unit='batches'):
        x, y = sess.run(td.test_sample)  # array(?,1,512,2) array(?,5)
        test_dict = {xs: x, ys: y}
        predict = sess.run(y_predict, feed_dict=test_dict)
        predict = list(predict)
        y = list(np.argmax(y, axis=1))
        # add up
        predict_lst.extend(predict)
        y_lst.extend(y)
    print(classification_report(y_lst, predict_lst, [0, 1, 2, 3, 4]))
    print(confusion_matrix(y_lst, predict_lst, [0, 1, 2, 3, 4]))
