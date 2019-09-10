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
import pickle
from tqdm import tqdm
from testingdata2 import TestingData
from sklearn.metrics import classification_report, confusion_matrix
from namedtuples import SAMPLE
from q_utils import dense_to_one_hot

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# yp (5) todo (66)

# aa = 'ch01/i17/res1/art'
aa = 'record/i14_124'
checkno = 160
# bb = '/highest'
bb = ''
# np_datasetall(89647/4)=22411  np_dataset24(29725)(14862) np_dataset14s(33853/2=16926) np_dataset11v(26915/2=13457)
# => 精心调整的filter
# np_datasetlla(89647/4)=22411  np_dataset42(29725)(14862) np_dataset41s(33853/2=16926) np_datasetv11(26915/2=13457)
# => 使用精心调整的filter + normalized into [0,1]的数据
# np_datasetlal(89647/4)=22411  np_dataset4r(29725)(14862) np_dataset1s4(33853/2=16926) np_dataset1v1(26915/2=13457)
# data_dir = '/share/donghao/data/patient_specific/np_dataset1v1'
data_dir = '/share/donghao/data2/exp2/per_record_dataset/124_dataset'

# '/share/donghao/data2/'+exp+f'/per_record_dataset/{no}_dataset'
# batch_size = 16926 if 's' in data_dir[-1:-4:-1] else 22411 if 'a' in data_dir[-1:-3:-1] else 13457 if 'v' in data_dir[-1:-3:-1] else 14862
batch_size = 100
print('[i] Configuring the testing data...')
td = None
try:
    td = TestingData(data_dir, batch_size)
    print('[i] testing samples:              ', td.num_test)
    print('[i] classes:                      ', td.num_classes)
    print('[i] ecg_chip size:                ', f'({td.sample_width}, {td.sample_length})')
except (AttributeError, RuntimeError) as e:
    exit('[!] Unable to load training data: ' + str(e))

with tf.Session(config=config) as sess:
    ch_dir = f'/share/donghao/demo10/exp2/trained_models/{aa}/sesenet' + f'{bb}'
    metagraph_file = f'/share/donghao/demo10/exp2/trained_models/{aa}/sesenet' + f'{bb}' + f'/e{checkno}.ckpt.meta'
    state = tf.train.get_checkpoint_state(checkpoint_dir=ch_dir, latest_filename=None)
    ckpt_paths = state.all_model_checkpoint_paths
    saver = tf.train.import_meta_graph(metagraph_file)  # import metagraph
    saver.restore(sess, ckpt_paths[-1])  # restore
    xs = sess.graph.get_tensor_by_name('inputs/x_input:0')  # (?,1,512,2)
    ys = sess.graph.get_tensor_by_name('inputs/y_input:0')  # (?,5)
    process_tag = sess.graph.get_tensor_by_name('inputs/tag:0')
    y_predict = sess.graph.get_tensor_by_name('compute_accuracy/ArgMax:0')  # (?,) [0, 0, 1, 3, 4]

    y_lst = []
    predict_lst = []
    # total wrong samples
    supply_sample_list_np = []
    sample_id = 0
    # sub set of total wrong samples
    # todo rate_n=rate_s=500 -> 1n 6s
    # todo rate_n=rate_s=600 -> 1n 5s
    # todo rate_n=100 rate_s=600  2n  5s
    # todo rate_n=100 rate_s=1200 -> 2s+2n
    rate_n = 10000  # 每隔500个sample抽取一个作为sub set  +1n -1s
    rate_s = 32  #
    sub_supply_sample_list_np = []
    sub_sample_id = 0
    #
    __ = 0
    ___ = 0
    # supply/sub_supply label
    sub_label = []
    label = []
    # Test ->
    td.test_iter(process='test', num_epoch=1)
    description = '[i] Test {:>2}/{}'.format(1, 1)
    for _ in tqdm(iterable=td.test_tqdm_iter, total=td.num_test_iter, desc=description, unit='batches'):
        x, y = sess.run(td.test_sample)  # array(?,1,512,2) array(?,5)
        test_dict = {xs: x, ys: y, process_tag: False}  # (1,1,512,2)
        # test_dict = {xs: x, ys: y}  # (1,1,512,2)
        predict = sess.run(y_predict, feed_dict=test_dict)
        predict = list(predict)  # (?,)
        y = list(np.argmax(y, axis=1))  # (?,)
        # todo
        # if y[0] == 0 and predict[0] != y[0]:  # if n
        #     sample_path_np = f'/share/donghao/data2/exp2/supply/train/{sample_id}.npy'
        #     supply_sample_list_np.append(SAMPLE(sample_path_np, y))
        #     sample_id = sample_id + 1
        #     np.save(sample_path_np, np.squeeze(x, axis=0))
        #     label.append(dense_to_one_hot(np.array(y), 5))
        #     if __ % rate_n == 0 and __ != 0:
        #         print('n')
        #         sub_sample_path_np = f'/share/donghao/data2/exp2/sub_supply/train/{sub_sample_id}.npy'
        #         sub_supply_sample_list_np.append(SAMPLE(sub_sample_path_np, y))
        #         sub_sample_id = sub_sample_id + 1
        #         np.save(sub_sample_path_np, np.squeeze(x, axis=0))
        #         sub_label.append(dense_to_one_hot(np.array(y), 5))
        #     __ = __ + 1
        # elif y[0] == 1 and predict[0] != y[0]:  # if s
        #     sample_path_np = f'/share/donghao/data2/exp2/supply/train/{sample_id}.npy'
        #     supply_sample_list_np.append(SAMPLE(sample_path_np, y))
        #     sample_id = sample_id + 1
        #     np.save(sample_path_np, np.squeeze(x, axis=0))
        #     label.append(dense_to_one_hot(np.array(y), 5))
        #     if ___ % rate_s == 0:
        #         print('s')
        #         sub_sample_path_np = f'/share/donghao/data2/exp2/sub_supply/train/{sub_sample_id}.npy'
        #         sub_supply_sample_list_np.append(SAMPLE(sub_sample_path_np, y))
        #         sub_sample_id = sub_sample_id + 1
        #         np.save(sub_sample_path_np, np.squeeze(x, axis=0))
        #         sub_label.append(dense_to_one_hot(np.array(y), 5))
        #     ___ = ___ + 1
        # todo
        # add up
        predict_lst.extend(predict)
        y_lst.extend(y)
    print(sample_id)
    print(sub_sample_id)
    # todo
    # with open('/share/donghao/data2/exp2/supply/TRAIN_SAMPLE_LIST.pkl', 'wb') as f:
    #     pickle.dump(supply_sample_list_np, f)
    # with open('/share/donghao/data2/exp2/sub_supply/TRAIN_SAMPLE_LIST.pkl', 'wb') as f:
    #     pickle.dump(sub_supply_sample_list_np, f)
    # np.save('/share/donghao/data2/exp2/sub_supply/label.npy', np.squeeze(sub_label))  # (?,1,5)->(?,5)
    # np.save('/share/donghao/data2/exp2/supply/label.npy', np.squeeze(label))  # (?,1,5)->(?,5)
    # todo
    cr = classification_report(y_lst, predict_lst, [0, 1, 2, 3, 4], digits=3)  # str
    cm = confusion_matrix(y_lst, predict_lst, [0, 1, 2, 3, 4])  # arr
    print(cr)
    print(cm)

    print('[', end=' ')
    for _ in cm:
        print('[', end=' ')
        for __ in _:
            print(str(__) + ',', end=' ')
        print('],')
    print(']', end=' ')

# todo 更具rp 和 yp的对照实验 决定使用 normalized_data[0-1]中的yp_dataset进行实验... 通常recall更好
# todo => 再通过使用supply数据将 best model->74-78之间
# yp 66
# 24
# [[49914    84   109    61     0]
#  [ 1259  1504    46     3     0]
#  [  439     5  5206    74     0]
#  [   57     0    34   698     0]
#  [    6     1     0     0     0]]
# 14s
# [[28180    37    25    54     0]
#  [  619  1353    21     2     0]
#  [   99     4  3023    46     0]
#  [   20     0    18   350     0]
#  [    2     0     0     0     0]]

