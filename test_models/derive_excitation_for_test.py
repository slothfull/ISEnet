#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@time:2019/5/10 上午10:08
@author:bigmelon
将n s v f q各500个数据 得到相应的 excitation summary 然后 分别进行平均 得到最终的excitation数据 用于展示
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# yp (5) todo (66)

aa = 'i14d'
# aa = 'record/i14_124'
checkno = 23
bb = '/highest'
# bb = ''
# np_datasetall(89647/4)=22411  np_dataset24(29725)(14862) np_dataset14s(33853/2=16926) np_dataset11v(26915/2=13457)
# => 精心调整的filter
# np_datasetlla(89647/4)=22411  np_dataset42(29725)(14862) np_dataset41s(33853/2=16926) np_datasetv11(26915/2=13457)
# => 使用精心调整的filter + normalized into [0,1]的数据
# np_datasetlal(89647/4)=22411  np_dataset4r(29725)(14862) np_dataset1s4(33853/2=16926) np_dataset1v1(26915/2=13457)
data_dir = '/share/donghao/data/patient_specific/np_datasetlal'
# data_dir = '/share/donghao/data2/exp2/per_record_dataset/124_dataset'

# '/share/donghao/data2/'+exp+f'/per_record_dataset/{no}_dataset'
# batch_size = 16926 if 's' in data_dir[-1:-4:-1] else 22411 if 'a' in data_dir[-1:-3:-1] else 13457 if 'v' in data_dir[-1:-3:-1] else 14862
batch_size = 10000
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

    # supply/sub_supply label
    n_lst = []
    s_lst = []
    v_lst = []
    f_lst = []

    sese_chosen_excitations = ['resnet_v2_14_1d_dual_channel/block1/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0',
                               'resnet_v2_14_1d_dual_channel/block2/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0',
                               'resnet_v2_14_1d_dual_channel/block3/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0',
                               'resnet_v2_14_1d_dual_channel/block4/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0']

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
        # todo 将每次正确分类的n s v f q 分别保存 500 500 500 500

        for index, ele in enumerate(predict):
            if ele == y[index]:
                if ele == 0:
                    dict2 = {xs: np.expand_dims(x[index], axis=0), process_tag: False}
                    e1 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[0])), feed_dict=dict2)
                    e2 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[1])), feed_dict=dict2)
                    e3 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[2])), feed_dict=dict2)
                    e4 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[3])), feed_dict=dict2)
                    n_lst.append((e1, e2, e3, e4))
                elif ele == 1:
                    dict2 = {xs: np.expand_dims(x[index], axis=0), process_tag: False}
                    e1 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[0])), feed_dict=dict2)
                    e2 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[1])), feed_dict=dict2)
                    e3 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[2])), feed_dict=dict2)
                    e4 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[3])), feed_dict=dict2)
                    s_lst.append((e1, e2, e3, e4))
                elif ele == 2:
                    dict2 = {xs: np.expand_dims(x[index], axis=0), process_tag: False}
                    e1 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[0])), feed_dict=dict2)
                    e2 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[1])), feed_dict=dict2)
                    e3 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[2])), feed_dict=dict2)
                    e4 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[3])), feed_dict=dict2)
                    v_lst.append((e1, e2, e3, e4))
                elif ele == 3:
                    dict2 = {xs: np.expand_dims(x[index], axis=0), process_tag: False}
                    e1 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[0])), feed_dict=dict2)
                    e2 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[1])), feed_dict=dict2)
                    e3 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[2])), feed_dict=dict2)
                    e4 = sess.run(sess.graph.get_tensor_by_name((sese_chosen_excitations[3])), feed_dict=dict2)
                    f_lst.append((e1, e2, e3, e4))
                else:
                    n_lst = None; s_lst = None; v_lst = None; f_lst = None; q_lst = None

with open('/share/donghao/demo10/test_models/n.pkl', 'wb') as f:
    pickle.dump(n_lst, f)
with open('/share/donghao/demo10/test_models/s.pkl', 'wb') as f:
    pickle.dump(s_lst, f)
with open('/share/donghao/demo10/test_models/v.pkl', 'wb') as f:
    pickle.dump(v_lst, f)
with open('/share/donghao/demo10/test_models/f.pkl', 'wb') as f:
    pickle.dump(f_lst, f)

