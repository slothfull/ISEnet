#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@time:2019/5/10 上午10:09
@author:bigmelon
用于逐个batch送入test的数据
"""

import tensorflow as tf
import numpy as np
import pickle
import math
from namedtuples import SAMPLE

# from q_utils import dense_to_one_hot


# data_dir = '/share/donghao/data2/exp2/np_dataset'
class TestingData:
    def __init__(self, data_dir, batch_size):
        # xxx_source_train_val_test_sample ->
        try:
            with open(data_dir + '/TEST_SAMPLE_LIST.pkl', 'rb') as f:
                test_sample_list = pickle.load(f)
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(str(e))
        self.data_dir = data_dir
        # preset
        self.num_classes = 5
        self.sample_width = 1  # default=1
        self.sample_length = 512  # default=512
        self.sample_channel = 2
        self.batch_size = batch_size
        # list
        self.test_sample_list = test_sample_list
        # num samples
        self.num_test = len(test_sample_list)
        # num iterations
        self.num_test_iter = int(math.ceil(len(test_sample_list) / self.batch_size))
        # self.num_test_iter = 22
        # iter
        self.train_iterator = None
        self.test_iterator = None
        self.valid_iterator = None
        self.train_tqdm_iter = None
        self.test_tqdm_iter = None
        self.valid_tqdm_iter = None
        # sample
        self.train_sample = None
        self.test_sample = None
        self.valid_sample = None
        # hook
        self.train_iter = self.__batch_generator()
        self.test_iter = self.__batch_generator()

        # todo preset-unused
        # self.preset = ...
        # todo compensatory_info
        # self.compensatory= ...

    def __batch_generator(self):
        # npy example reader
        def generate_data(para_path):
            raw_data = np.load(para_path)
            return raw_data  # array(1,512,2)

        def _test_generator():
            for index, ele in enumerate(self.test_sample_list):
                # para_path = para_path + f'train/{_}.bmp'
                # tmp_path = para_path + f'test/{_}.npy'
                tmp_path = ele[0]
                tmp_label = ele[1]
                # yield generate_data(tmp_path), np.squeeze(dense_to_one_hot(para_label[_], self.num_classes))
                yield generate_data(tmp_path), tmp_label

        def set_iterator(process, num_epoch):  # set train/validate params
            """
            The body of `generator` will not be serialized in a `GraphDef`, and you should not use this method if you
            need to serialize your model and restore it in a different environment.
            """
            # load sample to tensor
            if process == 'test':
                dataset = tf.data.Dataset.from_generator(_test_generator, output_types=(tf.float32, tf.float32),
                                                         output_shapes=(tf.TensorShape([1, 512, 2]), tf.TensorShape([5])))
            else:
                dataset = None
                exit('[!] Wrong process flag - trainingdata_v2!')

            # dataset = dataset.shuffle(buffer_size=10000, seed=None, reshuffle_each_iteration=False)
            # todo for ease of attention_module to get fixed batch_size
            dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=False)  # generate_batch
            dataset = dataset.repeat(count=num_epoch)  # set num epoch

            if process == 'test':
                self.test_iterator = dataset.make_one_shot_iterator()  # return a iterator for num_epoch data
                self.test_tqdm_iter = np.arange(self.num_test_iter)  # num_batches per epoch
                self.test_sample = self.test_iterator.get_next()
                self.valid_iterator = None
                self.valid_tqdm_iter = None
                self.train_iterator = None
                self.train_tqdm_iter = None
            else:
                exit(f'wrong process flag: {process}!')
        return set_iterator
