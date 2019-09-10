#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@time:2019/3/7 上午9:05
@author:bigmelon
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import itertools
import argparse
# todo use traceback to find out the str name of the caller
import traceback

from sklearn.metrics import confusion_matrix


def compute_lr(lr_values, lr_boundaries):
    """
    PIECEWISE_CONSTANT_LEARNING_RATE -> for train.py
    :param lr_values:
    :param lr_boundaries: piecewise-global-step-boundaries
    """
    # use a piecewise constant learning rate
    with tf.variable_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False, name='global_step')  # init global_step=0 | add it in optimizer.minimize
        lr = tf.train.piecewise_constant(global_step, lr_boundaries, lr_values)
    return lr, global_step


def compute_lr_v2(lr_values, para_decay_steps, para_decay_rate):
    """
    PIECEWISE_CONSTANT_LEARNING_RATE -> for train.py
    :param para_decay_rate:
    :param para_decay_steps:
    :param lr_values:
    decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    """
    # use a piecewise constant learning rate
    with tf.variable_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False, name='global_step')  # init global_step=0 | add it in optimizer.minimize
        lr = tf.train.exponential_decay(lr_values, global_step, para_decay_steps, para_decay_rate, staircase=True)
    return lr, global_step


def str2bool(v):
    """
    Convert a command line input string to a boolean | to expand the boolean_flag_holder -> for xxxx_train,py
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True ecg_example_label')
    plt.xlabel('Predicted ecg_example_label')
    plt.tight_layout()


def feed_and_derive_confusion_matrix(para_y_test, para_y_pred, para_normalization, para_class_names):
    """
    feed the true ecg_example_label and predicted ecg_example_label, can plot the confusion matrix...
    :param para_y_test: np.array with shape(xxx,)
    :param para_y_pred: np.array with shape(xxx,)
    :param para_normalization: true/false
    :param para_class_names: np.array with shape(xxx,) elements_type=np.str_
    :return: none
    """
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(para_y_test, para_y_pred)
    # set the precision of the floats
    np.set_printoptions(precision=4)
    if para_normalization:
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=para_class_names, normalize=True, title='Normalized confusion matrix')
        # todo change plot name according to the caller
        if 'train_senet_v2' in traceback.extract_stack()[-2][0] or 'train_senet' in traceback.extract_stack()[-2][0]:
            plt.savefig('Normalized_confusion_matrix_senet_ch01.jpg')
        elif 'train_sesenet_v2' in traceback.extract_stack()[-2][0] or 'train_sesenet' in traceback.extract_stack()[-2][0]:
            plt.savefig('Normalized_confusion_matrix_sesenet_ch01.jpg')
        else:
            exit('[!] Wrong in channel_0_and_1_utils - line 130!')
        plt.show()
    else:
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=para_class_names, normalize=False, title='non-Normalized Confusion matrix')
        # todo change plot name according to the caller
        if 'senet_v2' in traceback.extract_stack()[-2][0] or 'train_senet' in traceback.extract_stack()[-2][0]:
            plt.savefig('Confusion_matrix_senet_ch01.jpg')
        elif 'sesenet_v2' in traceback.extract_stack()[-2][0] or 'train_sesenet' in traceback.extract_stack()[-2][0]:
            plt.savefig('Confusion_matrix_sesenet_ch01.jpg')
        else:
            exit('[!] Wrong in channel_0_and_1_utils - line 130!')
        plt.show()


# convert the dense label to one_hot label
def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_of_labels = labels_dense.shape[0]
    # print(num_labels)
    index_offset = np.arange(num_of_labels) * num_classes
    # print(index_offset)
    labels_one_hot = np.zeros((num_of_labels, num_classes))
    # print(labels_one_hot)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    # print('label_one_hot', labels_one_hot, 'type of labels_one_hot', type(labels_one_hot))
    # 将经过one_hot形式的label_list由float64转化成int类型,下面加引号的程序不能实现此功能
    return labels_one_hot


class Dataset:
    def __init__(self, data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        pass

    @property
    def data(self):
        return self._data

    def next_batch(self, para_batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0 and shuffle:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle index
            self._data = self.data[idx]  # get list of `num` random samples

        # go to the next batch
        if start + para_batch_size > self._num_examples and shuffle:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of `num` random samples

            start = 0
            # avoid the case where the #sample != integer times of batch_size
            self._index_in_epoch = para_batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += para_batch_size
            end = self._index_in_epoch
            return self._data[start:end]
