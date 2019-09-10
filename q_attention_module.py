#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@time:2019/4/13 下午11:02
@author:bigmelon
"""

import tensorflow as tf
slim = tf.contrib.slim


def derive_sese_output(t, c):
    """
    this implementation is the implementation of softmax to F_l[i][c][j] in the paper
    """
    # t(batch_size,height,width,channel)=(0,1,2,3)
    t = tf.transpose(t, perm=[3, 0, 1, 2])  # (c,b,h,w)=(4,2,1,3)
    t = tf.reshape(t, shape=(c, -1))  # (c, b*h*w)=(4,6)
    # log_softmax = logits - log(reduce_sum(exp(logits), axis))
    tt = tf.nn.log_softmax(t, axis=1)  # (c, b*h*w)=(4,6)
    ttt = tf.nn.softmax(t, axis=1)  # (c, b*h*w)=(4,6)
    ttt = tt*ttt  # (c, b*h*w)=(4,6)
    ttt = -tf.reduce_sum(ttt, axis=1)  # (c,)=(4,)
    tttt = tf.reshape(ttt, shape=(c, 1, 1, 1))  # (c,b,h,w)=(4,1,1,1)
    ttttt = tf.transpose(tttt, perm=[1, 2, 3, 0])  # (b,h,w,c)=(1,1,1,4)
    return ttttt


def derive_sese_output_v1(t):
    """ v1 compute softmax for each channel | compute entropy for each channel

    only need to compute softmax in axis=2 for a 1d_cnn
    todo for 2d feature map in 1 batch: should first use << reshape + softmax + reshape >>
    tmp = tf.reshape(arr1, shape=(-1, 4, 2))  #
    tmp = tf.nn.softmax(tmp, axis=1)  # Calculate for both axes
    rs = tf.reshape(tmp, shape=(-1, 2, 2, 2))"""
    st = tf.nn.softmax(t, axis=2)  # (b,h,w,c) axis=2 so only w changes for softmax
    lst = tf.nn.log_softmax(t, axis=2)
    t = tf.multiply(st, lst)
    t = -tf.reduce_sum(t, axis=2, keepdims=True)
    return t


def sese_block(input_feature, name, v, ratio=8):
    """
    Contains the implementation of Softmax-Entropy-Squeeze-and-Excitation block. => standard sese-block
    todo the way for ratio settings seems important for this algorithm!!!!
    ratio should not be larger than input channels????
    """
    kernel_initializer1 = tf.contrib.layers.variance_scaling_initializer()
    kernel_initializer2 = tf.contrib.layers.xavier_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    with tf.variable_scope(name):
        # height = input_feature.get_shape()[1]
        # width = input_feature.get_shape()[2]
        channel = input_feature.get_shape()[-1]
        # sese
        entropy_fmap = derive_sese_output(input_feature, channel) if v == 1 else derive_sese_output_v1(input_feature)
        # two fc
        # outputs = activation(inputs.kernel + bias)
        # units = channel // ratio # todo must define the shape by tf.reshape else it raise value error "shape undefined"
        excitation = tf.layers.dense(inputs=tf.reshape(entropy_fmap, [-1, 1, 1, channel]),
                                     units=channel // ratio,
                                     activation=tf.nn.relu,
                                     kernel_initializer=kernel_initializer1,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        # print(excitation)
        # print(excitation.name)
        # unit = channel
        excitation = tf.layers.dense(inputs=excitation,  # (2,1,1,4)
                                     units=channel,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=kernel_initializer2,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')
        # print(excitation)
        # print(excitation.name)
        # todo step-4 Fscale op -> using broadcast of python
        scale = input_feature * excitation  # (2,2,3,4)
        # print(scale)
        # print(scale.name)
        # convert to float32
        try:
            scale = tf.to_float(x=scale, name='ToFloat')
        except TypeError as t:
            exit(f'[!] {str(t)}')
    return scale


def se_block(input_feature, name, ratio=8):
    """
    Contains the implementation of Squeeze-and-Excitation block
    As described in https://arxiv.org/abs/1709.01507.
    """
    # todo find out why kernel&bias use different initializer?
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]  # get last dimension size = channel
        # todo step-1 Global average pooling by reduce_mean is more efficient!
        squeeze = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)  # input_feature=(?,h,w,c) squeeze=(?,1,1,c)
        # todo step-2 two stacked fc layers
        # outputs=activation(inputs.kernel + bias)
        # units=channel // ratio => excitation = 1*1*channel//ratio
        excitation = tf.layers.dense(inputs=squeeze,
                                     units=channel // ratio,
                                     activation=tf.nn.relu,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='bottleneck_fc')
        # unit=channel => excitation = 1*1*channel
        # print(excitation)
        # print(excitation.name)
        excitation = tf.layers.dense(inputs=excitation,
                                     units=channel,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name='recover_fc')
        # todo after tf.nn.sigmoid -> the value of excitation should be range in 0-1
        # print(excitation)
        # print(excitation.name)
        # step-3
        # (b,1,w,c)*(b,1,1,c) 自动进行了 broadcast 相当于(b,1,w,c)*(b,1,w,c) -> 对每个map进行了加权
        scale = input_feature * excitation
        # print(scale)
        # print(scale.name)
        # convert to float32
        try:
            scale = tf.to_float(x=scale, name='ToFloat')
        except TypeError as t:
            exit(f'[!] {str(t)}')
    return scale


# def fully_connected(inputs,
#                     num_outputs,
#                     activation_fn=nn.relu,
#                     normalizer_fn=None,
#                     normalizer_params=None,
#                     weights_initializer=initializers.xavier_initializer(),
#                     weights_regularizer=None,
#                     biases_initializer=init_ops.zeros_initializer(),
#                     biases_regularizer=None,
#                     reuse=None,
#                     variables_collections=None,
#                     outputs_collections=None,
#                     trainable=True,
#                     scope=None):
