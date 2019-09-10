# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains building blocks for various versions of Residual Networks.

Residual networks (ResNets) were proposed in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition. arXiv:1512.03385, 2015

More variants were introduced in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks. arXiv: 1603.05027, 2016

We can obtain different ResNet variants by changing the network depth, width,
and form of residual unit. This module implements the infrastructure for
building them. Concrete ResNet units and full ResNet networks are implemented in
the accompanying q_resnet_v1.py and resnet_v2.py modules.

Compared to https://github.com/KaimingHe/deep-residual-networks, in the current
implementation we subsample the output activations in the last residual unit of
each block, instead of subsampling the input activations in the first residual
unit of each block. The two implementations give identical results but our
implementation is more memory efficient.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

slim = tf.contrib.slim


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """
    A named tuple describing a ResNet block.

    Its parts are:
        scope: The scope of the `Block`.
        unit_fn: The ResNet unit function which takes as input a `Tensor` and
               returns another `Tensor` with the output of the ResNet unit.
        args: A list of length equal to the number of units in the `Block`. The list
            contains one (depth, depth_bottleneck, stride) tuple for each unit in the
            block to serve as argument to unit_fn.
    """


def subsample_2d(inputs, factor, scope=None):
    """
    Subsamples the input along the spatial dimensions. implementation for 2d resnet
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def subsample_1d(inputs, factor, scope=None):
    """
    implementation for 1d resnet => stride=[1,factor]
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=[1, factor], scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    """
    implementation for 2d resnet -> rate=>atrous_conv
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate, padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, rate=rate, padding='VALID', scope=scope)


def conv1d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    """
    implementation for 1d resnet -> rate=>atrous_conv
    """
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, [1, kernel_size], stride=1, rate=rate, padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)  # ?
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [0, 0]])  # (?,h,w,c) 只在w维度补0
        return slim.conv2d(inputs, num_outputs, [1, kernel_size], stride=stride, rate=rate, padding='VALID',
                           scope=scope)


@slim.add_arg_scope
def stack_blocks_dense_2d(net, blocks, output_stride=None,
                          store_non_strided_activations=False,
                          outputs_collections=None):
    """
    for sparse prediction -> output_stride=None
    for dense prediction(segmentation etc.) -> output_stride=not None
    """
    # The current_stride variable keeps track of the effective stride of the
    # activations. This allows us to invoke atrous convolution whenever applying
    # the next residual unit would result in the activations having stride larger
    # than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            block_stride = 1
            for i, unit in enumerate(block.args):
                if store_non_strided_activations and i == len(block.args) - 1:
                    # Move stride from the block's last unit to the end of the block.
                    block_stride = unit.get('stride', 1)
                    unit = dict(unit, stride=1)

                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)

                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
                        if output_stride is not None and current_stride > output_stride:
                            raise ValueError('The target output_stride cannot be reached.')

            # Collect activations at the block's end before performing subsampling.
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

            # Subsampling of the block's output activations.
            if output_stride is not None and current_stride == output_stride:
                rate *= block_stride
            else:
                # todo use subsample_2d for 2d resnet
                net = subsample_2d(net, block_stride)
                current_stride *= block_stride
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net


@slim.add_arg_scope
def stack_blocks_dense_1d(net, blocks, output_stride=None, store_non_strided_activations=False,
                          outputs_collections=None):
    """
    for sparse prediction -> output_stride=None
    for dense prediction(segmentation etc.) -> output_stride=not None
    """
    # The current_stride variable keeps track of the effective stride of the
    # activations. This allows us to invoke atrous convolution whenever applying
    # the next residual unit would result in the activations having stride larger
    # than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            block_stride = 1
            for i, unit in enumerate(block.args):
                if store_non_strided_activations and i == len(block.args) - 1:
                    # Move stride from the block's last unit to the end of the block.
                    block_stride = unit.get('stride', 1)  # dict.get 返回key对应的value 如果key不存在 则返回default=1
                    unit = dict(unit, stride=1)

                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)

                    else:
                        # todo 将unit其他参数字典传入block.unit_fn <=> bottleneck_1d
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
                        if output_stride is not None and current_stride > output_stride:
                            raise ValueError('The target output_stride cannot be reached.')

            # Collect activations at the block's end before performing subsampling.
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

            # Subsampling of the block's output activations.
            if output_stride is not None and current_stride == output_stride:
                rate *= block_stride
            else:
                # todo use subsample_1d for 1d resnet
                net = subsample_1d(net, block_stride)
                current_stride *= block_stride
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net


# todo set default params for BN
def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True):
    """Defines the default ResNet arg scope.

    TODO(gpapan): The batch-normalization related default values above are
        appropriate for use in conjunction with the reference ResNet models
        released at https://github.com/KaimingHe/deep-residual-networks. When
        training ResNets from scratch, they might need to be tuned.

    Args:
        weight_decay: The weight decay to use for regularizing the model.
        batch_norm_decay: The moving average decay when estimating layer activation
            statistics in batch normalization.
        batch_norm_epsilon: Small constant to prevent division by zero when
            normalizing activations by their variance in batch normalization.
        batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
            activations in the batch normalization layer.
        activation_fn: The activation function which is used in ResNet.
        use_batch_norm: Whether or not to use batch normalization.
    Returns:
        An `arg_scope` to use for the resnet models.
    """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': None,  # Use fused batch norm if possible.
    }
    # todo set params for slim.conv2d ->
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),  # todo
                        activation_fn=activation_fn,
                        normalizer_fn=slim.batch_norm if use_batch_norm else None,
                        normalizer_params=batch_norm_params):
        # todo set params for slim.batch_norm ->
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            # The following implies padding='SAME' for pool1, which makes feature
            # alignment easier for dense prediction tasks. This is also used in
            # https://github.com/facebook/fb.resnet.torch. However the accompanying
            # code of 'Deep Residual Learning for Image Recognition' uses
            # padding='VALID' for pool1. You can switch to that choice by setting
            # slim.arg_scope([slim.max_pool2d], padding='VALID').
            # todo set params for slime.maxpool ->
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc
