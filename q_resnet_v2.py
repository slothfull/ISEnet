#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@time:2019/4/13 下午11:06
@author:bigmelon
resnet-v2 implementation for paper
"""

import tensorflow as tf

from q_resnet_utils import subsample_1d, stack_blocks_dense_1d, conv1d_same, Block
from q_attention_module import se_block, sese_block

slim = tf.contrib.slim


@slim.add_arg_scope
def standard_bottleneck_1d(inputs, depth, depth_bottleneck, stride, v, rate=1,
                           outputs_collections=None, scope=None, attention_module=None):
    """
    Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth、depth_bottleneck、stride三个参数是前面blocks类中的args
    depth: 一个block中的某个unit中(第三个conv)输出的feature-map的个数
    depth_bottleneck:  一个block中的某个unit中(前面两个conv)输出的feature-map个数
    stride: 是short_cut路径对于para_inputs/pre_act(经过bn层的para_inputs)的subsample_2d的步长 -- (是否经过bn层主要看输入输出通道数是否一致)
            以及unit中conv-2的步长
    rate: An integer, rate for atrous convolution.
    outputs_collections: 是收集end_points的collection
    scope: 是这个unit的名称
    attention_module: SE-blocks or SESE-blocks
    """
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        # pre activate + bn + relu
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        # shortcut fine-tune
        if depth == depth_in:
            shortcut = subsample_1d(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None, scope='shortcut')
        # convs
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv1d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')
        # add se
        if attention_module == 'se_block':
            residual = se_block(residual, name='se_block', ratio=2 if residual.get_shape()[-1] <= 8 else 8)
        if attention_module == 'sese_block':
            residual = sese_block(input_feature=residual, name='sese_block', v=v, ratio=2 if residual.get_shape()[-1] <= 8 else 8)
        # junction
        output = shortcut + residual
        return slim.utils.collect_named_outputs(outputs_collections, sc.name,  output)


@slim.add_arg_scope
def pre_bottleneck_1d(inputs, depth, depth_bottleneck, stride, v, rate=1,
                      outputs_collections=None, scope=None, attention_module=None):
    """
    Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth、depth_bottleneck、stride三个参数是前面blocks类中的args
    depth: 一个block中的某个unit中(第三个conv)输出的feature-map的个数
    depth_bottleneck:  一个block中的某个unit中(前面两个conv)输出的feature-map个数
    stride: 是short_cut路径对于para_inputs/pre_act(经过bn层的para_inputs)的subsample_2d的步长 -- (是否经过bn层主要看输入输出通道数是否一致)
            以及unit中conv-2的步长
    rate: An integer, rate for atrous convolution.
    outputs_collections: 是收集end_points的collection
    scope: 是这个unit的名称
    attention_module: SE-blocks or SESE-blocks
    """
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        # add se
        if attention_module == 'se_block':
            residual = se_block(inputs, name='se_block', ratio=2 if inputs.get_shape()[-1] <= 8 else 8)
        # add sese
        elif attention_module == 'sese_block':
            # todo ratio to be defined...
            residual = sese_block(input_feature=inputs, name='sese_block', v=v, ratio=2 if inputs.get_shape()[-1] <= 8 else 8)
        # no other block implemented
        else:
            residual = inputs
        # pre activate
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        # shortcut fine-tune
        if depth == depth_in:
            shortcut = subsample_1d(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None, scope='shortcut')
        # convs
        residual = slim.conv2d(residual, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv1d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')
        # junction
        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name,  output)


@slim.add_arg_scope
def identity_bottleneck_1d(inputs, depth, depth_bottleneck, stride, v, rate=1,
                           outputs_collections=None, scope=None, attention_module=None):
    """
    Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth、depth_bottleneck、stride三个参数是前面blocks类中的args
    depth: 一个block中的某个unit中(第三个conv)输出的feature-map的个数
    depth_bottleneck:  一个block中的某个unit中(前面两个conv)输出的feature-map个数
    stride: 是short_cut路径对于para_inputs/pre_act(经过bn层的para_inputs)的subsample_2d的步长 -- (是否经过bn层主要看输入输出通道数是否一致)
            以及unit中conv-2的步长
    rate: An integer, rate for atrous convolution.
    outputs_collections: 是收集end_points的collection
    scope: 是这个unit的名称
    attention_module: SE-blocks or SESE-blocks
    """
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        # pre activate
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        # shortcut fine-tune
        if depth == depth_in:
            # pre se/sese in shortcut
            if attention_module == 'se_block':
                shortcut = se_block(inputs, name='se_block', ratio=2 if inputs.get_shape()[-1] <= 8 else 8)
            elif attention_module == 'sese_block':
                # todo ratio to be defined...
                shortcut = sese_block(input_feature=inputs, name='sese_block', v=v, ratio=2 if inputs.get_shape()[-1] <= 8 else 8)
            else:
                shortcut = inputs
            shortcut = subsample_1d(shortcut, stride, 'shortcut')
        else:
            # pre se/sese in shortcut
            if attention_module == 'se_block':
                shortcut = se_block(inputs, name='se_block', ratio=2 if inputs.get_shape()[-1] <= 8 else 8)
            elif attention_module == 'sese_block':
                # todo ratio to be defined ratio should be smaller than inputs.get_shape()[-1]
                shortcut = sese_block(input_feature=inputs, name='sese_block', v=v, ratio=2 if inputs.get_shape()[-1] <= 8 else 8)
            else:
                shortcut = inputs
            shortcut = slim.conv2d(shortcut, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None, scope='shortcut')
        # convs
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv1d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')
        # add se
        if attention_module == 'se_block':
            residual = se_block(residual, name='se_block', ratio=2 if residual.get_shape()[-1] <= 8 else 8)
        if attention_module == 'sese_block':
            # todo ratio tobe defined...
            residual = sese_block(input_feature=residual, name='sese_block', v=v, ratio=2 if residual.get_shape()[-1] <= 8 else 8)
        # junction
        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name,  output)


def resnet_v2_1d(inputs, blocks, num_classes=None, is_training=True, global_pool=True,
                 output_stride=None, include_root_block=True,  spatial_squeeze=True,
                 reuse=None, scope=None, s=None):
    """
    implementation for resnet_v2 1d | more detail see tf/slim/.../nets/resnet_v2_discarded.py
    """
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # todo slim.conv2d?????????
        with slim.arg_scope([slim.conv2d, standard_bottleneck_1d if s == 0 else pre_bottleneck_1d if s == 1
                            else identity_bottleneck_1d, stack_blocks_dense_1d],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):  # 单独为batch_norm设置train的参数状态
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    # We do not include batch normalization or activation functions in
                    # conv1 because the first ResNet unit will perform these. Cf.
                    # Appendix of [2].
                    with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                        net = conv1d_same(net, num_outputs=8, kernel_size=4, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [1, 3], stride=[1, 2], scope='pool1')
                net = stack_blocks_dense_1d(net, blocks, output_stride)
                # This is needed because the pre-activation variant does not have batch
                # normalization or activation functions in the residual unit output. See
                # Appendix of [2].
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)  # (x,1,xx,xxx) -> (x,xxx)
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    # todo make it clear before run
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        end_points[sc.name + '/spatial_squeeze'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points


def resnet_v2_block_1d(scope, base_depth, num_units, stride, v, attention_module, switch):
    """
    implementation for resnet_v2_1d
    Args:(depth, depth_bottleneck, stride) | for general cases => depth=4*depth_bottleneck
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: implemented as a stride in the last unit, other units have stride=1.
    """
    if switch != 0 and switch != 1 and switch != 2:
        exit('[!] Wrong args.switch input resnet_v1 - line395!')
    return Block(scope, standard_bottleneck_1d if switch == 0
                 else pre_bottleneck_1d if switch == 1
                 else identity_bottleneck_1d if switch == 2
                 else None, [{'depth': base_depth * 2,
                              'depth_bottleneck': base_depth,
                              'stride': 1,
                              'v': v,
                              'attention_module': attention_module
                              }] * (num_units - 1) + [{
                                    'depth': base_depth * 2,
                                    'depth_bottleneck': base_depth,
                                    'stride': stride,
                                    'v': v,
                                    'attention_module': attention_module
                              }])


def resnet_v2_block_1d_v1(scope, base_depth, v, attention_module, switch, num_units=None):
    """
    implementation for resnet_v2_1d
    Args:(depth, depth_bottleneck, stride) | for general cases => depth=4*depth_bottleneck
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: implemented as a stride in the last unit, other units have stride=1.
    """
    if switch != 0 and switch != 1 and switch != 2:
        exit('[!] Wrong args.switch input resnet_v1 - line395!')
    return Block(scope, standard_bottleneck_1d if switch == 0
                 else pre_bottleneck_1d if switch == 1
                 else identity_bottleneck_1d if switch == 2
                 else None, [{'depth': base_depth * 2,
                              'depth_bottleneck': base_depth,
                              'stride': 2,
                              'v': v,
                              'attention_module': attention_module
                              }] * (num_units - 1) + [{'depth': base_depth * 2,
                                                       'depth_bottleneck': base_depth,
                                                       'stride': 1,
                                                       'v': v,
                                                       'attention_module': attention_module}])


def resnet_v2_block_1d_v2(scope, base_depth, v, attention_module, switch, num_units=None):
    """
    implementation for resnet_v2_1d
    Args:(depth, depth_bottleneck, stride) | for general cases => depth=4*depth_bottleneck
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: implemented as a stride in the last unit, other units have stride=1.
    """
    if switch != 0 and switch != 1 and switch != 2:
        exit('[!] Wrong args.switch input resnet_v1 - line395!')
    return Block(scope, standard_bottleneck_1d if switch == 0
                 else pre_bottleneck_1d if switch == 1
                 else identity_bottleneck_1d if switch == 2
                 else None, [{'depth': base_depth * 2,
                              'depth_bottleneck': base_depth,
                              'stride': 1,
                              'v': v,
                              'attention_module': attention_module
                              }] * (num_units - 1) + [{'depth': base_depth * 4,
                                                       'depth_bottleneck': base_depth * 2,
                                                       'stride': 2,
                                                       'v': v,
                                                       'attention_module': attention_module}])


def resnet_v2_11_1d_dual_channel(inputs, v, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                                 spatial_squeeze=True, reuse=None, scope='resnet_v2_11_1d_dual_channel',
                                 attention_module=None, switch=None):
    """
    """
    blocks = [
        resnet_v2_block_1d_v1('block1', base_depth=8, num_units=1, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block2', base_depth=16, num_units=1, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block3', base_depth=32, num_units=1, v=v, attention_module=attention_module, switch=switch)
    ]
    return resnet_v2_1d(inputs, blocks, num_classes, is_training=is_training, global_pool=global_pool,
                        output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze,
                        reuse=reuse, scope=scope, s=switch)


def resnet_v2_14_1d_dual_channel(inputs, v, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                                 spatial_squeeze=True, reuse=None, scope='resnet_v2_14_1d_dual_channel',
                                 attention_module=None, switch=None):
    """
    """
    blocks = [
        resnet_v2_block_1d_v1('block1', base_depth=8, num_units=1, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block2', base_depth=16, num_units=1, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block3', base_depth=32, num_units=1, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block4', base_depth=64, num_units=1, v=v, attention_module=attention_module, switch=switch),
    ]
    return resnet_v2_1d(inputs, blocks, num_classes, is_training=is_training, global_pool=global_pool,
                        output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze,
                        reuse=reuse, scope=scope, s=switch)


def resnet_v2_17_1d_dual_channel(inputs, v, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                                 spatial_squeeze=True, reuse=None, scope='resnet_v2_17_1d_dual_channel',
                                 attention_module=None, switch=None):
    """
    """
    blocks = [
        resnet_v2_block_1d_v1('block1', base_depth=8, num_units=1, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block2', base_depth=16, num_units=1, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block3', base_depth=32, num_units=2, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block4', base_depth=64, num_units=1, v=v, attention_module=attention_module, switch=switch),
    ]
    return resnet_v2_1d(inputs, blocks, num_classes, is_training=is_training, global_pool=global_pool,
                        output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze,
                        reuse=reuse, scope=scope, s=switch)


def resnet_v2_20_1d_dual_channel(inputs, v, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                                 spatial_squeeze=True, reuse=None, scope='resnet_v2_20_1d_dual_channel',
                                 attention_module=None, switch=None):
    """
    """
    blocks = [
        resnet_v2_block_1d_v1('block1', base_depth=8, num_units=1, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block2', base_depth=16, num_units=2, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block3', base_depth=32, num_units=2, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block4', base_depth=64, num_units=1, v=v, attention_module=attention_module, switch=switch),
    ]
    return resnet_v2_1d(inputs, blocks, num_classes, is_training=is_training, global_pool=global_pool,
                        output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze,
                        reuse=reuse, scope=scope, s=switch)


# todo -> 23
def resnet_v2_23_1d_dual_channel(inputs, v, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                                 spatial_squeeze=True, reuse=None, scope='resnet_v2_23_1d_dual_channel',
                                 attention_module=None, switch=None):
    """
    """
    blocks = [
        resnet_v2_block_1d_v1('block1', base_depth=8, num_units=1, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block2', base_depth=16, num_units=2, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block3', base_depth=32, num_units=2, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block4', base_depth=64, num_units=2, v=v, attention_module=attention_module, switch=switch),
    ]
    return resnet_v2_1d(inputs, blocks, num_classes, is_training=is_training, global_pool=global_pool,
                        output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze,
                        reuse=reuse, scope=scope, s=switch)


# todo -> 26
def resnet_v2_26_1d_dual_channel(inputs, v, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                                 spatial_squeeze=True, reuse=None, scope='resnet_v2_26_1d_dual_channel',
                                 attention_module=None, switch=None):
    """
    """
    blocks = [
        resnet_v2_block_1d_v1('block1', base_depth=8, num_units=1, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block2', base_depth=16, num_units=2, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block3', base_depth=32, num_units=3, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block4', base_depth=64, num_units=2, v=v, attention_module=attention_module, switch=switch),
    ]
    return resnet_v2_1d(inputs, blocks, num_classes, is_training=is_training, global_pool=global_pool,
                        output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze,
                        reuse=reuse, scope=scope, s=switch)


# todo -> 34
def resnet_v2_34_1d_dual_channel(inputs, v, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                                 spatial_squeeze=True, reuse=None, scope='resnet_v2_34_1d_dual_channel',
                                 attention_module=None, switch=None):
    """resnet_v2_34_1d for dual channel"""
    blocks = [
        resnet_v2_block_1d_v1('block1', base_depth=8, num_units=3, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block2', base_depth=16, num_units=4, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block3', base_depth=32, num_units=6, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d_v1('block4', base_depth=64, num_units=3, v=v, attention_module=attention_module, switch=switch),
    ]
    return resnet_v2_1d(inputs, blocks, num_classes, is_training=is_training, global_pool=global_pool,
                        output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze,
                        reuse=reuse, scope=scope, s=switch)


# todo -> used
def resnet_v2_50_1d_dual_channel(inputs, v, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                                 spatial_squeeze=True, reuse=None, scope='resnet_v2_50_1d_dual_channel',
                                 attention_module=None, switch=None):
    """resnet_v2_50_1d for dual channel"""
    blocks = [
        resnet_v2_block_1d('block1', base_depth=8, num_units=3, stride=2, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d('block2', base_depth=16, num_units=4, stride=2, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d('block3', base_depth=32, num_units=6, stride=2, v=v, attention_module=attention_module, switch=switch),
        resnet_v2_block_1d('block4', base_depth=64, num_units=3, stride=1, v=v, attention_module=attention_module, switch=switch),
    ]
    return resnet_v2_1d(inputs, blocks, num_classes, is_training=is_training, global_pool=global_pool,
                        output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze,
                        reuse=reuse, scope=scope, s=switch)
