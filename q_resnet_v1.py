"""
@time:2019/3/5 上午9:27
@author:bigmelon

se/sese_resnet_v1 implementation
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from q_resnet_utils import stack_blocks_dense_2d, stack_blocks_dense_1d, conv2d_same, conv1d_same, subsample_2d, \
    subsample_1d, Block
from q_attention_module import se_block, sese_block

slim = tf.contrib.slim


class NoOpScope(object):
    """
    No-op context manager - a special method for "with xxx :" => with = __enter_- + __exit__
    This Noop scope is for
    todo find more about build a context manager => https://www.cnblogs.com/flashBoxer/p/9664813.html
    """

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        # if return True => no exception report
        return False


@slim.add_arg_scope
def bottleneck_2d(inputs, depth, depth_bottleneck, stride, v, rate=1, outputs_collections=None,
                  scope=None, use_bounded_activations=False, attention_module=None):
    """
    todo bottleneck_2d sesenet is not in use!!!!
    Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth、depth_bottleneck、stride三个参数是前面blocks类中的args
    depth: 一个block中的某个unit中(第三个conv)输出的feature-map的个数
    depth_bottleneck:  一个block中的某个unit中(前面两个conv)输出的feature-map个数
    stride: 是short_cut路径对于para_inputs/pre_act(经过bn层的para_inputs)的subsample_2d的步长 -- (是否经过bn层主要看输入输出通道数是否一致)
            以及unit中conv-2的步长
    rate: An integer, rate for atrous convolution.
    outputs_collections: 是收集end_points的collection
    scope: 是这个unit的名称。

    todo what is quantized inference??? what is bounded activations???
    use_bounded_activations: Whether or not to use bounded activations. Bounded
        activations better lend themselves to quantized inference.
    attention_module: SE-blocks or SESE-blocks
    """
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        # shortcut fine-tune
        if depth == depth_in:
            shortcut = subsample_2d(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride, activation_fn=tf.nn.relu6 if use_bounded_activations else None,
                                   scope='shortcut')
        # convs
        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')
        # post-relu
        if use_bounded_activations:
            # Use clip_by_value to simulate bandpass activation.
            # todo why using bandpass act?????
            residual = tf.clip_by_value(residual, -6.0, 6.0)
            # add se
            if attention_module == 'se_block':
                residual = se_block(residual, name='se_block', ratio=2 if residual.get_shape()[-1] <= 8 else 8)
            # add sese
            if attention_module == 'sese_block':
                # todo ratio to be defined...
                residual = sese_block(input_feature=residual, name='sese_block', v=v, ratio=2 if residual.get_shape()[-1] <= 8 else 8)
            output = tf.nn.relu6(shortcut + residual)
        else:
            # add se
            if attention_module == 'se_block':
                residual = se_block(residual, name='se_block', ratio=2 if residual.get_shape()[-1] <= 8 else 8)
            # add sese
            if attention_module == 'sese_block':
                # todo ratio to be defined...
                residual = sese_block(input_feature=residual, name='sese_block', v=v, ratio=2 if residual.get_shape()[-1] <= 8 else 8)
            output = tf.nn.relu(shortcut + residual)

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


@slim.add_arg_scope
def standard_bottleneck_1d(inputs, depth, depth_bottleneck, stride, v, rate=1, outputs_collections=None,
                           scope=None, use_bounded_activations=False, attention_module=None):
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
    scope: 是这个unit的名称。

    todo what is quantized inference??? what is bounded activations???  => must make it clear before any runs
    use_bounded_activations: Whether or not to use bounded activations. Bounded
        activations better lend themselves to quantized inference.
    attention_module: SE-blocks or SESE-blocks
    """
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        # shortcut fine-tune
        if depth == depth_in:
            shortcut = subsample_1d(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride, activation_fn=tf.nn.relu6 if use_bounded_activations else None,
                                   scope='shortcut')
        # convs
        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv1d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')
        # post-relu
        if use_bounded_activations:
            # Use clip_by_value to simulate bandpass activation.
            # todo why using bandpass act?????
            residual = tf.clip_by_value(residual, -6.0, 6.0)
            # add se
            if attention_module == 'se_block':
                residual = se_block(residual, name='se_block', ratio=2 if residual.get_shape()[-1] <= 8 else 8)
            # add sese
            elif attention_module == 'sese_block':
                # todo ratio to be defined...
                residual = sese_block(input_feature=residual, name='sese_block', v=v, ratio=2 if residual.get_shape()[-1] <= 8 else 8)
            output = tf.nn.relu6(shortcut + residual)
        else:
            # add se
            if attention_module == 'se_block':
                residual = se_block(residual, name='se_block', ratio=2 if residual.get_shape()[-1] <= 8 else 8)
            # add sese
            elif attention_module == 'sese_block':
                # todo ratio to be defined...
                residual = sese_block(input_feature=residual, name='sese_block', v=v, ratio=2 if residual.get_shape()[-1] <= 8 else 8)
            output = tf.nn.relu(shortcut + residual)

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


@slim.add_arg_scope
def pre_bottleneck_1d(inputs, depth, depth_bottleneck, stride, v, rate=1, outputs_collections=None,
                      scope=None, use_bounded_activations=False, attention_module=None):
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
    scope: 是这个unit的名称。

    todo what is quantized inference??? what is bounded activations???  => must make it clear before any runs
    use_bounded_activations: Whether or not to use bounded activations. Bounded
        activations better lend themselves to quantized inference.
    attention_module: SE-blocks or SESE-blocks
    """
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        # shortcut fine-tune
        if depth == depth_in:
            shortcut = subsample_1d(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride, activation_fn=tf.nn.relu6 if use_bounded_activations else None,
                                   scope='shortcut')
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
        # convs
        residual = slim.conv2d(residual, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv1d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')
        # post-relu
        if use_bounded_activations:
            # Use clip_by_value to simulate bandpass activation.
            # todo why using bandpass act?????
            residual = tf.clip_by_value(residual, -6.0, 6.0)
            output = tf.nn.relu6(shortcut + residual)
        else:
            output = tf.nn.relu(shortcut + residual)

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


@slim.add_arg_scope
def identity_bottleneck_1d(inputs, depth, depth_bottleneck, stride, v, rate=1, outputs_collections=None,
                           scope=None, use_bounded_activations=False, attention_module=None):
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
    scope: 是这个unit的名称。

    todo what is quantized inference??? what is bounded activations???  => must make it clear before any runs
    use_bounded_activations: Whether or not to use bounded activations. Bounded
        activations better lend themselves to quantized inference.
    attention_module: SE-blocks or SESE-blocks
    """
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
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
                # todo ratio to be defined...
                shortcut = sese_block(input_feature=inputs, name='sese_block', v=v, ratio=2 if inputs.get_shape()[-1] <= 8 else 8)
            else:
                shortcut = inputs
            shortcut = slim.conv2d(shortcut, depth, [1, 1], stride=stride, activation_fn=tf.nn.relu6 if use_bounded_activations else None,
                                   scope='shortcut')
        # convs
        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv1d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')
        # post-relu
        if use_bounded_activations:
            # Use clip_by_value to simulate bandpass activation.
            # todo why using bandpass act?????
            residual = tf.clip_by_value(residual, -6.0, 6.0)
            output = tf.nn.relu6(shortcut + residual)
        else:
            output = tf.nn.relu(shortcut + residual)

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


def resnet_v1_2d(inputs, blocks, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                 include_root_block=True, spatial_squeeze=True, store_non_strided_activations=False,
                 reuse=None, scope=None):
    """
    Args:
    output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
        ratio of input to output spatial resolution.
    store_non_strided_activations: If True, we compute non-strided (undecimated)
        activations at the last unit of each block and store them in the
        `outputs_collections` before subsampling them. This gives us access to
        higher resolution intermediate activations which are useful in some
        dense prediction problems but increases 4x the computation and memory cost
        at the last unit of each block.
    reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    """
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck_2d, stack_blocks_dense_2d], outputs_collections=end_points_collection):
            # todo if not training => enter noop scope!
            with (slim.arg_scope([slim.batch_norm], is_training=is_training) if is_training is not None else NoOpScope()):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = stack_blocks_dense_2d(net, blocks, output_stride, store_non_strided_activations)
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        end_points[sc.name + '/spatial_squeeze'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points


def resnet_v1_1d(inputs, blocks, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                 include_root_block=True, spatial_squeeze=True, store_non_strided_activations=False,
                 reuse=None, scope=None, s=None):
    """
    Args:
    s = None (s is short for switch)!!!
    output_stride: If None, then the output will be computed at the nominal
        network stride. If output_stride is not None, it specifies the requested
        ratio of input to output spatial resolution.
    store_non_strided_activations: If True, we compute non-strided (undecimated)
        activations at the last unit of each block and store them in the
        `outputs_collections` before subsampling them. This gives us access to
        higher resolution intermediate activations which are useful in some
        dense prediction problems but increases 4x the computation and memory cost
        at the last unit of each block.
    reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    """
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, standard_bottleneck_1d if s == 0 else pre_bottleneck_1d if s == 1
                            else identity_bottleneck_1d, stack_blocks_dense_1d], outputs_collections=end_points_collection):
            # todo if not training => enter noop scope!
            with (slim.arg_scope([slim.batch_norm], is_training=is_training) if is_training is not None else NoOpScope()):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    net = conv1d_same(net, num_outputs=8, kernel_size=4, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, kernel_size=[1, 3], stride=[1, 2], scope='pool1')
                # todo stack 函数根据blocks中的args参数dict进行参数解析
                net = stack_blocks_dense_1d(net, blocks, output_stride, store_non_strided_activations)
                # convert to dict 'end_points'
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                if global_pool:  # gap
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    if spatial_squeeze:
                        # todo ... check this before run ->
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        end_points[sc.name + '/spatial_squeeze'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points


def resnet_v1_block_2d(scope, base_depth, num_units, stride, attention_module):
    """
    implementation for resnet_v2_2d
    Args:(depth, depth_bottleneck, stride) | for general cases => depth=4*depth_bottleneck
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: implemented as a stride in the last unit, other units have stride=1.
    """
    return Block(scope, bottleneck_2d, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1,
        'attention_module': attention_module
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride,
        'attention_module': attention_module
    }])


def resnet_v1_block_1d(scope, base_depth, num_units, stride, v, attention_module, switch):
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


def resnet_v1_50(inputs, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                 spatial_squeeze=True, store_non_strided_activations=False, reuse=None,
                 scope='resnet_v1_50', attention_module=None):
    """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block_2d('block1', base_depth=64, num_units=3, stride=2, attention_module=attention_module),
        resnet_v1_block_2d('block2', base_depth=128, num_units=4, stride=2, attention_module=attention_module),
        resnet_v1_block_2d('block3', base_depth=256, num_units=6, stride=2, attention_module=attention_module),
        resnet_v1_block_2d('block4', base_depth=512, num_units=3, stride=1, attention_module=attention_module),
    ]
    return resnet_v1_2d(inputs, blocks, num_classes, is_training, global_pool=global_pool, output_stride=output_stride,
                        include_root_block=True, spatial_squeeze=spatial_squeeze,
                        store_non_strided_activations=store_non_strided_activations, reuse=reuse, scope=scope)


def resnet_v1_50_1d_dual_channel(inputs, v, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                                 spatial_squeeze=True, store_non_strided_activations=False, reuse=None,
                                 scope='resnet_v1_50_1d_dual_channel', attention_module=None, switch=None):
    """resnet_v1_50_1d for dual channel"""
    blocks = [
        resnet_v1_block_1d('block1', base_depth=8, num_units=3, stride=2, v=v, attention_module=attention_module, switch=switch),
        resnet_v1_block_1d('block2', base_depth=16, num_units=4, stride=2, v=v, attention_module=attention_module, switch=switch),
        resnet_v1_block_1d('block3', base_depth=32, num_units=6, stride=2, v=v, attention_module=attention_module, switch=switch),
        resnet_v1_block_1d('block4', base_depth=64, num_units=3, stride=1, v=v, attention_module=attention_module, switch=switch),
    ]
    return resnet_v1_1d(inputs, blocks, num_classes, is_training, global_pool=global_pool, output_stride=output_stride,
                        include_root_block=True, spatial_squeeze=spatial_squeeze,
                        store_non_strided_activations=store_non_strided_activations, reuse=reuse, scope=scope, s=switch)


def resnet_v1_101(inputs, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                  spatial_squeeze=True, store_non_strided_activations=False, reuse=None, scope='resnet_v1_101',
                  attention_module=None):
    """ResNet-101 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block_2d('block1', base_depth=64, num_units=3, stride=2, attention_module=attention_module),
        resnet_v1_block_2d('block2', base_depth=128, num_units=4, stride=2, attention_module=attention_module),
        resnet_v1_block_2d('block3', base_depth=256, num_units=23, stride=2, attention_module=attention_module),
        resnet_v1_block_2d('block4', base_depth=512, num_units=3, stride=1, attention_module=attention_module),
    ]
    return resnet_v1_2d(inputs, blocks, num_classes, is_training, global_pool=global_pool, output_stride=output_stride,
                        include_root_block=True, spatial_squeeze=spatial_squeeze,
                        store_non_strided_activations=store_non_strided_activations, reuse=reuse, scope=scope)


def resnet_v1_152(inputs, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                  store_non_strided_activations=False, spatial_squeeze=True, reuse=None, scope='resnet_v1_152',
                  attention_module=None):
    """ResNet-152 model of [1]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block_2d('block1', base_depth=64, num_units=3, stride=2, attention_module=attention_module),
        resnet_v1_block_2d('block2', base_depth=128, num_units=8, stride=2, attention_module=attention_module),
        resnet_v1_block_2d('block3', base_depth=256, num_units=36, stride=2, attention_module=attention_module),
        resnet_v1_block_2d('block4', base_depth=512, num_units=3, stride=1, attention_module=attention_module),
    ]
    return resnet_v1_2d(inputs, blocks, num_classes, is_training, global_pool=global_pool, output_stride=output_stride,
                        include_root_block=True, spatial_squeeze=spatial_squeeze,
                        store_non_strided_activations=store_non_strided_activations, reuse=reuse, scope=scope)


def resnet_v1_200(inputs, num_classes=None, is_training=True, global_pool=True, output_stride=None,
                  store_non_strided_activations=False, spatial_squeeze=True, reuse=None, scope='resnet_v1_200',
                  attention_module=None):
    """ResNet-200 model of [2]. See resnet_v1() for arg and return description."""
    blocks = [
        resnet_v1_block_2d('block1', base_depth=64, num_units=3, stride=2, attention_module=attention_module),
        resnet_v1_block_2d('block2', base_depth=128, num_units=24, stride=2, attention_module=attention_module),
        resnet_v1_block_2d('block3', base_depth=256, num_units=36, stride=2, attention_module=attention_module),
        resnet_v1_block_2d('block4', base_depth=512, num_units=3, stride=1, attention_module=attention_module),
    ]
    return resnet_v1_2d(inputs, blocks, num_classes, is_training, global_pool=global_pool, output_stride=output_stride,
                        include_root_block=True, spatial_squeeze=spatial_squeeze,
                        store_non_strided_activations=store_non_strided_activations, reuse=reuse, scope=scope)
