#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@time:1419/4/15 上午9:44
@author:bigmelon
"""

# unused
se_excitations = ['resnet_v2_14_1d_dual_channel/block1/unit_1/bottleneck_v2/se_block/bottleneck_fc/Relu:0',
                  'resnet_v2_14_1d_dual_channel/block1/unit_1/bottleneck_v2/se_block/recover_fc/Sigmoid:0',
                  'resnet_v2_14_1d_dual_channel/block1/unit_1/bottleneck_v2/se_block/mul_1:0',
                  'resnet_v2_14_1d_dual_channel/block2/unit_1/bottleneck_v2/se_block/bottleneck_fc/Relu:0',
                  'resnet_v2_14_1d_dual_channel/block2/unit_1/bottleneck_v2/se_block/recover_fc/Sigmoid:0',
                  'resnet_v2_14_1d_dual_channel/block2/unit_1/bottleneck_v2/se_block/mul_1:0',
                  'resnet_v2_14_1d_dual_channel/block3/unit_1/bottleneck_v2/se_block/bottleneck_fc/Relu:0',
                  'resnet_v2_14_1d_dual_channel/block3/unit_1/bottleneck_v2/se_block/recover_fc/Sigmoid:0',
                  'resnet_v2_14_1d_dual_channel/block3/unit_1/bottleneck_v2/se_block/mul_1:0',
                  'resnet_v2_14_1d_dual_channel/block4/unit_1/bottleneck_v2/se_block/bottleneck_fc/Relu:0',
                  'resnet_v2_14_1d_dual_channel/block4/unit_1/bottleneck_v2/se_block/recover_fc/Sigmoid:0',
                  'resnet_v2_14_1d_dual_channel/block4/unit_1/bottleneck_v2/se_block/mul_1:0']

# se using
se_chosen_excitations = ['resnet_v2_14_1d_dual_channel/block1/unit_1/bottleneck_v2/se_block/recover_fc/Sigmoid:0',
                         'resnet_v2_14_1d_dual_channel/block2/unit_1/bottleneck_v2/se_block/recover_fc/Sigmoid:0',
                         'resnet_v2_14_1d_dual_channel/block3/unit_1/bottleneck_v2/se_block/recover_fc/Sigmoid:0',
                         'resnet_v2_14_1d_dual_channel/block4/unit_1/bottleneck_v2/se_block/recover_fc/Sigmoid:0']

# unused
sese_excitations = ['resnet_v2_14_1d_dual_channel/block1/unit_1/bottleneck_v2/sese_block/bottleneck_fc/Relu:0',
                    'resnet_v2_14_1d_dual_channel/block1/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0',
                    'resnet_v2_14_1d_dual_channel/block1/unit_1/bottleneck_v2/sese_block/mul_1:0',
                    'resnet_v2_14_1d_dual_channel/block2/unit_1/bottleneck_v2/sese_block/bottleneck_fc/Relu:0',
                    'resnet_v2_14_1d_dual_channel/block2/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0',
                    'resnet_v2_14_1d_dual_channel/block2/unit_1/bottleneck_v2/sese_block/mul_1:0',
                    'resnet_v2_14_1d_dual_channel/block3/unit_1/bottleneck_v2/sese_block/bottleneck_fc/Relu:0',
                    'resnet_v2_14_1d_dual_channel/block3/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0',
                    'resnet_v2_14_1d_dual_channel/block3/unit_1/bottleneck_v2/sese_block/mul_1:0',
                    'resnet_v2_14_1d_dual_channel/block4/unit_1/bottleneck_v2/sese_block/bottleneck_fc/Relu:0',
                    'resnet_v2_14_1d_dual_channel/block4/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0',
                    'resnet_v2_14_1d_dual_channel/block4/unit_1/bottleneck_v2/sese_block/mul_1:0']

# sese using
sese_chosen_excitations = ['resnet_v2_14_1d_dual_channel/block1/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0',
                           'resnet_v2_14_1d_dual_channel/block2/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0',
                           'resnet_v2_14_1d_dual_channel/block3/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0',
                           'resnet_v2_14_1d_dual_channel/block4/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0']

# using senet & sesenet 14 share
simplified_chosen_excitations = ['block1/unit1', 'block2/unit1', 'block3/unit1', 'block4/unit1']

# Tensor("resnet_v2_14_1d_dual_channel/block1/unit_1/bottleneck_v2/sese_block/bottleneck_fc/Relu:0", shape=(1, 1, 1, 2), dtype=float32)
# Tensor("resnet_v2_14_1d_dual_channel/block1/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0", shape=(1, 1, 1, 16), dtype=float32)
# Tensor("resnet_v2_14_1d_dual_channel/block1/unit_1/bottleneck_v2/sese_block/mul_1:0", shape=(?, 1, 64, 16), dtype=float32)
# Tensor("resnet_v2_14_1d_dual_channel/block2/unit_1/bottleneck_v2/sese_block/bottleneck_fc/Relu:0", shape=(1, 1, 1, 4), dtype=float32)
# Tensor("resnet_v2_14_1d_dual_channel/block2/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0", shape=(1, 1, 1, 32), dtype=float32)
# Tensor("resnet_v2_14_1d_dual_channel/block2/unit_1/bottleneck_v2/sese_block/mul_1:0", shape=(?, 1, 32, 32), dtype=float32)
# Tensor("resnet_v2_14_1d_dual_channel/block3/unit_1/bottleneck_v2/sese_block/bottleneck_fc/Relu:0", shape=(1, 1, 1, 8), dtype=float32)
# Tensor("resnet_v2_14_1d_dual_channel/block3/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0", shape=(1, 1, 1, 64), dtype=float32)
# Tensor("resnet_v2_14_1d_dual_channel/block3/unit_1/bottleneck_v2/sese_block/mul_1:0", shape=(?, 1, 16, 64), dtype=float32)
# Tensor("resnet_v2_14_1d_dual_channel/block4/unit_1/bottleneck_v2/sese_block/bottleneck_fc/Relu:0", shape=(1, 1, 1, 16), dtype=float32)
# Tensor("resnet_v2_14_1d_dual_channel/block4/unit_1/bottleneck_v2/sese_block/recover_fc/Sigmoid:0", shape=(1, 1, 1, 128), dtype=float32)
# Tensor("resnet_v2_14_1d_dual_channel/block4/unit_1/bottleneck_v2/sese_block/mul_1:0", shape=(?, 1, 8, 128), dtype=float32)
