#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@time:2019/4/13 下午11:17
@author:bigmelon
"""

import tensorflow as tf
import sys

from tqdm import tqdm

from q_resnet_utils import resnet_arg_scope
from q_resnet_v2 import resnet_v2_50_1d_dual_channel
from q_resnet_v1 import resnet_v1_50_1d_dual_channel
from q_utils import compute_lr, compute_lr_v2

slim = tf.contrib.slim


def initialize_uninitialized_variables(sess):
    """
    Only initialize the weights that have not yet been initialized by other
    means, such as importing a metagraph and a checkpoint. It's useful when
    extending an existing model.
    todo
    """
    with tf.name_scope('initialize_ops'):
        uninit_vars = []  # uninitialized variables list
        uninit_tensors = []  # uninitialized boolean_tensors list
        for var in tf.global_variables():
            uninit_vars.append(var)  # tf.get_variable() are all un-initialized...
            uninit_tensors.append(tf.is_variable_initialized(var))  # using tf.is_variable_initialized() to check vars init-state...
        uninit_bools = sess.run(uninit_tensors)  # derive the bool-value-list of bool-uninit-tensors-list
        # a=0->false [(1,797)->true] [(798,822)ecgnet_w&b optimizer->(823,993)resnet_w&b (993,1016)ecgnet_w&b->false]
        # a = uninit_bools.count(True)
        uninit = zip(uninit_bools, uninit_vars)
        uninit = [var for init, var in uninit if not init]  # extract uninitialized-tensor-var as list
        init_op = tf.variables_initializer(uninit, name='ops')
        sess.run(init_op)  # initialize un-inited vars(ecgnet-related)


def build_from_metagraph(session, metagraph_file, checkpoint_file):
    sess = session
    saver = tf.train.import_meta_graph(metagraph_file)  # import metagraph
    saver.restore(sess, checkpoint_file)  # restore previously saved variables
    xs = sess.graph.get_tensor_by_name('inputs/x_input:0')
    ys = sess.graph.get_tensor_by_name('inputs/y_input:0')
    return xs, ys


def build_optimizer_from_metagraph(session):
    sess = session
    # verified
    loss = sess.graph.get_tensor_by_name('losses/AddN:0')
    accuracy = sess.graph.get_tensor_by_name('compute_accuracy/Mean:0')
    y_predict = sess.graph.get_tensor_by_name('compute_accuracy/ArgMax:0')
    # l2_loss = sess.graph.get_tensor_by_name('l2_loss/Mul:0')
    train_op = sess.graph.get_operation_by_name('train_1')
    return loss, accuracy, y_predict, train_op


def build_model_and_optimizer(v, num_classes, sample_width, sample_length, sample_channel, moving_average_decay,
                              lr_decay_method_switch, lr_values, lr_boundaries, lr_value, decay_steps, decay_rate,
                              adam, momentum, adagrad, rmsprop, v1, attention_module, switch, weight_decay):

    with tf.name_scope('inputs'):
        # for 1d-resnet | placeholder is an op not var
        xs = tf.placeholder(tf.float32, [None, sample_width, sample_length, sample_channel], name='x_input')
        ys = tf.placeholder(tf.float32, [None, num_classes], name='y_input')  # (?,5)
    # todo using resnet_arg_scope to set default params for conv2d & bn & maxpool
    with slim.arg_scope(resnet_arg_scope(weight_decay=weight_decay, batch_norm_decay=0.997, batch_norm_epsilon=1e-5,
                                         batch_norm_scale=True)):
        if v1:
            y, end_points = resnet_v1_50_1d_dual_channel(xs, v=v, num_classes=num_classes, spatial_squeeze=False,
                                                         attention_module=attention_module, switch=switch)
            y = tf.squeeze(y, axis=[1, 2])  # (?,5)
            tf.cast(y, tf.float32)
        else:
            # y=shape=(?,1,1,5) | end_points['predictions']=end_points['logits']=shape(?,1,1,5) | no using squeeze
            y, end_points = resnet_v2_50_1d_dual_channel(xs, v=v, num_classes=num_classes, spatial_squeeze=False,
                                                         attention_module=attention_module, switch=switch)
            y = tf.squeeze(y, axis=[1, 2])  # (?,5)
            tf.cast(y, tf.float32)

    with tf.name_scope('compute_accuracy'):
        # tf.argmax().output_type=(int64)
        y_predict = tf.argmax(y, 1)  # y_predict=shape=(?,) | tf.argmax(ys,1)=shape(?,)
        correct_prediction = tf.equal(y_predict, tf.argmax(ys, 1))  # (?,)
        # accuracy per batch
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # ()
        # tf.summary.scalar(the parameter must be a tensor!!! but not a value)
        tf.summary.scalar('/results', accuracy)  # todo can be deleted

    with tf.name_scope('cross_entropy'):
        """return a tensor of the same shape as 'labels' and of the same type as 'logits'."""
        # logits=y.shape=(?,num_of_classes)     ys=ecg_example_label=(?,num_of_classes)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=ys, name='cross_entropy')  # (?,)
        # cross_entropy = tensor: shape(100,) dtype = float32
        # cross_entropy_sum = tensor: shape() dtype = float32 reduce the batch_size tensor into one number
        # cross_entropy_sum = tf.reduce_sum(cross_entropy, name='reduce_sum')  # () total ce per batch
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        # cross_entropy_mean = tf.divide(x=cross_entropy_sum, y=float(batch_size), name='avg_ce_per_batch')  # avg ce per batch
        # tf.summary.scalar('/cross_entropy', cross_entropy_mean)  # todo can be deleted

    with tf.name_scope('moving_average'):
        # define moving_avg_op moving_average_decay 是decay的初始值  同时decay和迭代轮数有关 公式如下
        # 所维护的影子变量和原来的变量的关系 ---> shadow_variable = decay * shadow_variable + (1 - decay) * variable
        # decay = min{decay, (1+num_updates)/(10+num_updates)}  ---> num_updates = global_step
        # moving_average_decay越小 shadow_variable 偏离原来的 variable越大
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
        # 给予所有trainable_variables维护一个滑动平均的"影子变量"
        # sess.run(variables_averages_op) ---> updates all shadow variables as described above.
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope('losses'):
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n([cross_entropy_mean] + regularization_losses)  # loss per sample

    with tf.name_scope('train'):
        """
         decayed_learning_rate = learning_rate * decay_rate^(global_step/decay_steps)
        """
        if lr_decay_method_switch == 0:
            lr_values = lr_values.split(';')  # list [0.001, 0.0001, 0.00001]
            try:
                lr_values = [float(x) for x in lr_values]  # convert to float
            except ValueError:
                print('[!] Learning rate values must be floats')
                sys.exit(1)

            lr_boundaries = lr_boundaries.split(';')  # learning_rate change boundaries = list [320000, 400000]
            try:
                lr_boundaries = [int(x) for x in lr_boundaries]  # convert to int
            except ValueError:
                print('[!] Learning rate boundaries must be ints')
                sys.exit(1)

            ret = compute_lr(lr_values, lr_boundaries)
            learning_rate, global_step = ret  # derive lr g_stp tensor
        else:
            ret = compute_lr_v2(lr_value, decay_steps, decay_rate)
            learning_rate, global_step = ret
        # minimize()函数分为两个部分 compute_gradients() apply_gradients()  global_step在apply_gradients()里面实现自动+1
        # 每次计算gradients的时候 global_step+1 所以global_steps=num_iterations
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        if adam:
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        elif momentum:
            train_step = tf.train.MomentumOptimizer(learning_rate, momentum=momentum).minimize(loss, global_step=global_step)
        elif adagrad:
            train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
        elif rmsprop:
            train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)
        else:
            train_step = None
            exit('[!] Wrong optimizer chosen!')
    with tf.control_dependencies([train_step, variables_averages_op]):
        # tf.control_dependencies([op_a,op_b]) 是一个上下文管理器 sess.run(train_op)表示在执行完op_a和op_b之前不会执行train_op
        # 即:当含有sess.run([train_op,...])的其他的tensor的数值 只有在执行了op_a和op_b之后才执行
        # 每更新一次global_step updates moving_average_variables once
        train_op = tf.no_op(name='train')
    return xs, ys, y_predict, loss, accuracy, train_op


# todo this function is built specially for channel_0_and_1_trainingdata.py
# todo 这个函数实现了连个graph之间(正在训练的graph(一般为冗余的graph) && 训练之初保存的初始graph(没有多余节点的graph))的value传递
# todo 相当于做了一个graph"清洗"的工作 但是由于tf.assign()的过程非常消耗时间(清洗trainingdata-100-epoch需要几个小时) 这个方式没有被采用
def construct_and_save_new_checkpoint(session, epoch, dst_vars, dst_checkpoint, prior_ckpt_path, prior_meta_graph_path):
    with tf.Graph().as_default() as graph:  # create a new graph & set as default...
        with tf.Session() as sess:
            # load prior meta graph(include vars) in current graph...
            saver = tf.train.import_meta_graph(prior_meta_graph_path)
            saver.restore(sess, prior_ckpt_path)
            # derive trainable variables in prior graph
            t_names = []
            for t in tf.trainable_variables():
                t_names.append(t.name)
            v_names = []
            # derive global variables in prior graph
            for v in tf.global_variables():
                try:
                    v_names.append(v.name)
                except AttributeError as e:
                    print(f'variable object has no attribute names{e}')
                    print(v)
            dst_saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
            i = 0
            description = '[i] Assign {:>2}/{}'.format(epoch, 1000)  # epoch_No/total_epoch_No -> e+1/200
            for var in tqdm(iterable=dst_vars, desc=description):
                if var.name in v_names:
                    i = i+1
                    # derive the tensor
                    tensor = graph.get_tensor_by_name(var.name)
                    # derive the value tensor
                    weight = sess.run(tensor)
                    # assign values to destination vars
                    session.run(var.assign(weight))
            dst_saver.save(sess, dst_checkpoint)
            print('[i] Checkpoint saved:', dst_checkpoint)
            exit('[i] Restart the programme! ')
