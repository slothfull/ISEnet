#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@time:2019/5/11 下午9:29
@author:bigmelon
mp2 is adam  && mp is momentum
"""

import tensorflow as tf
import numpy as np
import argparse
import os

from tqdm import tqdm

from q_utils import str2bool
# feed_and_derive_confusion_matrix, Dataset, dense_to_one_hot
from q_trainingdata import TrainingData
from q_net_utils_11_to_26 import build_from_metagraph, build_optimizer_from_metagraph, build_model_and_optimizer, initialize_uninitialized_variables
from q_summaries_classes_14 import PrecisionSummary, LossSummary, NetSummary, ExcitationSummary

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
exp = 'exp2'

num_train_samples = 46074  # todo to be defined
# path_ease = 'ch01/i17/res0/normal'
path_ease = 'i14d'
epoch = 200
batch = 8
continue_training = '0'
num_iteration = float(num_train_samples // batch + 1)  # decay step=1 epoch =5760
a = batch

# 10e=55260 20e=110520 40e=221040  80e=442080  160e=884160  320e=1768320


def main():
    # Parse the command line input
    # more about parser see: https://www.cnblogs.com/zknublx/p/6106343.html
    parser = argparse.ArgumentParser(description='Train SESE-Net')

    parser.add_argument('--name', default='/share/donghao/demo10/'+exp+'/trained_models/'+path_ease+'/sesenet', help='project name')

    parser.add_argument('--v1', type=int, default=14, help='chosen form 6 10 14')

    parser.add_argument('--attention-module', type=str, default='sese_block', help='input se_block or sese_block or others')
    # todo
    parser.add_argument('--softmax-flag', type=int, default=0, help='1 for paper 0 for v1')

    parser.add_argument('--switch', type=int, default=0, help='0=standard | 1=pre | 2=identity')

    parser.add_argument('--train-or-not', type=str2bool, default='1', help='train flag')

    parser.add_argument('--validate-or-not', type=str2bool, default='0', help='validate flag')

    parser.add_argument('--test-or-not', type=str2bool, default='1', help='test flag')
    # todo
    parser.add_argument('--data-dir', default='/share/donghao/data2/'+exp+'/6p_dataset', help='dataset info directory')

    parser.add_argument('--epochs', type=int, default=epoch, help='number of training epochs')  # todo fine-tune

    parser.add_argument('--batch-size', type=int, default=batch, help='batch size')  # todo fixed set it as 2^n

    parser.add_argument('--tensorboard-dir', default='/share/donghao/demo10/'+exp+'/logs/'+path_ease+'/sesenet_tb', help='name of the tensorboard data directory')

    parser.add_argument('--pb-model-save-path', default='/share/donghao/demo10/'+exp+'/trained_models/'+path_ease+'/sesenet_pb', help='pb model dir')

    parser.add_argument('--tag-string', default='sesenet_pb', help='tag string for model')

    parser.add_argument('--checkpoint-interval', type=int, default=1, help='checkpoint interval')

    parser.add_argument('--max-to-keep', type=int, default=2, help='num of checkpoint files max to keep')

    parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay for bn')
    # todo
    parser.add_argument('--lr-decay-method-switch', type=int, default=0, help='0=piecewise|others=exponential')

    parser.add_argument('--lr-values', type=str, default='0.001;0.0005;0.0001;0.00005;0.00001;0.000005;0.000001', help='learning rate values')  # todo piecewise

    parser.add_argument('--lr-boundaries', type=str, default='17280;57600;115200;230400;460800;921600', help='3:10:20:40:80:160 b=8')

    parser.add_argument('--lr-value', type=float, default=0.001, help='learning rate for exp decay')

    parser.add_argument('--decay-steps', type=float, default=num_iteration, help='decay_steps=1 epoch')

    parser.add_argument('--decay-rate', type=float, default=0.97, help='decay rate: for 100 epoch -> lr=0.0001')

    parser.add_argument('--moving-average-decay', type=float, default=0.9997, help='moving avg decay')  # todo 降低decay 企图增加泛化能力

    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for the optimizer')  # todo 0.9

    parser.add_argument('--adam', type=float, default=0, help='adam optimizer')

    parser.add_argument('--adagrad', type=float, default=0, help='adagrad optimizer')

    parser.add_argument('--rmsprop', type=float, default=0, help='rmsprop optimizer')

    parser.add_argument('--num-samples-for-test-summary', type=int, default=1000, help='range in 0-10044')

    parser.add_argument('--confusion-matrix-normalization', type=str2bool, default='1', help='confusion matrix norm flag')

    parser.add_argument('--class-names', type=list, default=[np.str_('N'), np.str_('S'), np.str_('V'), np.str_('F'), np.str_('Q')], help='...')

    parser.add_argument('--continue-training', type=str2bool, default=continue_training, help='continue training flag')

    args = parser.parse_args()

    print('[i] Project name:                 ', args.name)
    print('[i] Model categories(18/34):      ', args.v1)
    print('[i] Attention module categories): ', args.attention_module)
    print('[i] Softmax flag(for sese):       ', args.softmax_flag)
    print('[i] Attention switch:             ', args.switch)
    print('[i] Train or not:                 ', args.train_or_not)
    print('[i] Validate or not:              ', args.validate_or_not)
    print('[i] Test or not:                  ', args.test_or_not)
    print('[i] Data directory:               ', args.data_dir)
    print('[i] epochs:                       ', args.epochs)
    print('[i] Batch size:                   ', args.batch_size)
    print('[i] Tensorboard directory:        ', args.tensorboard_dir)
    print('[i] Pb model save path:           ', args.pb_model_save_path)
    print('[i] Tag string:                   ', args.tag_string)
    print('[i] Checkpoint interval:          ', args.checkpoint_interval)
    print('[i] Checkpoint max2keep:          ', args.max_to_keep)
    print('[i] Weight decay(bn):             ', args.weight_decay)
    print('[i] Learning rate decay switch    ', args.lr_decay_method_switch)
    print('[i] Learning rate values:         ', args.lr_values)
    print('[i] Learning rate boundaries:     ', args.lr_boundaries)
    print('[i] Learning rate value(exp):     ', args.lr_value)
    print('[i] Decay steps:                  ', args.decay_steps)
    print('[i] Decay rate:                   ', args.decay_rate)
    print('[i] Moving average decay:         ', args.moving_average_decay)
    print('[i] Momentum:                     ', args.momentum)
    print('[i] Adam:                         ', args.adam)
    print('[i] Adagrad:                      ', args.adagrad)
    print('[i] Rmsprop:                      ', args.rmsprop)
    print('[i] Num of samples for test       ', args.num_samples_for_test_summary)
    print('[i] Confusion matrix norm:        ', args.confusion_matrix_normalization)
    print('[i] Class names:                  ', args.class_names)
    print('[i] Continue training:            ', args.continue_training)

    # Find an existing checkpoint & continue training...
    start_epoch = 0
    if args.continue_training:
        state = tf.train.get_checkpoint_state(checkpoint_dir=args.name, latest_filename=None)
        if state is None:
            print('[!] No network state found in ' + args.name)
            return 1
        # check ckpt path
        ckpt_paths = state.all_model_checkpoint_paths
        if not ckpt_paths:
            print('[!] No network state found in ' + args.name)
            return 1

        # find the latest checkpoint file to go on train-process...
        last_epoch = None
        checkpoint_file = None
        for ckpt in ckpt_paths:
            # os.path.basename return the final component of a path
            # for e66.ckpt.data-00000-of-00001 we got ckpt_num=66
            ckpt_num = os.path.basename(ckpt).split('.')[0][1:]
            try:
                ckpt_num = int(ckpt_num)
            except ValueError:
                continue
            if last_epoch is None or last_epoch < ckpt_num:
                last_epoch = ckpt_num
                checkpoint_file = ckpt

        if checkpoint_file is None:
            print('[!] No checkpoints found, cannot continue!')
            return 1

        metagraph_file = checkpoint_file + '.meta'

        if not os.path.exists(metagraph_file):
            print('[!] Cannot find metagraph', metagraph_file)
            return 1
        start_epoch = last_epoch
    else:
        metagraph_file = None
        checkpoint_file = None
        try:
            print('[i] Creating directory             {}...'.format(args.name))
            os.makedirs(args.name)
        except IOError as e:
            print('[!]', str(e))
            return 1

    print('[i] Configuring the training data...')
    try:
        td = TrainingData(args.data_dir, args.batch_size)
        print('[i] training samples:             ', td.num_train)
        print('[i] classes:                      ', td.num_classes)
        print('[i] ecg_chip size:                ', f'({td.sample_width}, {td.sample_length})')
    except (AttributeError, RuntimeError) as e:
        print('[!] Unable to load training data:', str(e))
        return 1

    print('[i] Training ...')
    with tf.Session(config=config) as sess:
        if start_epoch != 0:
            print('[i] Building model from metagraph...')
            xs, ys = build_from_metagraph(sess, metagraph_file, checkpoint_file)
            process_flag, loss, accuracy, y_predict, train_op = build_optimizer_from_metagraph(sess)
        else:
            print('[i] Building model for dual channel...')
            xs, ys, process_flag, y_predict, loss, accuracy, train_op = \
                build_model_and_optimizer(args.softmax_flag, td.num_classes, td.sample_width,
                                          td.sample_length, td.sample_channel,
                                          args.moving_average_decay,
                                          args.lr_decay_method_switch, args.lr_values,
                                          args.lr_boundaries, args.lr_value, args.decay_steps,
                                          args.decay_rate, args.adam, args.momentum,
                                          args.adagrad, args.rmsprop, args.v1, attention_module=args.attention_module,
                                          switch=args.switch, weight_decay=args.weight_decay)
        # todo a typical wrong implement of initializer for a "reload" model
        # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # sess.run(init_op)
        # todo right way to do this: initialize w&b as the last update value...
        initialize_uninitialized_variables(sess)
        # create various helpers
        # If `None`, defaults to the list of all saveable objects.
        # can use the code below to save the trainable vars and bn 'mean' & 'variance'
        # var_list = tf.trainable_variables()
        # g_list = tf.global_variables()
        # bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        # bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        # var_list += bn_moving_vars
        # saver = tf.train.Saver(var_list=var_list, max_to_keep=3)
        summary_writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
        saver = tf.train.Saver(max_to_keep=args.max_to_keep)
        # saver1 = tf.train.Saver(max_to_keep=args.max_to_keep * 2)
        saver2 = tf.train.Saver(max_to_keep=args.max_to_keep * 2)

        # build summaries
        # precision & loss summary
        train_precision = PrecisionSummary(sess, summary_writer, 'train', td.num_train_iter, args.continue_training) \
            if args.train_or_not else None
        test_precision = PrecisionSummary(sess, summary_writer, 'test', td.num_test_iter, args.continue_training) \
            if args.test_or_not else None
        train_loss = LossSummary(sess, summary_writer, 'train', td.num_train_iter, args.continue_training) \
            if args.train_or_not else None
        test_loss = LossSummary(sess, summary_writer, 'test', td.num_test_iter, args.continue_training) \
            if args.test_or_not else None
        # excitation summary
        train_excitation = ExcitationSummary(sess, summary_writer, 'train', args.attention_module, args.continue_training, path_ease) \
            if args.train_or_not else None
        test_excitation = ExcitationSummary(sess, summary_writer, 'test', args.attention_module, args.continue_training, path_ease) \
            if args.test_or_not else None
        # net summary
        net_summary = NetSummary(sess)
        net_summary.build_summaries(args.continue_training)
        if start_epoch == 0:
            net_summaries = sess.run(net_summary.summary_op)  # run
            summary_writer.add_summary(net_summaries, 0)  # add
            summary_writer.flush()  # flush
        # set saved_model builder
        if start_epoch != 0:
            builder = tf.saved_model.builder.SavedModelBuilder(args.pb_model_save_path + f'_{start_epoch}')
        else:
            builder = tf.saved_model.builder.SavedModelBuilder(args.pb_model_save_path)

        print('[i] Training...')
        # max_acc = 0
        pre_num = 0
        # if train the first time, start_epoch=0 else start_epoch=last_epoch(from checkpoint file...)
        for e in range(start_epoch, args.epochs):
            # Train ->
            train_cache = []
            train_flag_lst = [0, 1, 2, 3]
            if args.train_or_not:
                td.train_iter(process='train', num_epoch=args.epochs)
                description = '[i] Train {:>2}/{}'.format(e + 1, args.epochs)  # epoch_No/total_epoch_No -> e+1/200
                for _ in tqdm(iterable=td.train_tqdm_iter, total=td.num_train_iter, desc=description, unit='batches'):
                    x, y = sess.run(td.train_sample)  # array(?,1,512,2) array(?,5)
                    train_dict = {xs: x, ys: y, process_flag: True}
                    _, acc, los = sess.run([train_op, accuracy, loss], feed_dict=train_dict)
                    # sample
                    if train_flag_lst:  # if not find all yet
                        dense_y = np.argmax(y, axis=1)  # (?,)
                        for index, ele in enumerate(dense_y):
                            if ele in train_flag_lst:
                                train_cache.append((x[index], ele))  # example(1,512,2) label()
                                train_flag_lst.remove(ele)
                                break
                    # add for precision & ce loss
                    sub_tp = 0; sub_fn = 0; sub_fp = 0
                    train_precision.add(acc, sub_tp, sub_fn, sub_fp)
                    train_loss.add(values=los)
            # Test ->
            test_cache = []
            test_flag_lst = [0, 1, 2, 3]
            if args.test_or_not:
                td.test_iter(process='test', num_epoch=args.epochs)
                description = '[i] Test {:>2}/{}'.format(e + 1, args.epochs)
                for _ in tqdm(iterable=td.test_tqdm_iter, total=td.num_test_iter, desc=description, unit='batches'):
                    x, y = sess.run(td.test_sample)  # array(?,1,512,2) array(?,5)
                    test_dict = {xs: x, ys: y, process_flag: False}
                    acc, los, predict = sess.run([accuracy, loss, y_predict], feed_dict=test_dict)
                    # sample for excitation
                    if test_flag_lst:  # if not find all yet
                        dense_y = np.argmax(y, axis=1)  # (?,)
                        for index, ele in enumerate(dense_y):
                            if ele in test_flag_lst:
                                test_cache.append((x[index], ele))  # example(1,512,2) label()
                                test_flag_lst.remove(ele)
                                break
                    # derive test prediction for sveb
                    sub_tp = 0; sub_fn = 0; sub_fp = 0
                    for index, _ in enumerate(np.argmax(y, axis=1)):
                        if _ == 1:  # count for recall
                            if predict[index] == 1:
                                sub_tp = sub_tp + 1
                            else:
                                sub_fn = sub_fn + 1
                        else:  # if not sveb -> count for ppr
                            if predict[index] == 1:
                                sub_fp = sub_fp + 1
                            pass
                    # add for precision & ce loss
                    test_precision.add(acc, sub_tp, sub_fn, sub_fp)
                    test_loss.add(values=los)
            # check
            if not args.train_or_not and not args.validate_or_not and not args.test_or_not:
                exit('[!] No procedures implemented!')

            # todo check
            # sess.graph.finalize()
            # todo push & flush tb
            # self, train, valid, test, train_lst, valid_lst, test_lst, xs, epoch
            if args.train_or_not:
                train_excitation.push(train_cache, xs, process_flag, e)
                train_precision.push(e)
                train_loss.push(e)
            if args.test_or_not:
                test_excitation.push(test_cache, xs, process_flag, e)
                test_precision.push(e)
                test_loss.push(e)
            # run-add net w&b
            net_summaries = sess.run(net_summary.summary_op)  # run again to derive 'next-step' summary
            summary_writer.add_summary(net_summaries, e + 1)  # add again

            # flush all(summaries of loss/precision/ecg_chip & summaries of ecgnet) protocol buf into disk
            summary_writer.flush()

            # save checkpoint
            if (e + 1) % args.checkpoint_interval == 0:
                checkpoint = '{}/e{}.ckpt'.format(args.name, e + 1)
                saver.save(sess, checkpoint)
                print('[i] Checkpoint saved:', checkpoint)
            # for best acc
            sveb_ppr = test_precision.ppr
            sveb_recall = test_precision.recall
            if sveb_recall >= 0.6:
                num = 1 * sveb_ppr + 1 * sveb_recall
                if (e + 1) % args.checkpoint_interval == 0 and num >= pre_num:
                    checkpoint3 = '{}/highest/e{}.ckpt'.format(args.name, e + 1)
                    saver2.save(sess, checkpoint3)
                    # refresh
                    pre_num = num if num > pre_num else pre_num

        # close writer
        summary_writer.close()

        # after all epochs goes out, save pb model...
        print('[i] Saving pb model(after training steps goes up)...')
        builder.add_meta_graph_and_variables(sess, [args.tag_string])
        builder.save()
        print('[i] programme finished!')    

if __name__ == '__main__':
    main()
