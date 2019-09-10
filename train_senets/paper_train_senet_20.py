#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@time:2019/4/6 下午10:26
@author:bigmelon
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
from q_summaries_classes_20 import PrecisionSummary, LossSummary, ExcitationSummary

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
exp = 'exp3'
# set test_models data path (ch0 & ch1)
test_data_path0 = '/share/donghao/demo5/'+exp+'/channel0/augmented_sample/test_example.npy'
test_label_path0 = '/share/donghao/demo5/'+exp+'/channel0/augmented_sample/test_example_label.npy'
test_data_path1 = '/share/donghao/demo5/'+exp+'/channel1/augmented_sample/test_example.npy'
test_label_path1 = '/share/donghao/demo5/'+exp+'/channel1/augmented_sample/test_example_label.npy'

num_train_samples = ...  # todo to be defined
path_ease = 't2/art'
batch = 1000
epoch = 8
continue_training = '0'
num_iteration = float(num_train_samples // batch + 1)  # decay step=1 epoch


def main():
    # Parse the command line input
    # more about parser see: https://www.cnblogs.com/zknublx/p/6106343.html
    parser = argparse.ArgumentParser(description='Train SE-Net')

    parser.add_argument('--name', default='/share/donghao/demo6/'+exp+'/trained_models/'+path_ease+'/senet', help='project name')
    # todo do not change!
    parser.add_argument('--v1', type=str2bool, default=20, help='v1=18: res-18 | v1=34: res-34')
    # todo do not change!
    parser.add_argument('--attention-module', type=str, default='se_block', help='input se_block or sese_block or others')
    # todo do not change! no matters
    parser.add_argument('--softmax-flag', type=str2bool, default='0', help='only for sese_block | 0=single channel 1=all channel')
    # todo do not change!
    parser.add_argument('--switch', type=int, default=0, help='0=standard | 1=pre | 2=identity')

    parser.add_argument('--train-or-not', type=str2bool, default='1', help='train flag')

    parser.add_argument('--validate-or-not', type=str2bool, default='0', help='validate flag')

    parser.add_argument('--test_models-or-not', type=str2bool, default='1', help='test_models flag')

    parser.add_argument('--data-dir', default='/share/donghao/demo5/'+exp+'/np_dataset', help='dataset info directory')
    # todo
    parser.add_argument('--epochs', type=int, default=epoch, help='number of training epochs')  # todo fine-tune
    # todo
    parser.add_argument('--batch-size', type=int, default=batch, help='batch size')  # todo fixed set it as 2^n
    # todo
    parser.add_argument('--tensorboard-dir', default='/share/donghao/demo6/'+exp+'/logs/'+path_ease+'/senet_tb', help='name of the tensorboard data directory')
    # todo
    parser.add_argument('--pb-model-save-path', default='/share/donghao/demo6/'+exp+'/trained_models/'+path_ease+'/senet_pb', help='pb model dir')
    # todo for t0-exp-senet tag_string='channel01_1d_resnet_pb_model'
    parser.add_argument('--tag-string', default='senet_pb', help='tag string for model')

    parser.add_argument('--checkpoint-interval', type=int, default=1, help='checkpoint interval')

    parser.add_argument('--max-to-keep', type=int, default=2, help='num of checkpoint files max to keep')

    parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay for bn')

    parser.add_argument('--lr-decay-method-switch', type=int, default=1, help='0=piecewise|others=exponential')

    parser.add_argument('--lr-values', type=str, default='0.001;0.0005;0.0001;0.00001', help='learning rate values')  # todo piecewise

    parser.add_argument('--lr-boundaries', type=str, default='353481;706962;2120886', help='learning rate change boundaries (in batches)')

    parser.add_argument('--lr-value', type=float, default=0.0001, help='learning rate for exp decay')  # todo exp decay

    parser.add_argument('--decay-steps', type=float, default=num_iteration, help='decay_steps=1 epoch')

    parser.add_argument('--decay-rate', type=float, default=0.99, help='decay rate: for 100 epoch -> lr=0.0001')

    parser.add_argument('--moving-average-decay', type=float, default=0.9999, help='moving avg decay')

    parser.add_argument('--momentum', type=float, default=0, help='momentum for the optimizer')  # todo 0.9

    parser.add_argument('--adam', type=float, default=1, help='adam optimizer')  # todo 控制变量进行和单一channel进行对比实验

    parser.add_argument('--adagrad', type=float, default=0, help='adagrad optimizer')

    parser.add_argument('--rmsprop', type=float, default=0, help='rmsprop optimizer')

    parser.add_argument('--num-samples-for-test_models-summary', type=int, default=1000, help='range in 0-10044')

    parser.add_argument('--confusion-matrix-normalization', type=str2bool, default='1', help='confusion matrix norm flag')

    parser.add_argument('--class-names', type=list, default=[np.str_('N'), np.str_('S'), np.str_('V'), np.str_('F'), np.str_('Q')], help='...')
    # todo
    parser.add_argument('--continue-training', type=str2bool, default=continue_training, help='continue training flag')

    args = parser.parse_args()

    print('[i] Project name:                 ', args.name)
    print('[i] Model categories(1=v1|0=v2):  ', args.v1)
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
    print('[i] Num of samples for test_models       ', args.num_samples_for_test_summary)
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
        print('[i] validation samples:           ', td.num_valid)
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
            loss, accuracy, y_predict, train_op = build_optimizer_from_metagraph(sess)
        else:
            print('[i] Building model for dual channel...')
            xs, ys, y_predict, loss, accuracy, train_op = \
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
        summary_writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)
        saver = tf.train.Saver(max_to_keep=args.max_to_keep)
        saver1 = tf.train.Saver(max_to_keep=args.max_to_keep)

        # build summaries
        train_precision = PrecisionSummary(sess, summary_writer, 'train', td.num_train_iter, args.continue_training) \
            if args.train_or_not else None
        valid_precision = PrecisionSummary(sess, summary_writer, 'valid', td.num_valid_iter, args.continue_training) \
            if args.validate_or_not else None
        test_precision = PrecisionSummary(sess, summary_writer, 'test_models', td.num_test_iter, args.continue_training) \
            if args.test_or_not else None
        train_loss = LossSummary(sess, summary_writer, 'train', td.num_train_iter, args.continue_training) \
            if args.train_or_not else None
        valid_loss = LossSummary(sess, summary_writer, 'valid', td.num_valid_iter, args.continue_training) \
            if args.validate_or_not else None
        test_loss = LossSummary(sess, summary_writer, 'test_models', td.num_test_iter, args.continue_training) \
            if args.test_or_not else None
        # self, session, writer, process_flag, restore=False
        train_excitation = ExcitationSummary(sess, summary_writer, 'train', args.attention_module, args.continue_training, path_ease) \
            if args.train_or_not else None
        valid_excitation = ExcitationSummary(sess, summary_writer, 'valid', args.attention_module, args.continue_training, path_ease) \
            if args.validate_or_not else None
        test_excitation = ExcitationSummary(sess, summary_writer, 'test_models', args.attention_module, args.continue_training, path_ease) \
            if args.test_or_not else None
        # set saved_model builder
        if start_epoch != 0:
            builder = tf.saved_model.builder.SavedModelBuilder(args.pb_model_save_path + f'_{start_epoch}')
        else:
            builder = tf.saved_model.builder.SavedModelBuilder(args.pb_model_save_path)

        print('[i] Training...')
        max_acc = 0
        # if train the first time, start_epoch=0 else start_epoch=last_epoch(from checkpoint file...)
        for e in range(start_epoch, args.epochs):
            # Train ->
            train_cache = []
            train_flag_lst = [0, 1, 2, 3, 4]
            if args.train_or_not:
                td.train_iter(process='train', num_epoch=args.epochs)
                description = '[i] Train {:>2}/{}'.format(e + 1, args.epochs)  # epoch_No/total_epoch_No -> e+1/200
                for _ in tqdm(iterable=td.train_tqdm_iter, total=td.num_train_iter, desc=description, unit='batches'):
                    x, y = sess.run(td.train_sample)  # array(?,1,512,2) array(?,5)
                    train_dict = {xs: x, ys: y}
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
                    train_precision.add(acc=acc)
                    train_loss.add(values=los)
            # Validate ->
            validate_cache = []
            validate_flag_lst = [0, 1, 2, 3, 4]
            if args.validate_or_not:
                td.valid_iter(process='validate', num_epoch=args.epochs)
                description = '[i] Valid {:>2}/{}'.format(e + 1, args.epochs)
                for _ in tqdm(iterable=td.valid_tqdm_iter, total=td.num_valid_iter, desc=description, unit='batches'):
                    x, y = sess.run(td.valid_sample)  # array(?,1,512,2) array(?,)
                    validate_dict = {xs: x, ys: y}
                    acc, los = sess.run([accuracy, loss], feed_dict=validate_dict)
                    # sample
                    if validate_flag_lst:  # if not find all yet
                        dense_y = np.argmax(y, axis=1)  # (?,)
                        for index, ele in enumerate(dense_y):
                            if ele in train_flag_lst:
                                validate_cache.append((x[index], ele))  # example(1,512,2) label()
                                validate_flag_lst.remove(ele)
                                break
                    # add for precision & ce loss
                    valid_precision.add(acc=acc)
                    valid_loss.add(values=los)
            # Test ->
            test_cache = []
            test_flag_lst = [0, 1, 2, 3, 4]
            if args.test_or_not:
                td.test_iter(process='test_models', num_epoch=args.epochs)
                description = '[i] Test {:>2}/{}'.format(e + 1, args.epochs)
                for _ in tqdm(iterable=td.test_tqdm_iter, total=td.num_test_iter, desc=description, unit='batches'):
                    x, y = sess.run(td.test_sample)  # array(?,1,512,2) array(?,5)
                    test_dict = {xs: x, ys: y}
                    acc, los = sess.run([accuracy, loss], feed_dict=test_dict)
                    # sample
                    if test_flag_lst:  # if not find all yet
                        dense_y = np.argmax(y, axis=1)  # (?,)
                        for index, ele in enumerate(dense_y):
                            if ele in test_flag_lst:
                                test_cache.append((x[index], ele))  # example(1,512,2) label()
                                test_flag_lst.remove(ele)
                                break
                    # add for precision & ce loss
                    test_precision.add(acc=acc)
                    test_loss.add(values=los)
            # check
            if not args.train_or_not and not args.validate_or_not and not args.test_or_not:
                exit('[!] No procedures implemented!')

            # todo check
            # sess.graph.finalize()
            # todo push & flush tb
            # self, train, valid, test_models, train_lst, valid_lst, test_lst, xs, epoch
            if args.train_or_not:
                train_excitation.push(train_cache, xs, e)
                train_precision.push(e)
                train_loss.push(e)
            if args.validate_or_not:
                valid_excitation.push(validate_cache, xs, e)
                valid_precision.push(e)
                valid_loss.push(e)
            if args.test_or_not:
                test_excitation.push(test_cache, xs, e)
                test_precision.push(e)
                test_loss.push(e)

            # flush all(summaries of loss/precision/ecg_chip & summaries of ecgnet) protocol buf into disk
            summary_writer.flush()

            # save checkpoint
            if (e + 1) % args.checkpoint_interval == 0:
                checkpoint = '{}/e{}.ckpt'.format(args.name, e + 1)
                saver.save(sess, checkpoint)
                print('[i] Checkpoint saved:', checkpoint)
            avg_acc = test_precision.precision_cache
            # todo 这个脚本的功能是找到准确率最高的模型 ^-^
            if (e + 1) % args.checkpoint_interval == 0 and avg_acc >= max_acc:
                checkpoint2 = '{}/highest/e{}.ckpt'.format(args.name, e + 1)
                saver1.save(sess, checkpoint2)
            # refresh max_acc
            max_acc = avg_acc if avg_acc > max_acc else max_acc

        # close writer
        summary_writer.close()

        # after all epochs goes out, save pb model...
        print('[i] Saving pb model(after training steps goes up)...')
        builder.add_meta_graph_and_variables(sess, [args.tag_string])
        builder.save()
        print('[i] programme finished!')
        # # 读取一定数量的test数据集 运行训练好的模型 并根据在test上的表现绘制confusion matrix
        # print('[i] Testing and drawing...')
        # # test_models will take up 210M memories
        # total_test_example_array_channel0 = np.load(test_data_path0)  # (10044,512)
        # total_test_label_array_channel0 = np.load(test_label_path0)  # (10044,1)
        # total_test_example_array_channel1 = np.load(test_data_path1)  # (10044,512)
        # # combine both channel  (10044,1,512,2)
        # print('[i] Combining 2 channel test_models example...')
        # total_test_example_array = np.zeros(shape=(1, 1, 512, 2))
        # total_test_label_array = total_test_label_array_channel0
        # for _ in range(len(total_test_label_array_channel0)):  # 0-10044
        #     # 将两个channel的ecg_chip进行堆叠
        #     tmp0 = np.zeros(shape=(2 * td.sample_length,))
        #     for __ in range(len(tmp0)):
        #         if __ % 2 == 0:
        #             tmp0[__] = total_test_example_array_channel0[_][__ // 2]  # channel-1
        #         else:
        #             tmp0[__] = total_test_example_array_channel1[_][(__ - 1) // 2]  # channel-2
        #     tmp1 = tmp0.reshape((td.sample_length, 2))  # (512,2)
        #     tmp2 = np.expand_dims(tmp1, axis=0)  # (1,512,2)
        #     tmp3 = np.expand_dims(tmp2, axis=0)  # (1,1,512,2)
        #     total_test_example_array = np.concatenate((total_test_example_array, tmp3), axis=0)  # (?,1,512,2)
        # total_test_example_array = total_test_example_array[1:, ...]  # (10044,1,512,2)
        # y_test_range = np.arange(0, 10044)
        # y_test = Dataset(y_test_range)
        # test_index_batch_summary = y_test.next_batch(para_batch_size=args.num_samples_for_test_summary)
        # example_batch_for_test_summary = total_test_example_array[test_index_batch_summary]  # shape=(1000,1,512,2)
        # label_batch_for_test_summary = total_test_label_array[test_index_batch_summary]  # shape=(1000, 1)
        # one_hot_label_batch_for_test_summary = dense_to_one_hot(label_batch_for_test_summary, td.num_classes)
        # # save labels for matrix-plot
        # squeezed_summary = np.squeeze(label_batch_for_test_summary)  # shape=(1000,)
        # total_test_dict = {xs: example_batch_for_test_summary, ys: one_hot_label_batch_for_test_summary}
        # # save results for matrix-plot
        # print('[i] Predicting results...')
        # y_pred = sess.run(y_predict, feed_dict=total_test_dict)
        # # if testing and y_pred is not null array:
        # if args.confusion_matrix_normalization and y_pred.shape[0]:
        #     # when epochs done plot confusion matrix...
        #     feed_and_derive_confusion_matrix(squeezed_summary, y_pred, args.confusion_matrix_normalization, args.class_names)


if __name__ == '__main__':
    main()