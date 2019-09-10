#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@time:2019/4/15 上午10:37
@author:bigmelon
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from matplotlib.backends import backend_agg
from q_tensor_name_lists_20 import se_chosen_excitations, sese_chosen_excitations, \
    simplified_chosen_excitations

plt.switch_backend('agg')


class PrecisionSummary:
    """
    summaries for avg precision per epoch(computed by acc per batch)
    """

    def __init__(self, session, writer, process_flag, num_batches, restore=False):
        self.session = session
        self.writer = writer
        self.num_batches = num_batches  # num batches per epoch
        # statistics per epoch -> avg precision on 1 epoch = xxx example
        self.Precision = float(0)
        self.precision_cache = float(0)
        self.sveb_tp = float(0)
        self.sveb_fn = float(0)
        self.recall = float(0)
        self.process_flag = process_flag
        # 首先判断每个batch中 哪几个样本是sveb 然后判断这几个sveb有没有分类正确 分类正确 -> tp+1 分类错误 -> fn+1
        # 最后将每个batch内部的sub_tp & sub_fn 在push处累加 然后做处理即可...
        # construct & restore placeholders & summary_ops for <<< all classes per epoch >>>
        placeholder_name = process_flag + '_precision_placeholder'
        summary_name = process_flag + '_precision'
        # todo for avg epoch precision
        if restore:
            self.precision_placeholder = self.session.graph.get_tensor_by_name(placeholder_name + ':0')
            self.precision_summary = self.session.graph.get_tensor_by_name(summary_name + ':0')
        else:
            self.precision_placeholder = tf.placeholder(dtype=tf.float32, name=placeholder_name)
            self.precision_summary = tf.summary.scalar(name=summary_name, tensor=self.precision_placeholder)

    def add(self, acc, sub_tp, sub_fn):  # acc = accuracy per batch
        # for total classes precision per epoch
        self.Precision += acc
        self.sveb_tp = self.sveb_tp + sub_tp
        self.sveb_fn = self.sveb_fn + sub_fn

    def push(self, epoch):  # total train/valid num batches
        self.precision_cache = self.Precision / self.num_batches
        if self.process_flag == 'test':
            self.recall = self.sveb_tp/(self.sveb_tp + self.sveb_fn)
        else:
            pass
        feed = {self.precision_placeholder: self.Precision / self.num_batches}  # compute average acc per batch
        # run
        summary = self.session.run(self.precision_summary, feed_dict=feed)
        # add
        self.writer.add_summary(summary, epoch)
        self.__clear_for_epoch()

    def __clear_for_epoch(self):
        # reset the value for convenience of add_op in next epoch
        self.Precision = float(0)
        self.sveb_tp = float(0)
        self.sveb_fn = float(0)


class LossSummary:
    """
    summaries for avg ce loss per epoch & per batch
    """

    def __init__(self, session, writer, process_flag, num_batches, restore=False):
        self.session = session
        self.writer = writer
        self.num_batches = num_batches
        self.loss_value = float(0)  # key='loss' value=loss value | need to reset for each epoch
        # build summary & placeholder
        placeholder_name = process_flag + '_loss_placeholder'
        summary_name = process_flag + '_loss'
        if restore:
            self.placeholder = self.session.graph.get_tensor_by_name(placeholder_name + ':0')
            self.summary_op = self.session.graph.get_tensor_by_name(summary_name + ':0')
        else:
            self.placeholder = tf.placeholder(tf.float32, name=placeholder_name)
            self.summary_op = tf.summary.scalar(summary_name, self.placeholder)

    def add(self, values):  # values=total ce loss per batch | num_samples=args.batch_size
        self.loss_value += values

    def push(self, epoch):
        # feed dict
        feed = {self.placeholder: self.loss_value / self.num_batches}  # avg ce loss per epoch
        # derive summary & add it to writer
        summary = self.session.run(self.summary_op, feed_dict=feed)
        self.writer.add_summary(summary, epoch)  # add current epoch's loss into summary
        # reset the loss=0 for next epoch data to fed in
        self.loss_value = float(0)


class ExcitationSummary:
    """
    use for derive the profile of 1*1*c excitation output => only work for test_models procedures
    想要通过这个summary实现 在训练的过程中 抽取一个batch抽取各个nsvfq各个类别的样例各一个 然后将其push进summary
    主要考虑如何抽取相应的样例 => 比较关键!!! todo 在train_senet_v2上实现抽取样例的代码
    """

    def __init__(self, session, writer, process_flag, model_flag, restore=False, path_ease=None):
        # path_ease is for change save excitation arr easily
        self.session = session
        self.writer = writer
        self.process_flag = process_flag  # add
        self.model = model_flag[0:len(model_flag)-6:1] + 'net'  # change 'sese_block' or 'se_block' into sesenet/senet
        self.num_channel = None
        self.num_classes = 5
        self.sequenced_lst = None
        self.path_ease = path_ease
        # tensor names
        # self.excitations = se_excitations  # todo unused
        # self.simplified_excitations = simplified_excitations  # todo unused
        self.chosen_excitations = se_chosen_excitations if model_flag == 'se_block' else sese_chosen_excitations
        self.simplified_chosen_excitations = simplified_chosen_excitations  # name simplified list of chosen

        # construct & restore placeholders & summary_ops for certain excitation layer tensors
        self.placeholders = {}
        summary_ops = []  # hold a dict of summary pb
        placeholder_name = process_flag + '_excitation_placeholder'
        summary_name = process_flag + '_excitation'
        for excitation in self.simplified_chosen_excitations:
            ph_name = placeholder_name + '_' + excitation
            sum_name = summary_name + '_' + excitation
            if restore:
                placeholder = self.session.graph.get_tensor_by_name(ph_name + ':0')
                summary_op = self.session.graph.get_tensor_by_name(sum_name + ':0')
            else:
                # shape=the line chart's shape -> tobe defined
                placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1050, 2100, 3), name=ph_name)
                summary_op = tf.summary.image(name=sum_name, tensor=placeholder)  # todo  funk this! tf.summary.scalar!
            self.placeholders[excitation] = placeholder
            summary_ops.append(summary_op)
        # todo 1 feed for summaries use merge | >=2 feed for summaries use feed_dict & ph_dict
        self.summary_ops = summary_ops

    def push(self, sample_lst, xs, epoch):
        """
        train/valid/test_models - bool to chose mode | xs tensor of the input placeholder
        """
        # 3 lst with tuples [(x,dy),(x,dy),(x,dy),(x,dy),(x,dy)] x=(1,512,2) => expand(1,1,512,2) than feed it| dy=()
        self.__sequencing(sample_lst)
        # construct lists for classes | len(lst)=num of total se blocks = 16
        excitation_list_n = []
        excitation_list_s = []
        excitation_list_v = []
        excitation_list_f = []
        # excitation_list_q = []
        for i, xi in enumerate(self.sequenced_lst):  # nsvfq
            feed_dict = {xs: xi}
            for _ in self.chosen_excitations:
                tensor = self.session.graph.get_tensor_by_name(_)
                if i == 0:
                    excitation_list_n.append(self.session.run(tensor, feed_dict=feed_dict))
                elif i == 1:
                    excitation_list_s.append(self.session.run(tensor, feed_dict=feed_dict))
                elif i == 2:
                    excitation_list_v.append(self.session.run(tensor, feed_dict=feed_dict))
                elif i == 3:
                    excitation_list_f.append(self.session.run(tensor, feed_dict=feed_dict))
                elif i == 4:
                    pass
                    # excitation_list_q.append(self.session.run(tensor, feed_dict=feed_dict))
                else:
                    exit('[!] Wrong sequenced list! summaries_classes - line163')
        # reconstruct lists according to the excitation tensors(total 16 tensors chose 4 of them -2-6-12-15-)
        # lists with len=5 (nsvfq of various shapes of numpy array)
        # excitation_2 = [excitation_list_n[0]] + [excitation_list_s[0]] + [excitation_list_v[0]] + \
        #     [excitation_list_f[0]] + [excitation_list_q[0]]
        # excitation_6 = [excitation_list_n[1]] + [excitation_list_s[1]] + [excitation_list_v[1]] + \
        #     [excitation_list_f[1]] + [excitation_list_q[1]]
        # excitation_12 = [excitation_list_n[2]] + [excitation_list_s[2]] + [excitation_list_v[2]] + \
        #     [excitation_list_f[2]] + [excitation_list_q[2]]
        # excitation_15 = [excitation_list_n[3]] + [excitation_list_s[3]] + [excitation_list_v[3]] + \
        #     [excitation_list_f[3]] + [excitation_list_q[3]]
        excitation_2 = [excitation_list_n[0]] + [excitation_list_s[0]] + [excitation_list_v[0]] + \
            [excitation_list_f[0]]
        excitation_6 = [excitation_list_n[1]] + [excitation_list_s[1]] + [excitation_list_v[1]] + \
            [excitation_list_f[1]]
        excitation_12 = [excitation_list_n[2]] + [excitation_list_s[2]] + [excitation_list_v[2]] + \
            [excitation_list_f[2]]
        excitation_15 = [excitation_list_n[3]] + [excitation_list_s[3]] + [excitation_list_v[3]] + \
            [excitation_list_f[3]]
        # save for data analysis every 50 epoch
        if epoch % 50 == 0 and epoch != 0:
            self.__save_tensor_for_statistics(excitation_2, 1, 1, epoch)
            self.__save_tensor_for_statistics(excitation_6, 2, 2, epoch)
            self.__save_tensor_for_statistics(excitation_12, 3, 2, epoch)
            self.__save_tensor_for_statistics(excitation_15, 4, 1, epoch)
        # write summaries
        self.__excitation_line_chart(excitation_2, 1, 1)  # [n:(1,1,1,x) s:(1,1,1,x)...]
        self.__fig_to_rgb_array()
        feed = {self.placeholders[self.simplified_chosen_excitations[0]]: self.fig_array}
        summary = self.session.run(self.summary_ops[0], feed_dict=feed)
        self.writer.add_summary(summary, epoch)
        self.__excitation_line_chart(excitation_6, 2, 2)  # [n:(1,1,1,x) s:(1,1,1,x)...]
        self.__fig_to_rgb_array()
        feed = {self.placeholders[self.simplified_chosen_excitations[1]]: self.fig_array}
        summary = self.session.run(self.summary_ops[1], feed_dict=feed)
        self.writer.add_summary(summary, epoch)
        self.__excitation_line_chart(excitation_12, 3, 2)  # [n:(1,1,1,x) s:(1,1,1,x)...]
        self.__fig_to_rgb_array()
        feed = {self.placeholders[self.simplified_chosen_excitations[2]]: self.fig_array}
        summary = self.session.run(self.summary_ops[2], feed_dict=feed)
        self.writer.add_summary(summary, epoch)
        self.__excitation_line_chart(excitation_15, 4, 1)  # [n:(1,1,1,x) s:(1,1,1,x)...]
        self.__fig_to_rgb_array()
        feed = {self.placeholders[self.simplified_chosen_excitations[3]]: self.fig_array}
        summary = self.session.run(self.summary_ops[3], feed_dict=feed)
        self.writer.add_summary(summary, epoch)

    def __sequencing(self, lst):
        """
        sequence the train/valid/test_models/lst -> nsvfq ordered lst
        """
        sequenced_lst = [0]*self.num_classes
        for _ in lst:
            sequenced_lst[_[1]] = np.expand_dims(_[0], axis=0)  # convert (1,512,2) -> (1,1,512,2)
        self.sequenced_lst = sequenced_lst

    def __excitation_line_chart(self, excitation_lst, i, j):  # (1,5)/(1,5)
        self.num_channel = excitation_lst[0].shape[-1]  # derive the channels
        fig = plt.figure(figsize=(3.5, 1.75), dpi=600)  # figsize unit= inch -> (3.5x600,1.75x600,3)=(2100,1050,3)
        plt.rcParams['font.family'] = 'Helvetica'
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号
        x = np.arange(self.num_channel)
        n = np.squeeze(excitation_lst[0])  # (1,1,1,x)
        s = np.squeeze(excitation_lst[1])
        v = np.squeeze(excitation_lst[2])
        f = np.squeeze(excitation_lst[3])
        # q = np.squeeze(excitation_lst[4])
        plt.grid(linestyle='-', linewidth=0.2)
        ax = plt.gca()  # 得到坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.2)
        ax.spines['bottom'].set_linewidth(0.2)
        ax.tick_params(which='both', width=0.2, length=1, colors='k', direction='in', pad=1)
        # todo color need to be defined!
        plt.plot(x, n, color='black', label='N', linewidth=0.5)
        plt.plot(x, s, color='royalblue', label='S', linewidth=0.5)
        plt.plot(x, v, color='orangered', label='V', linewidth=0.5)
        plt.plot(x, f, color='mediumseagreen', label='F', linewidth=0.5)
        # plt.plot(x, q, color='grey', label='Q', linewidth=0.5)
        x_scales = np.arange(start=0, stop=self.num_channel, step=1)  # stop=num of channels of input tensor
        plt.xticks(x, x_scales, fontsize=2.1, fontweight='light')
        plt.yticks(fontsize=2.1, fontweight='light')
        # todo set i&j
        plt.title(f'SE_block{i}_unit{j}', fontsize=4.2, fontweight='semibold')
        plt.xlabel('Channel Index', fontsize=3, fontweight='semibold', labelpad=0.5)
        plt.ylabel('Activation', fontsize=3, fontweight='semibold', labelpad=0.5)
        plt.xlim(0, self.num_channel)  # todo set x range -> range from 0-num_channel
        plt.ylim(0, 1)  # todo set y range -> range from 0-1 for activation
        plt.legend(loc='upper right', frameon=True, markerscale=0, labelspacing=0.1, handlelength=1, handletextpad=0.1)
        legend = plt.gca().get_legend()
        legend_text = legend.get_texts()
        plt.setp(legend_text, fontsize=0.6, fontweight='light')
        self.fig = fig

    def __fig_to_rgb_array(self):
        canvas = backend_agg.FigureCanvasAgg(self.fig)  # draw canvas
        canvas.draw()
        buf = self.fig.canvas.tostring_rgb()
        cols, rows = self.fig.canvas.get_width_height()  # cols=2100 rows=1050
        self.fig_array = np.fromstring(buf, dtype=np.uint8).reshape(1, rows, cols, 3)  # (3.5*dpi,1.75*dpi,3)
        # todo try to close the plots in order to getting rid of warnings: -> rcParam `figure.max_open_warning`)
        # todo 并没有解决 max_open_warning的问题!!!!!
        self.fig.clf()  # close current fig

    def __save_tensor_for_statistics(self, lst, i, j, e):
        """
        save the value of tensors in numpy for data statistics.
        """
        arr = np.array(lst)
        # save array according to the model categories/process flag
        # todo ----->
        path = '/share/donghao/demo10/exp2/logs/'+self.path_ease+f'/{self.model}_excitation_arrays/{self.process_flag}'
        if os.path.exists(path):
            np.save(path + f'/e{e}_block{i}_unit{j}.npy', arr)
        else:
            os.makedirs(path)
            np.save(path + f'/e{e}_block{i}_unit{j}.npy', arr)
# todo max_open_warning:
# More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`)
# are retained until explicitly closed and may consume too much memory.
# (To control this warning, see the rcParam `figure.max_open_warning`).


class NetSummary:
    def __init__(self, session):
        self.session = session
        self.scopes = []
        self.weight_scopes = []  # 应该放在__build_name之前
        self.bias_scopes = []
        self.__build_names()
        self.summary_op = None

    def __build_names(self, shadow_var_flag=False):
        """
        Not using shadow variables by default...
        """
        # build total weight/biases scope
        if not shadow_var_flag:
            for _ in tf.trainable_variables():
                if _.name.find('weights') == -1:  # not found
                    pass
                else:
                    self.weight_scopes.append(_)
                if _.name.find('biases') == -1:
                    pass
                else:
                    self.bias_scopes.append(_)
            pass
        else:
            ...
        # build total scope
        self.scopes = self.weight_scopes + self.bias_scopes

    def build_summaries(self, restore):
        """
        this func is to build summaries for net object params like (weight/bias/etc)
        """
        if restore:  # name verified
            self.summary_op = self.session.graph.get_tensor_by_name('net_summaries/net_summaries:0')

        sess = self.session
        summaries = []
        with tf.variable_scope('filter_summaries'):
            for _ in self.weight_scopes:
                tensor = sess.graph.get_tensor_by_name(_.name)
                summary = tf.summary.histogram(_.name, tensor)
                summaries.append(summary)
        with tf.variable_scope('bias_summaries'):
            for _ in self.bias_scopes:
                tensor = sess.graph.get_tensor_by_name(_.name)
                summary = tf.summary.histogram(_.name, tensor)
                summaries.append(summary)

        # this op creates a [`Summary`] protocol buffer that contains the union of all the values in the input summaries.
        self.summary_op = tf.summary.merge(summaries, name='net_summaries')
