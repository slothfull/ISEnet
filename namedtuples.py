#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@time:2019/3/8 下午11:50
@author:bigmelon

@time:2019/2/25 下午4:30
@author:bigmelon
todo 注意对于这种使用collections.namedtuple结构作为载体传递数据的方式
todo 通常需要传递前后的文件都含有同样的namedtuple定义 然而推荐使用单独定义namedtuple文件
todo 在数据发送文件和数据接受文件中都包含此namedtuple的方式
todo 不推荐直接引用的方式 -> from 数据发送.py import namedtuple的方式
"""
from collections import namedtuple

SAMPLE = namedtuple('SAMPLE', ['filename', 'label'])
