# [项目文件说明]
  + train_isenets & train_senets -> 分别是针对一维度信号的senet&isenet的python训练执行文件
  + q_resnet_v1 & q_resnet_v2 -> 是模型结构定义文件(针对一维信号的5分类模型)
  + q_trainingdata -> 模块负责整个神经网络模型的 "data feeding pipeline" 
                      采用了python3 multiprocessing模块 并尝试构建数据池的方式加速数据录入
  + q_attention_module.py -> 模块负责构建网络结构中的通道注意力机制结构模块 分为se-block 和 ise-block两种
  + q_resnet_v1.py -> 模块负责构建resnet-v1网络结构的基本架构
  + q_resnet_v2.py -> 模块负责构建resnet-v2网络结构的基本架构
  + tensor_name_lsts -> 模块定义了针对不同深度(11层-26层)的网络架构中需要提取的tensor_names 
                        用于恢复训练状态参数、方便tensorboard参数可视化分析
  + utils
    - q_net_utils_11_to_26.py -> 模块包含了针对不同深度(11层-26层)网络架构中止训练后 恢复上一个训练节点的参数状态所需的函数
    - q_net_utils_50.py -> 模块包含了针对50层网络架构中止训练后 恢复上一个训练节点参数状态所需的函数
    - q_resnet_utils.py -> 模块包含了构建Resnet网络结构所需的函数
    - q_utils.py -> 模块包含了训练、测试结果混淆矩阵的绘制代码 
                    以及 训练时动态衰减的学习速率设置(支持根据针对每个网络结构、kernel分别定制不同的学习率)
                    以及 一个数据集(训练集、测试集random_shuffle工具类)
  + namedtuples.py -> 模块定义了训练集、测试集中数据样本的基本格式
  + test_models -> 包含模型测试程序样例
  + summaries -> 模块包含了：tensorboard中针对模型参数可视化的监测代码部分 
                以及 isenet/senet 训练、测试中的结果、各项评价指标的数值变动的监测跟踪部分
