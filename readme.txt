# [项目文件说明]
  + train_isenets & train_senets - 分别是针对一维度信号的senet&isenet的python训练执行文件
  + q_resnet_v1 & q_resnet_v2 - 是模型结构定义文件(针对一维信号的5分类模型)
  + q_trainingdata - 模块负责整个神经网络模型的 "data feeding pipeline" 
                     采用了python3 multiprocessing模块 并尝试构建数据池的方式加速数据录入
  + test_models - 包含模型测试程序样例
  + summaries - 模块包含了：tensorboard中针对模型参数可视化的监测代码部分 
                以及 isenet/senet 训练、测试中的结果、各项评价指标的数值变动的监测跟踪部分
