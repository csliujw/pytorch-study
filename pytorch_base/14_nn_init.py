"""
参数初始化策略
    - 好的参数初始化让模型迅速收敛
    - 坏的参数初始化让模型迅速奔溃
PyTorch中 nn.Module模块参数都采取了比较合适的初始化策略。一般无需我们考虑，但是也可手动设置

参数初始化时要避免极大值和极小值，防止出现梯度爆炸和梯度消失

PyTorch中的   nn.init模块专门为初始化设计，实现了常用的初始化策略
"""

import torch as t
import torch.nn as nn
from torch.nn import init

linear = nn.Linear(3, 4)
t.manual_seed(1)


def init1():
    # 等价于 linear.weight.data.normal_(0,std)
    init.xavier_normal_(linear.weight)
    print(linear.weight)


def init2():
    import math
    t.manual_seed(1)
    std = math.sqrt(2) / math.sqrt(7.)
    linear.weight.data.normal_(0, std)
    print(linear.weight.data)


def init3():
    model = None  # 假设是一个网络
    """
    对所有模型参数进行初始化
    """
    for name, params in model.named_parameters():
        if name.find('linear') != -1:
            # init linear
            params[0]  # weight
            params[1]  # bias
        elif name.find('conv') != -1:
            pass
        elif name.find('norm') != -1:
            pass


if __name__ == '__main__':
    init2()
