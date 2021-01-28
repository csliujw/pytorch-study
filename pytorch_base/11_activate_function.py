"""
激活函数
@date 2021-01-22 16：00
"""

import torch
import torch.nn
import torch.nn.functional as F
# inplace设为True的化，会把输出直接覆盖到输入中，可节省内存开销。一般不推荐inplace=True
torch.nn.ReLU(inplace=True)
