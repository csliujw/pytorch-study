"""
多gpu并行计算,只是做个科普，我这种水平，基本用不到。

def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module (Module): the module to evaluate in parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
- device_ids 指定在那些GPU上进行优化
- output_device 指定输出到那个gpu上。
- 上面那个可以直接利用多GPU并行计算得出结果。
"""

import torch
import torch.nn as nn

net = nn.Module
new_net = nn.DataParallel(net, device_ids=[0, 1])
output = new_net("input data")

output = nn.parallel.data_parallel(net,"input",device_ids=[0,1])

