"""
用nn.Module构建网络
两种我觉得常用的，好用的方式
"""
import torch
import torch.nn


class Module1(torch.nn.Module):
    def __init__(self):
        super().__init__()  # python2.7的继承才要在super里写方法
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.dense1 = torch.nn.Linear(32 * 3 * 3, 128)
        self.dense2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.max_pool2d(torch.relu(self.conv(x)), kernel_size=2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.dense1(x))
        x = self.dense2(x)
        return x


class Module2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # in_channels, out_channels, kernel_size
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 1)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(1984, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out.view(conv_out.size(0), -1)
        out = self.dense(conv_out)
        return out