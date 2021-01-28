"""
优化器

常用的优化方法封装在torch.optim中，设计灵活，可方便拓展成自定义的优化方法
SGD举例
- 优化方法的基本使用
- 对模型的不同部分设置不同的学习率
- 如何调整学习率
"""
import torch
import torch.nn
import torch.optim


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.MaxPool2d(2, 2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16 * 5 * 5, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        # x的shape 需要变成 (?,400)
        x = x.view(-1, 16 * 5 * 5)
        # 这里输入的x是[n,400] n应该是数据的输入量，如10个样本
        x = self.classifier(x)
        return x


def main():
    model = Module()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)
    optimizer.zero_grad()
    input = torch.randn(1, 3, 32, 32)
    output = model(input)
    # fake backward
    output.backward(output)
    optimizer.step()


def optimizerTest():
    """
    为不同网络设置不同的学习率
    如果对某个参数不指定学习率，就使用默认学习率
    """
    model = Module()
    optimizer = torch.optim.SGD([
        {'params': model.feature.parameters()},
        {'params': model.classifier.parameters(), 'lr': 1e-2},
    ], lr=1e-5)

    optimizer2 = torch.optim.SGD(model.parameters(), lr=1e-2)
    # print(optimizer2)
    print(optimizer)


def adjustLr():
    """
    调整学习率
    - 法一、重新创建优化器
        - 简单，便捷，重建优化器的开销也小。但是新建优化器会重新初始化动量等状态信息，对动量优化器来说，可能会造成损失函数在收敛的过程中出现震荡。
    - 法二、修改优化器的optimizer.param_groups中对应的学习率
        - 这个我也不怎么用
    - 法三、torch自带的学习率调整器lr_scheduler
    """
    model = Module()
    # 重建优化器调整学习率。不用多说。easy。

    # 手动调整学习率，
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)
    old_lr = optimizer.param_groups[0]['lr']
    optimizer.param_groups[0]['lr'] = 1e-4
    # print(optimizer.param_groups[0]['lr'])

    # torch的学习率调整器，还有高级用法，以后补充
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)
    for i in range(200):
        optimizer.zero_grad()
        optimizer.step()
        scheduler.step()
        if i != 0 and i % 20 == 0:
            print(optimizer.param_groups[0]['lr'])


if __name__ == '__main__':
    adjustLr()
