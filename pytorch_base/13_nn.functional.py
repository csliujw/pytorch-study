"""
nn.functional & nn.Module
    - 如果模型有学习参数，最好用nn.Module 即 nn.Linear(3, 4)这种形式的方法
    - 如果模型没有学习参数，可以用nn.Module 或者 nn.functional.linear,
    - 二者性能上没太大的差距

- 激活函数（ReLU、sigmoid、tanh）、池化（MaxPool）等层没有可学习参数，可以用对应的functional函数代替。
- 卷积层、全连接层等具有可学习参数的网络建议使用nn.Module 即 nn.Conv2d这种形式
- dropout操作虽然没有可学习的参数，但是建议使用nn.Dropout，因为dropout在训练和测试两个阶段的行为有所差别。使用nn.Module对象能够通过model.eval操作加以区分

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def test1():
    input = torch.randn(2, 3)
    model = nn.Linear(3, 4)
    output1 = model(input)
    output2 = nn.functional.linear(input, model.weight, model.bias)
    print(output1 == output2)


class NetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        """就不写了"""
        pass

if __name__ == '__main__':
    test1()
