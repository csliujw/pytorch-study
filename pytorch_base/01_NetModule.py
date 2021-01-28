"""定义网络"""
import torch
import torch.nn


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1,
             bias=True, padding_mode='zeros'):
        """
        self.conv1 = torch.nn.Conv2d(1, 6, 3)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        """
        def __init__(self, in_features, out_features, bias=True):
        [1,400] * [400,120] = [1,120]??
        """
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 5)
        """
        反推以下原图的大小
        16*5*5 16是通道数。5*5是图的大小。
        原图 = 现图 + (kernel-1) (因为stride=1)
        他经过了两次池化，两次卷积。
        原来的顺序是 卷积 --> 激活 --> 池化
        推回去的时候要反推
        5*5 --> 10*10(2*2最大池化前) --> 10+3-1(卷积前) 
            --> 24*24(2*2最大池化前) --> 24+3-1(卷积前)
        所以原图是1*26*26 . 1为通道数，26*26为图的大小。
        """

    def forward(self, x):
        # 卷积 --> 激活 --> 池化
        """
        c1 = self.conv1(x)
        a1 = torch.relu(c1)
        p1 = torch.max_pool2d(a1)
        """
        x1 = torch.max_pool2d(torch.relu(self.conv1(x)), 2)
        x2 = torch.max_pool2d(torch.relu(self.conv2(x1)), 2)
        """
        在进行全连接操作。全连接操作前需要把数据都进行展平,
        最后一个全连接层一般不进行激活。
        """
        # 这个api不是很熟悉 x2.size() is torch.Size([1, 16, 5, 5])
        # x2.size()[0] 就是1
        # x2.view(1,-1) 意思就是 1,max
        f = x2.view(x2.size()[0], -1)
        f1 = torch.relu(self.fc1(f))
        f2 = torch.relu(self.fc2(f1))
        f3 = torch.relu(self.fc3(f2))
        return f3


def train():
    label = ['dog', 'cat', 'pig', 'duck', 'fish']
    model = Net()
    # model.train() 训练
    # 验证
    model.eval()
    # 假数据。 10个1*26*26的图。 就是10个通道数为1，图片26*26像素的图
    train_data = torch.randn(10, 1, 26, 26)
    # 开始预测
    for tmp in train_data:
        out = model(tmp.unsqueeze(0))
        # 对预测结果用softmax进行分类,计算每种类别的概率
        result = torch.softmax(out, 1)
        arg = torch.Tensor.argmax(result)
        # 输出预测的那个图片的类别
        print(label[arg])
        # 输出预测结果的最大概率
        # print(torch.Tensor.max(result))


def net_parameter():
    """网络模型的一些参数"""
    model = Net()
    for name, parameter in model.named_parameters():
        print(f"name is {name} \t parameter size is {parameter.size()}")


if __name__ == '__main__':
    train()
