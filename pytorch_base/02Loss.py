"""
损失函数的使用
[batchSize, channel, H, w]

常用的损失函数
- 均方误差损失 MSELoss
- 交叉熵损失 CrossEntropyLoss
"""
import torch.nn

"""定义网络"""
import torch
import torch.nn

# 真实标记
target = torch.arange(0, 5, dtype=torch.float32)
# 训练数据集
train_data = torch.randn(10, 1, 26, 26)


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 3)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 5)

    def forward(self, x):
        # 卷积 --> 激活 --> 池化
        x1 = torch.max_pool2d(torch.relu(self.conv1(x)), 2)
        x2 = torch.max_pool2d(torch.relu(self.conv2(x1)), 2)
        # x2.view(1,-1) 意思就是 1,max
        f = x2.view(x2.size()[0], -1)
        f1 = torch.relu(self.fc1(f))
        f2 = torch.relu(self.fc2(f1))
        f3 = torch.relu(self.fc3(f2))
        return f3


def train():
    model = Net()
    model.train()
    criterion = torch.nn.MSELoss()
    epoch = 200
    for i in range(epoch):
        sum_loss = 0.0
        for tmp in train_data:
            out = model(tmp.unsqueeze(0))
            input = torch.softmax(out, 1)
            input = input.squeeze(0)
            # input and target need same size. target need float
            loss = criterion(input, target)
            loss.backward()
            sum_loss += loss.item()
        print(f"epoch {i} loss is {sum_loss}")


if __name__ == '__main__':
    train()
