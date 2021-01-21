"""
优化器
pytorch内置api调整学习率
pytorch调整学习率相应博客 https://www.freesion.com/article/603177699/
"""
import torch

import torch.nn

# 设置随机数种子保证结果确定
torch.manual_seed(0)
# 为当前GPU设置随机种子
torch.cuda.manual_seed(0)
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


def lr_scheduler(optimizer):
    """
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
    - optimizer：优化器
    - step_size：每多少个epoch更新一次学习率
    - gamma：学习率decay因子
    - last_epoch：最后一个更新学习率的epoch，默认为-1，一直更新
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.7, last_epoch=-1)


def train():
    model = Net()
    model.train()
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # 学习率调整器
    scheduler = lr_scheduler(optimizer)
    # 损失函数
    criterion = torch.nn.MSELoss()
    epoch = 60
    for i in range(epoch):
        # 每20轮调整一次学习率
        sum_loss = 0.0
        for tmp in train_data:
            optimizer.zero_grad()
            out = model(tmp.unsqueeze(0))
            input = torch.softmax(out, 1).squeeze(0)
            loss = criterion(input, target)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        scheduler.step()
        if i != 0 and i % 20 == 0:
            print(f"epoch {i} loss is {sum_loss}, lr is {optimizer.param_groups[0]['lr']}")


if __name__ == '__main__':
    train()
