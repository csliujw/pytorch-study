"""
优化器
学习率无变化
"""
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
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # 损失函数
    criterion = torch.nn.MSELoss()
    epoch = 200
    for i in range(epoch):
        sum_loss = 0.0
        for tmp in train_data:
            optimizer.zero_grad()

            out = model(tmp.unsqueeze(0))
            input = torch.softmax(out, 1).squeeze(0)
            loss = criterion(input, target)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        """
        {'state': {}, 'param_groups': [{'lr': 0.001, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [2470538377976, 2470538378048, 2470538378120, 2470538378192, 2470538378264, 2470538378336, 2470538378408, 2470538378480, 2470538378552, 2470538378624]}]}
        print(optimizer.state_dict())
        """
        # print(f"lr is {optimizer.state_dict()['param_groups'][0]['lr']}")
        # print(f"epoch {i} loss is {sum_loss}")


if __name__ == '__main__':
    train()
