"""
优化器
学习率无变化
"""
import torch

import torch.nn

# 设置随机数种子保证结果确定
torch.manual_seed(0)
#为当前GPU设置随机种子
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

# 自己写代码调整学习率
def adjustLr(epoch, optimizer):
    if epoch != 0 and epoch % 20 == 0:
        # 通过这个api调整学习率！！！
        optimizer.param_groups[0]['lr'] *= 0.5
        print(f"after adjust lr is {optimizer.param_groups[0]['lr']}")
        # print(f"adjust lr,current lr is {optimizer.state_dict()['param_groups'][0]['lr']}")


def train():
    model = Net()
    model.train()
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # 损失函数
    criterion = torch.nn.MSELoss()
    epoch = 1000
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

        if i is not 0 and i % 20 is 0:
            print(f"epoch {i} loss is {sum_loss}")
            adjustLr(i, optimizer)


if __name__ == '__main__':
    train()
