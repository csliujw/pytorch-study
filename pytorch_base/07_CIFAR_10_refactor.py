"""
CIFAR_10 分类Demo
步骤
    - torchvision加载数据集&预处理
    - 定义网络
    - 定义损失函数和优化器
    - 训练网络&更新参数
    - 测试网络
"""
import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision
import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DOWN_LOAD = False


def get_load():
    """
    :return: train_load, test_load, classes
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), )  # 数据归一化
    ])

    train_set = torchvision.datasets.CIFAR10(root="./", train=True, transform=transform, download=DOWN_LOAD)
    test_set = torchvision.datasets.CIFAR10(root="./", train=False, transform=transform, download=DOWN_LOAD)

    train_load = torch.utils.data.DataLoader(
        train_set,
        batch_size=96,
        shuffle=True,
        num_workers=4,
    )
    test_load = torch.utils.data.DataLoader(
        test_set,
        batch_size=96,
        shuffle=False,
        num_workers=4
    )
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_load, test_load, classes


"""
展示一个图片
"""


class NetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)

        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x1 = torch.max_pool2d(torch.relu(self.conv1(x)), 2)
        x2 = torch.max_pool2d(torch.relu(self.conv2(x1)), 2)
        x_ = x2.view(x2.size()[0], -1)
        x3 = torch.relu(self.fc1(x_))
        x4 = torch.relu(self.fc2(x3))
        return self.fc3(x4)


"""训练代码"""


def train(model, optimizer, criterion, epochs, train_load):
    model.train()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    for i in range(epochs):
        sum_loss = 0.000
        # 注意，我们是4张图片为一个batch size。所以inpupt的shape=torch.Size([4, 3, 32, 32])
        for input, target in train_load:
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            out = model(input)
            loss = criterion(out, target)
            sum_loss = loss.item()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        print(f"current  sum loss is {sum_loss} lr is {optimizer.param_groups[0]['lr']} ")


"""验证代码"""


def validation(model, test_load):
    model.eval()
    correct = 0.0
    total = 0.0

    for data in test_load:
        image, target = data
        image = image.to(device)
        target = target.to(device)
        out = model(image)
        other_2, predicted_2 = torch.max(out.data, 1)
        total += target.size(0)
        correct += (predicted_2 == target).sum()
    print(f"total = {total} \t correct = {100 * (correct.item() * 1.0 / total)}%")


def main():
    model = NetModel()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    train_load, test_load, classes = get_load()
    train(model, optimizer, criterion, 20, train_load)
    validation(model, test_load)


if __name__ == '__main__':
    main()