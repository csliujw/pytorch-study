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
<<<<<<< HEAD
import torch.nn
import torch.optim
import torch.utils.data
import torchvision
import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

DOWN_LOAD = False
# 方便数据的可视化 Tensor 转 Image
show = ToPILImage()

# 定义transform操作
=======
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

# 方便数据的可视化 Tensor 转 Image
show = ToPILImage()

>>>>>>> 38aa34c13f5cc9cf085a446fd6798cf01fd88f8d
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), )  # 数据归一化
])
"""
def __init__(self, root, train=True,
             transform=None, target_transform=None,
             download=False):
<<<<<<< HEAD
这个是加载PyTorch内部预置的一些数据集，了解即可。
"""
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
=======
"""
train_set = torchvision.datasets.CIFAR10(root="./", train=True, transform=transform, download=True)
train_set = torchvision.datasets.CIFAR10(root="./", train=False, transform=transform, download=True)

train_load = torch.utils.data.DataLoader(
    train_set,
    batch_size=4,
    shuffle=True,
    num_workers=4
)

test_load = torch.utils.data.DataLoader(
    train_set,
    batch_size=4,
    shuffle=True,
>>>>>>> 38aa34c13f5cc9cf085a446fd6798cf01fd88f8d
    num_workers=4
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

<<<<<<< HEAD
"""
展示一个图片
"""


def show_data():
    # data是
    (data, label) = train_set[100]
    # 通道转换
    # [3,32,32] --> [32,32,3]
    img = np.transpose(data.numpy(), (1, 2, 0))
    cv2.imshow("a.jpg", img)
    cv2.waitKey(0)


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


def train(model):
    model.cuda()
    model.train()
    # 这个最后是做了一次softmax的？
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    epochs = 100

    for i in range(epochs):
        sum_loss = 0.000
        # 注意，我们是4张图片为一个batch size。所以inpupt的shape=torch.Size([4, 3, 32, 32])
        for input, target in train_load:
            input = input.cuda()
            target = target.cuda()
            optimizer.zero_grad()
            out = model(input)
            loss = criterion(out, target)
            sum_loss = loss.item()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        print(f"current  sum loss is {sum_loss} lr is {optimizer.param_groups[0]['lr']} ")


"""验证代码"""


def validation(model):
    model.eval()
    model.cuda()
    correct = 0
    total = 0
    for data in test_load:
        image, target = data
        image = image.cuda()
        target = target.cuda()
        out = model(image)
        # out shape = [4,10]
        # torch.max(out.data,0) 0 按x轴方向取max（每一列看作一个整体），可以取到10个值 4*10的矩阵，画个矩阵出来就理解了
        # torch.max(out.data,1) 1 按y轴方向取max（每一行看作一个整体），可以取到4个值 4*10的矩阵，画个矩阵出来就理解了
        # torch.max的用法再看下官方api  predicted_2 就是预测的结果
        other_2, predicted_2 = torch.max(out.data, 1)
        # target shape = 4
        # predicted_2 shape = 4
        # 4张图片，4个对应的最大概率值 和 真实标记
        # 预测正确的总数
        total += target.size(0)
        correct += (predicted_2 == target).sum()
    # 记得加item 取到数值
    print(f"total = {total} \t correct = {100 * (correct.item() * 1.0 / total)}%")


if __name__ == '__main__':
    model = NetModel()
    train(model)
    validation(model)
=======
(data, label) = train_set[100]
print(classes[label])
show(((data + 1) / 2).resize(100, 100))
>>>>>>> 38aa34c13f5cc9cf085a446fd6798cf01fd88f8d
