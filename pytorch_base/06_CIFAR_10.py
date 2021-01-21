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
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

# 方便数据的可视化 Tensor 转 Image
show = ToPILImage()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), )  # 数据归一化
])
"""
def __init__(self, root, train=True,
             transform=None, target_transform=None,
             download=False):
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
    num_workers=4
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

(data, label) = train_set[100]
print(classes[label])
show(((data + 1) / 2).resize(100, 100))
