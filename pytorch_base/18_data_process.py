"""
数据处理
上一份代码只是加载了图片，但是图片的形状大小是不一样的。
    - 返回样本的形状不一，每张图片大小不一样，这对于需要取batch训练的神经网络说很不友好
    - 返回样本的数值较大、未归一化至[-1,1]

可用torchvision进行图像处理
    - transforms模块提供了对PIL Image对象和Tensor对象的常用操作
    - 对PIL常见操作如下：
        - Resize：调整图片尺寸
        - CenterCrop、RandomCrop、RandomSizedCrop：裁剪图片
        - Pad：填充
        - ToTensor：PIL Image对象转成Tensor，会自动将[0,255]归一化至[0,1]
    - 对Tensor的常见操作如下：
        - Normalize：标准化，即减均值，除以标准差。
        - ToPILImage：将Tensor转为PIL Image对象
    - 要对图片进行多个操作的话，可通过transforms.Compose([]) 进行拼接
        - eg:
            T.Compose([
                T.Resize(224),  # 缩放图片（Image） 保持长宽比不变，最短边为224像素
                T.CenterCrop(224),  # 从图片中间切出224*224的图片
                T.ToTensor(),  # 将图片（Image）转成 Tensor，归一化至[0,1]
                T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
            ])
"""
import torch as t
import torch.optim as optim
from torch.utils import data
import os
from PIL import Image
import numpy as np
import torchvision.transforms as T

"""
transform 还可用lambda表达式封装自定义的转换策略。
"""
transformObj = T.Compose([
    T.Resize(224),  # 缩放图片（Image） 保持长宽比不变，最短边为224像素
    T.CenterCrop(224),  # 从图片中间切出224*224的图片
    T.ToTensor(),  # 将图片（Image）转成 Tensor，归一化至[0,1]
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
])


class DogCat(data.Dataset):
    def __init__(self, root):
        # python2的遗留写法，python3中 super中是不用写方法的
        super(DogCat, self).__init__()
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transform = transformObj

    def __getitem__(self, item):
        img_path = self.imgs[item]
        label = 1 if 'dog' in img_path.split("/")[-1] else 0
        data = Image.open(img_path)
        if self.transform is not None:
            # transform的对象必须是Image对象。
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = DogCat("../data/DogAndCat/test/")
    data, label = dataset[3]
    data = np.transpose(data.numpy(), (1, 2, 0))
