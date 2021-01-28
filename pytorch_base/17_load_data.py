"""
数据处理
PyTorch提供了几个便捷的数据处理和增强工具。
- 数据加载
    - 自定义数据集实现
        - 集成Dataset
        - 实现 __getitem__ 方法 获取数据
        - 实现 __len__方法 返回数据长度
- 以猫狗大战数据集为例进行数据加载
    - 猫的图片命名是 cat.1231.jpg形式
    - 狗的图片命名是 dog.1233.jpg形式
"""
import torch as t
from torch.utils import data
import os
from PIL import Image
import numpy as np


class DogCat(data.Dataset):
    def __init__(self, root):
        # python2的遗留写法，python3中 super中是不用写方法的
        super(DogCat, self).__init__()
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]

    def __getitem__(self, item):
        img_path = self.imgs[item]
        # dog->1 cat->0
        label = 1 if 'dog' in img_path.split("/")[-1] else 0
        pil_img = Image.open(img_path)
        # 或者 np.array(pil_img)
        array = np.asarray(pil_img)
        # 返回tensor对象
        data = t.from_numpy(array)
        return data, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = DogCat("../data/DogAndCat/test/")
    data, label = dataset[0]
    print(data.size(), label)
