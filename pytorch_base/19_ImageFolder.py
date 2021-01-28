"""
ImageFolder 点进源码看一下。
ImageFolder(root=,transform=,target_transform=,loader=)
- root：在root指定的路径下寻找图片
- transform：对PIL Image进行转换操作，transform的输入是使用loader读取图片的返回对象
- target_transform：对label的转换
- loader：指定加载图片的函数，默认操作时读取为PIL Image对象
- PyTorch中图片对象一般都是 C*H*W的形式 即 [通道 高度 宽度]
======================================================
Dataset 只负责数据的抽象
DataLoader  负责数据的加载
"""
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.utils.data as data
import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from torch.utils.data._utils.collate import default_collate


def testImageFolder():
    dataset = ImageFolder("../data/DogAndCat/train/")
    class_to_idx = dataset.class_to_idx
    print(class_to_idx)
    imgs = dataset.imgs
    print(f"第1张图的路径信息：{imgs[0][0]},\t 第1张图的label {imgs[0][1]} \t label的含义看dataset.class_to_idx ]")


def showImage():
    dataset = ImageFolder("../data/DogAndCat/train/", transform=transformObj)
    # 还原图片 0.2 and 0.4 是标准差和均值的近似
    plt.imshow(T.ToPILImage()(dataset[0][0] * 0.2 + 0.4))
    plt.show()


transformObj = T.Compose([
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
])

dataset = ImageFolder("../data/DogAndCat/train/", transform=transformObj)
"""
def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
             batch_sampler=None, num_workers=0, collate_fn=default_collate,
             pin_memory=False, drop_last=False, timeout=0,
             worker_init_fn=None):
- dataset：      加载的数据集（Dataset对象）
- batch_size：   batch size（批大小）
- shuffle：      是否打乱数据
- sampler：      样本抽样
- num_workers：  使用多进程加载的进程数，0代表不使用多进程
- collate_fn：   如何将多个样本数据拼接成一个batch，一般使用默认拼接方式
- pin_memory：   是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
- drop_last：    dataset中数据若不是batch_size的整数倍，drop_last 为 True则会将不足一个batch_size的数据进行丢弃。
"""


def testDataLoader():
    dataLoader = DataLoader(dataset=dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)
    dataiter = iter(dataLoader)
    imgs, labels = next(dataiter)
    print(imgs.size())
    print(labels)


"""
个别样本错误，无法读取。
"""


class DogCat(data.Dataset):
    def __init__(self, root):
        # python2的遗留写法，python3中 super中是不用写方法的
        super(DogCat, self).__init__()
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transform = transformObj

    def __getitem__(self, item):
        try:
            img_path = self.imgs[item]
            label = 1 if 'dog' in img_path.split("/")[-1] else 0
            img = Image.open(img_path)
            if self.transform is not None:
                # transform的对象必须是Image对象。
                img = self.transform(img)
            return img, label
        except:
            return None, None

    def __len__(self):
        return len(self.imgs)


def my_collate_fn(batch):
    """
    自定义collate
    :param batch: sucn as (data,label)
    :return:
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    return default_collate(batch)  # 用默认方式拼接过滤后的batch数据


def test_my_collate_fn():
    dataset = DogCat("../data/")
    print(dataset[0][0])
    pass

test_my_collate_fn()
