"""
常用的图像相关层
"""
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

lena = Image.open("../image/lena.jpg")
Image._show(lena)

to_tensor = ToTensor()
to_PIL = ToPILImage()
"""
感觉上面的代码没什么可写的，只做笔记了。
- 卷积层   
- 池化层
- 全连接层 Linear
- 批规范化层 BatchNorm 分为1D 2D 3D，除了标准的BatchNorm外还有风格迁移种常用的InstanceNorm层
- Dropout层  用来防止过拟合，分为1D，2D，3D
"""