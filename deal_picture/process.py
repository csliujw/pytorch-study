# 处理图片mask的一些技巧
import numpy as np
import cv2
from PIL import Image


def test1(image_path):
    _, _, mask = cv2.split(cv2.imread(image_path))
    # 提取出mask中不同的色彩，这些可作为mask中的类别种数
    # 好像是按从小到达的顺序排列的。背景为黑色 即像素值为0 所以是最小的，在最前面
    ids = np.unique(mask)
    # 第一个id是背景色，不要。
    ids = ids[1:]
    # 这样 就生成了每个类别的mask二进制掩码. 像素值相同的归为一个类别。 且像素值是从小到大的顺序。
    # 这对图片的像素值要求相当的严格。不推荐使用.
    masks = mask == ids[:, None, None]
    import torch
    masks = torch.as_tensor(masks, dtype=torch.uint8)
    print(masks.shape)


def test2(image_path):
    _, _, mask = cv2.split(cv2.imread(image_path))
    np.where()
    pass


def main():
    test1("../image/02.jpg")


if __name__ == '__main__':
    main()
