"""
Tensor 张量 可用gpu加速计算。与numpy的ndarrays相似
tensor分类
- 接口角度
    - torch.function torch.方法  如 torch.save
    - tensor.function tensor类型数据.方法 如 tensor.view
- 存储角度来分类
    - 不会修改自身数据 a.add(b)，操作的结果会返回一个新tensor
    - 会修改自身数据 a.add_(b)，操作的结果仍存储在a中
"""
"""
- Python传参形式。*size 元组  **key 字典，请看08_tensor_syntax

- Tensor(*size)                     基础构造函数,创建Tensor时，系统不会马上分配空间，只会计算剩余的内存是否够用，用到时才分配。其他都是创建时完成分配。
- one(*size)                        全1的Tensor
- zeros(*size)                      全0的Tensor
- eye(*size)                        对角线为1，其他为0，即单位矩阵？
- arange(s,e,step)                  从s到e,步长为step
- linspace(s,e,steps)               从s到e,均匀切分成steps份
- rand/randn(*size)                 均匀/标准分布
- normal(mean,std)/uniform(from,to) 正态分布/均匀分布  normal这些的用法请百度。。。
- randperm(m)                       随机排列

==================
调整Tensor形状
- view      前后元素总数要一致，view不会修改自身的数据，会返回新的Tensor，与原有Tensor共享内存。怎么做到的？索引吗？
- resize    可修改尺寸，
- unsqueeze 增加一个维度 eg shape=[1,4] --> shape=[1,1,4]     与原Tensor共享内存  --> 这种共享内存是以牺牲速度为代价的吧
- squeeze   减少一个维度 eg shape=[1,4] --> shape=[4]         与原Tensor共享内存
"""
import torch


def testTensorConstruct():
    # 形状2*3的
    data1 = torch.Tensor(2, 3)
    # 数据为2 3的
    data2 = torch.Tensor((2, 3))
    # 接受一个list
    list = [1, 2, 3, 4, 5, 6, 6]
    data3 = torch.Tensor(list)
    # Tensor转list
    data4 = data3.tolist()
    # Tensor的大小，可用shape也可用size
    data3.shape
    data3.size()
    # 创建形状一样的Tensor
    data4 = torch.Tensor(data3.shape)
    # 统计Tensor中元素的总个数
    data4.numel()
    data4.nelement()


def testTensor():
    ones = torch.ones(2, 3)
    zeros = torch.zeros(2, 3)
    eye = torch.eye(3, 3)
    rand = torch.rand(3, 3)
    randn = torch.randn(30, 30)
    # print(randn.normal_(mean=1, std=0.5))
    # 长度为10的随机排列，可用于数据集随机取数据，随机排列生成索引,取索引中的数据
    torch.randperm(9)


def adjustShape():
    """
    调整Tensor形状
        - view      前后元素总数要一致，view不会修改自身的数据，会返回新的Tensor，与原有Tensor共享内存。怎么做到的？索引吗？
        - resize    可修改尺寸，
        - unsqueeze 增加一个维度 eg shape=[1,4] --> shape=[1,1,4]     与原Tensor共享内存  --> 这种共享内存是以牺牲速度为代价的吧
        - squeeze   减少一个维度 eg shape=[1,4] --> shape=[4]         与原Tensor共享内存
    """
    data = torch.Tensor([[1, 2, 3, 4]])
    d1 = data.unsqueeze(0)
    d2 = data.squeeze(0)
    data[0][0] = 100
    print(d1)


def slices():
    """
    Tensor的切片操作
    """
    data = torch.randn(3, 4)
    d1 = data[:, 0]
    d2 = data[0, -1]  # 第0行的最后一个元素
    d3 = data[:2]  # 前两行
    d4 = data > 1  # 返回一个ByteTensor 与原Tensor形状一致，符合条件的对应值为1，不符合的为0
    d5 = data[data > 1]  # 等价于 data.masked_select(data>1),选择结果与原Tensor不共享内存。选择结果是一维的Tensor
    print("===============")
    print(data)
    print("===============")
    print(d1)
    print("===============")
    print(d2)
    print("===============")
    print(d3)
    print("===============")
    print(d4)
    print("===============")
    print(d5)
    print("===============")


def other_slices():
    """  要用再说
    - index_select(input,dim,index)     指定维度dim上选取，例如选取某些行，某些列
    - masked_select(input,mask)         例子如上，a[a>0],使用ByteTensor进行选取
    - non_zero(input)                   非0元素的下标
    - gather(input,dim,index)           根据index，在dim维度上选取数据，输出的size与index一样
    """
    pass


if __name__ == '__main__':
    slices()
