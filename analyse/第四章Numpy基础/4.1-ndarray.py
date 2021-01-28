"""
多维数组对象
"""
import numpy as np

# shape的赋值为元组，这个设定号，尽量保持不可变。
np.zeros(shape=(3, 4))
"""
array       将输入数据（可以是列表、元组、数组以及其他序列）转换为naddary，如果不指明数据类型则数据类型会进行自动推断；默认赋值所有
asarray     将输入转为ndarray，若输入是ndarray则不在复制
arange      python内建函数range的数组版，返回一个数组

xxx.astype 转换数据类型,返回一个新的ndarray

numpy的计算有广播机制。

numpy的切片操作是原视图的，数据不是被复制了，任何对于视图的修改都会反映到原数组上
拷贝的话用.copy()方法
"""


def test1():
    data = np.array([1, 2, 3, 4, 5, 6])
    print(data.dtype)
    copy_data = data.astype(np.float32)
    print(copy_data.dtype)


def test2():
    data = np.asarray([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 11, 13]]])
    # (1, 4, 3)
    print(data.shape)
    # 取0 1
    print(data[0, 1,])


def test3():
    names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
    # 7 * 4 的 shape
    data = np.random.randn(7, 4)
    ret = names == 'Bob'
    print(ret)  # [ True False False  True False False False]  第0行和第3行
    # 布尔切片
    print(data)
    print("==========")
    print(data[ret])


def test4():
    # 取反 ~
    names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
    # 7 * 4 的 shape
    data = np.random.randn(7, 4)
    ret = names == 'Bob'
    print(data[~ret])


def test5():
    """多个条件"""
    names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
    mask = (names == 'Bob') | (names == 'Will')
    print(mask)
    pass


"""神奇索引"""


def test6():
    arr = np.empty((8, 4))
    for i in range(8):
        arr[i] = i  # 广播机制
    print(arr)
    print("==============")
    # 选出一个符合特定顺序的子集  屈 4 3 2 1 这四行 顺序也是4 3 2 1
    print(arr[[4, 3, 2, 1]])
    print("==============")
    # 负索引则从尾部取
    print(arr[[-4, -3, -2, -1]])
    print("==============")
    # 取得是 （1，1）（2，2）位置的值
    print(arr[[1,2],[1,2]])


if __name__ == '__main__':
    test6()
