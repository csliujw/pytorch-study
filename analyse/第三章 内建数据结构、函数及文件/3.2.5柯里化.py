"""
通过部分参数应用的方式从已有的函数中衍生出新的函数
"""


def add_numbers(x, y):
    return x + y


if __name__ == '__main__':
    # 第二个参数对于函数add_numers就是柯里化了
    new_func = lambda y: add_numbers(7, y)
    print(new_func(2))
