def test1():
    # 创建元组
    tup = 4, 5, 6
    print(tup)


def test2():
    tup = (4, 5, 6), (7, 8)
    print(tup)


# 记忆
def test3():
    # tuple 把任意类型转成元组
    tup = tuple("string")
    print(tup)


def test4():
    # 如果元组中的一个对象是可变的，例如列表，你可以在它内部进行修改
    tup = tuple(['foot', [1, 2]])
    tup[1].append(3)
    print(tup)


# 记忆
def test5():
    # 可以使用 + 号连接元组来生成更长的元组
    tup = (1, 2, 3, 4)
    tup = tup + (5, 6, 7)
    print(tup)


# 记忆
def test6():
    # 将元组乘以整数，则会和列表一样，生成含有多份拷贝的元组：
    # 新的元組哦 numpy好像是广播机制
    tup = (1, 2, 3, 4, 5)
    new_tup = tup * 4
    print(tup)
    print(new_tup)


def test7():
    # 元组拆包
    # a, b, c = (1, 2, 3)
    # print(a, b, c)
    tup = ((1, 2, 3), (11, 22, 33))
    for a, b, c in tup:
        print(a, b, c)


# 背
def test8():
    # 高级拆包
    value = 1, 2, 3, 4, 5, 6
    a, b, *rest = value
    print(a, b, rest)
    pass


# 背
def test9():
    # 统计xx出现的次数
    a = (1, 2, 3, 4, 5, 6, 7, 7, 7, 7)
    n_1 = a.count(7)
    b = [1, 2, 3, 3, 3, 3, 3]
    n_2 = b.count(3)
    print(n_1, n_2)


if __name__ == '__main__':
    pass