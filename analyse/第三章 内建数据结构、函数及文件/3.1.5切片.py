def test1():
    seq = [1, 2, 3, 4, 5, 6, 7, 8]
    print(seq[1:5])
    seq[3:4] = [6, 3]  # 3，4赋值为 6，3
    print(seq)
    print(seq[3:4])  # 6
    pass


def test2():
    seq = [1,  2, 3, 4, 5, 6, 7, 8]
    #      0  -1 -2 -3 -4 -5 -6 -7
    print(seq[-4:])  # 倒数四个
    print(seq[-6:-2])
    pass


def test3():
    pass


def test4():
    pass


def test5():
    pass


if __name__ == '__main__':
    test2()
