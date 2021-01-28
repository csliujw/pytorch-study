def fun1():
    # range是迭代的，不是一次生成。
    gen = range(0, 10)
    # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # 首                        尾？
    l = list(gen)
    print(l[0])
    l.pop()
    print(l)


def fun2():
    # 连接啥操作也是一样 +的连接是生成新的， extend是原列表扩展
    pass

if __name__ == '__main__':
    fun1()