import threading
from time import sleep, ctime

number = 0


def sing():
    for i in range(5):
        print("singe")
        sleep(1)


def dance():
    for i in range(5):
        print("dance")
        sleep(1)


def fun1(number):
    print(f"number is {number}")


def count(times):
    global number
    for i in range(times):
        number += 1


def test1():
    print('程序开始于：', ctime())
    threading.Thread(target=sing).start()
    threading.Thread(target=dance).start()
    print('任务完成于：', ctime())


def test2():
    th1 = threading.Thread(target=fun1, args=(100,))
    th2 = threading.Thread(target=fun1, args=(100,))
    th1.start()
    th2.start()


def test3():
    th1 = threading.Thread(target=count, args=(100000,))
    th2 = threading.Thread(target=count, args=(100000,))
    th1.start()
    th2.start()


def main():
    print('程序开始于：', ctime())
    threading.Thread(target=sing).start()
    threading.Thread(target=dance).start()
    print('任务完成于：', ctime())

    while True:
        # 查看当前有多少线程在运行
        length = len(threading.enumerate())
        print(length)
        if length <= 1:
            return


if __name__ == '__main__':
    # 并发问题
    test3()
    print(number)
