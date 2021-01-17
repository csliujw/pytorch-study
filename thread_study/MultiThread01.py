import threading
from time import sleep, ctime


def loop0():
    print(f"loop0 start at:{ctime()}")
    sleep(10)
    print(f"loop0 end at:{ctime()}")


def loop1():
    print(f"loop1 start at:{ctime()}")
    sleep(4)
    print(f"loop1 end at:{ctime()}")


def main():
    print('程序开始于：', ctime())
    threads = []
    threads.append(threading.Thread(target=loop0(), args=None))
    threads.append(threading.Thread(target=loop1(), args=None))

    for i in range(len(threads)):
        threads[i].start()
        threads[i].join()

    print('任务完成于：', ctime())


if __name__ == '__main__':
    main()
