import threading
from threading import Lock

lock = Lock()

exitFlag = 0


class MyThread(threading.Thread):
    def __init__(self, thread_name, counter):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.counter = counter

    def run(self) -> None:
        while self.counter["number"] > 0:
            # 加锁
            # lock.acquire()
            self.counter["number"] -= 1
            # 释放锁
            # lock.release()
            # sleep(0.2)
            print(f"{self.thread_name} current number is {self.counter['number']} \n")


def main():
    counter = {"number": 10000000}
    # 和Java的类似
    thread1 = MyThread("Thread-1", counter)
    thread2 = MyThread("Thread-2", counter)
    thread1.start()
    thread2.start()


if __name__ == '__main__':
    main()
