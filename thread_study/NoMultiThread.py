from time import sleep, ctime


def loop0():
    print(f"loop0 start at:{ctime()}")
    sleep(4)
    print(f"loop0 end at:{ctime()}")


def loop1():
    print(f"loop1 start at:{ctime()}")
    sleep(2)
    print(f"loop1 end at:{ctime()}")


def main():
    loop0()
    loop1()


if __name__ == '__main__':
    main()
