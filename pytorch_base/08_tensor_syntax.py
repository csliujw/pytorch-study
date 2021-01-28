def f1(*args):
    print(args)


def f2(**kwargs):
    print(kwargs)


def main():
    f1(1, 2, 3)  # output (1,2,3)
    f2(a=1, b=2, c=3)  # output {'a': 1, 'b': 2, 'c': 3}


if __name__ == '__main__':
    main()
