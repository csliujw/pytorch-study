"""
enumerate (index,value)
sorted ()
zip
reversed
"""


def enumerateTest():
    l = [1, 2, 3, 4, 5, 5, 6]
    for i, value in enumerate(l):
        print(i, value)


def sortedTest():
    new_ = sorted('horse race')
    print(new_)


def zipTest():
    seq1 = ['foo', 'bar', 'baz']
    seq2 = [1, 2, 3]
    a = zip(seq1, seq2)
    print(list(a))


def reversedTest():
    print(list(reversed(range(10))))


if __name__ == '__main__':
    reversedTest()
