def sortTest():
    a = [3, 4, 23, 4, 5, 64, 5, 7, 75, 7]
    # a本身没变，返回的是新列表
    a_sort = sorted(a)
    # print(a)
    print(a_sort)


# 背 原地排序
def sortTest2():
    a = [str(x) for x in range(20)]
    a[0] = "12323"
    a.sort(key=len)
    print(a)

if __name__ == '__main__':
    sortTest()