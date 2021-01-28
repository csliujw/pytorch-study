def bisectTest1():
    import bisect # 针对有序数组
    c = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    index = bisect.bisect(c, 10)
    print(index)

bisectTest1()