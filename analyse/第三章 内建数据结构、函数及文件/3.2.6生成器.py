"""
创建生成器
"""


def squares(n=10):
    for i in range(1, n):
        yield i ** 2


gen = squares(20)

for i in gen:
    print(i)
