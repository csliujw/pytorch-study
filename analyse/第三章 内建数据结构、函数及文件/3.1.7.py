"""
字典
update 合并字典，相同key会覆盖
"""


# 记
def mergerDict():
    d1 = {"key1": "value1"}
    d2 = {"key2": "value2"}

    d1.update(d2)
    print(d1)


# 记
def tupleToDict():
    d1 = dict(((1, 2), (3, 4)))
    print(d1)


# 记
def defaultValue():
    d1 = {"key1": "value1"}
    print(d1.get("key1", "123"))
    print(d1.get("key2", "123"))
    print(d1.get("key2"))


def setDefault():
    d1 = {"key1": "value1", "key2": "value2", "key3": None}
    for key, value in d1.items():
        if value is None:
            d1.setdefault(key, "[123]")
    print(d1)


if __name__ == '__main__':
    setDefault()
