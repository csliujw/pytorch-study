"""
IO读取
"""
path = "./file"
data = None
with open(path) as f:
    for line in f:
        print(line)
