import numpy as np
import time

rr = np.arange(100000)
rr = list(rr)

start1 = time.time()
for _ in range(10):
    rr = rr * 2
end1 = time.time()
print(end1 - start1)

start2 = time.time()
rr = [x * 2 for x in rr]
end2 = time.time()
print(end2 - start2)
