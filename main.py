import numpy as np
import pandas as pd

a = np.array([[1., 2], [3, 4]])
ap = pd.DataFrame(a)
print(ap)
aps = ap.to_string()
print(aps)

file = open("desu.txt", "w")

file.write(f"{aps}")

file.close()
#
# file = open("desu.txt", "r")
#
# str = file.read()
# print(str)
#
# b = np.fromstring(" 1 2")
# print(b)