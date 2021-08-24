import numpy as np
import pandas as pd
from io import StringIO

a = np.array([[1., 2], [3, 4]])
# ap = pd.DataFrame(a)
# print(ap)
# aps = ap.to_string()
# print(aps)

file = open("desu.txt", "w")

# file.write(f"{aps}")

file.close()

np.savetxt("d.csv", a, delimiter=",")

b = np.genfromtxt("d.csv", delimiter=",")
print(b)