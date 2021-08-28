import numpy as np
import pandas as pd
from io import StringIO

# a = np.array([[1., 2], [3, 4]])
# # ap = pd.DataFrame(a)
# # print(ap)
# # aps = ap.to_string()
# # print(aps)
#
# file = open("desu.txt", "w")
#
# # file.write(f"{aps}")
#
# file.close()
#
# np.savetxt("d.csv", a, delimiter=",")
#
# b = np.genfromtxt("d.csv", delimiter=",")
# print(b)
# from utility.Utility import generate_counting_problem_unique
#
# x, y = generate_counting_problem_unique(countTo=4)
from ann_point.Functions import QuasiCrossEntropy

d = QuasiCrossEntropy()
res = d.compute( np.array([[1], [0]]), np.array([[1], [0]]))
print(res)

