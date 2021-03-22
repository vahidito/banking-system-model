import numpy as np
import scipy as spy
from scipy.optimize import linprog

c = np.array([-29.0, -45.0, 0.0, 0.0])
A_ub = np.array([[1.0, -1.0, -3.0, 0.0], [-2.0, 3.0, 7.0, -3.0]])
b_ub = np.array([5.0, -10.0])
A_eq = np.array([[2.0, 8.0, 1.0, 0.0], [4.0, 4.0, 0.0, 1.0]])
b_eq = np.array([60.0, 60.0])
x0_bounds = (0, None)
x1_bounds = (0, 5.0)
x2_bounds = (-np.inf, 0.5)  # +/- np.inf can be used instead of None
x3_bounds = (-3.0, None)
bounds = [x0_bounds, x1_bounds, x2_bounds, x3_bounds]
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
print(result.x)
xxxx = result.x
print(type(xxxx))
aaa = list(xxxx)
print(type(aaa))
aaa.append(1)
print(aaa)
a = aaa[1]
print(type(a))


class What:
    def __init__(self, *args, **kwargs):

        pass


whatever = [What() for _ in range(200)]
obg = whatever[10]
print(obg)
