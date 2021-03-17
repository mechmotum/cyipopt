import numpy as np
from scipy.optimize import rosen, rosen_der
from cyipopt import minimize_ipopt

x0 = np.array([0.5, 0])

bounds = [np.array([0, 1]), np.array([-0.5, 2.0])]


eq_cons = {'type': 'eq',
           'fun' : lambda x: np.array([2*x[0] + x[1] - 1, x[0]**2 - 0.1])
          }

res = minimize_ipopt(rosen, x0, jac=rosen_der, bounds=bounds, constraints=[eq_cons])

print(res)
