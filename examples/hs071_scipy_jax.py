from jax.config import config

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update('jax_platform_name', 'cpu')

from cyipopt import minimize_ipopt
from jax import jit, grad, jacrev, jacfwd
import jax.numpy as np


# Test the scipy interface on the Hock & Schittkowski test problem 71:
#
# min x0*x3*(x0+x1+x2)+x2
#
# s.t. x0**2 + x1**2 + x2**2 + x3**2 - 40 = 0
#      x0 * x1 * x2 * x3 - 25 >= 0
#      1 <= x0,x1,x2,x3 <= 5
#
# We evaluate all derivatives (except the Hessian) by algorithmic differentation
# by means of the JAX library.


def objective(x):
    return x[0]*x[3]*np.sum(x[:3]) + x[2]


def eq_constraints(x):
    return np.sum(x**2) - 40


def ineq_constrains(x):
    return np.prod(x) - 25


# jit the functions
obj_jit = jit(objective)
con_eq_jit = jit(eq_constraints)
con_ineq_jit = jit(ineq_constrains)

# build the derivatives and jit them
obj_grad = jit(grad(obj_jit))  # objective gradient
obj_hess = jit(jacrev(jacfwd(obj_jit)))  # objective hessian
con_eq_jac = jit(jacfwd(con_eq_jit))  # jacobian
con_ineq_jac = jit(jacfwd(con_ineq_jit))  # jacobian
con_eq_hess = jacrev(jacfwd(con_eq_jit)) # hessian
con_eq_hessvp = jit(lambda x, v: con_eq_hess(x) * v[0]) # hessian vector-product
con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
con_ineq_hessvp = jit(lambda x, v: con_ineq_hess(x) * v[0])  # hessian vector-product

# constraints
# Note that 'hess' is the hessian-vector-product
cons = [
    {'type': 'eq', 'fun': con_eq_jit, 'jac': con_eq_jac, 'hess': con_eq_hessvp},
    {'type': 'ineq', 'fun': con_ineq_jit, 'jac': con_ineq_jac, 'hess': con_ineq_hessvp},
]

# initial guess
x0 = np.array([1.0, 5.0, 5.0, 1.0])

# variable bounds: 1 <= x[i] <= 5
bnds = [(1, 5) for _ in range(x0.size)]

res = minimize_ipopt(obj_jit, jac=obj_grad, hess=obj_hess, x0=x0, bounds=bnds,
                     constraints=cons, options={'disp': 5})

print(res)
