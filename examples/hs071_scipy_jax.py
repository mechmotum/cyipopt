#! /usr/bin/env python3

import jax.numpy as np
import jax
from jax import jit, grad, jacfwd
from cyipopt import minimize_ipopt

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

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
jax.config.update('jax_platform_name', 'cpu')


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
obj_grad = jit(grad(obj_jit))  # gradient
con_eq_jac = jit(jacfwd(con_eq_jit))  # jacobian
con_ineq_jac = jit(jacfwd(con_ineq_jit))  # jacobian

# constraints
cons = [
    {'type': 'eq', 'fun': con_eq_jit, 'jac': con_eq_jac},
    {'type': 'ineq', 'fun': con_ineq_jit, 'jac': con_ineq_jac},
]

# initial guess
x0 = np.array([1, 5, 5, 1])

# variable bounds: 1 <= x[i] <= 5
bnds = [(1, 5) for _ in range(x0.size)]

res = minimize_ipopt(obj_jit, jac=obj_grad, x0=x0, bounds=bnds,
                     constraints=cons, options={'disp': 5})
