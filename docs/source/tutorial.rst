.. _tutorial:

=====
Usage
=====

SciPy Compatible Interface
==========================

For simple cases where you do not need the full power of sparse and structured
Jacobians etc, ``cyipopt`` provides the function ``minimize_ipopt`` which has
the same behaviour as ``scipy.optimize.minimize``, for example::

   >>> from scipy.optimize import rosen, rosen_der
   >>> from cyipopt import minimize_ipopt
   >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
   >>> res = minimize_ipopt(rosen, x0, jac=rosen_der)
   >>> print(res)
       fun: 2.1256746564022273e-18
      info: {'x': array([1., 1., 1., 1., 1.]), 'g': array([], dtype=float64), 'obj_val': 2.1256746564022273e-18, 'mult_g': array([], dtype=float64), 'mult_x_L': array([0., 0., 0., 0., 0.]), 'mult_x_U': array([0., 0., 0., 0., 0.]), 'status': 0, 'status_msg': b'Algorithm terminated successfully at a locally optimal point, satisfying the convergence tolerances (can be specified by options).'}
    message: b'Algorithm terminated successfully at a locally optimal point, satisfying the convergence tolerances (can be specified by options).'
      nfev: 200
       nit: 37
      njev: 39
    status: 0
   success: True
         x: array([1., 1., 1., 1., 1.])

Problem Interface
=================

In this example we will use cyipopt to solve an example problem, number 71 from
the Hock-Schittkowsky test suite [1]_,

.. math::

    \min_{x \in R^4}\ &x_1 x_4 (x_1 + x_2 + x_3 ) + x_3 \\
    s.t.\ &x_1 x_2 x_3 x_4 \geq 25 \\
          &x_1^2 + x_2^2 + x_3^2 + x_4^2 = 40 \\
          &1 \leq x_1, x_2, x_3, x_4 \leq 5, \\

with the starting point,

.. math::

   x_0 = (1,\ 5,\ 5,\ 1),

and the optimal solution,

.. math::

   x_* = (1.0,\ 4.743,\ 3.821,\ 1.379)


Getting started
---------------

Before you can use cyipopt, you have to import it::

   import cyipopt

This problem will also make use of NumPy::

   import numpy as np

Defining the problem
--------------------

The first step is to define a class that computes the objective and its
gradient, the constraints and its Jacobian, and the Hessian. The following
methods can be defined on the class:

- :func:`cyipopt.Problem.objective`
- :func:`cyipopt.Problem.gradient`
- :func:`cyipopt.Problem.constraints`
- :func:`cyipopt.Problem.jacobian`
- :func:`cyipopt.Problem.hessian`

The :func:`cyipopt.Problem.jacobian` and :func:`cyipopt.Problem.hessian`
methods should return the non-zero values of the respective matrices as
flattened arrays. The hessian should return a flattened lower triangular
matrix.

The Jacobian and Hessian can be dense or sparse. If sparse, you must also
define:

- :func:`cyipopt.Problem.jacobianstructure`
- :func:`cyipopt.Problem.hessianstructure`

which should return a tuple of indices that indicate the location of the
non-zero values of the Jacobian and Hessian matrices, respectively. If not
defined then these matrices are assumed to be dense.

The :func:`cyipopt.Problem.intermediate` method is called every Ipopt iteration
algorithm and can be used to perform any needed computation at each iteration.

Define the problem class::

   class HS071():

       def objective(self, x):
           """Returns the scalar value of the objective given x."""
           return x[0] * x[3] * np.sum(x[0:3]) + x[2]

       def gradient(self, x):
           """Returns the gradient of the objective with respect to x."""
           return np.array([
               x[0]*x[3] + x[3]*np.sum(x[0:3]),
               x[0]*x[3],
               x[0]*x[3] + 1.0,
               x[0]*np.sum(x[0:3])
           ])

       def constraints(self, x):
           """Returns the constraints."""
           return np.array((np.prod(x), np.dot(x, x)))

       def jacobian(self, x):
           """Returns the Jacobian of the constraints with respect to x."""
           return np.concatenate((np.prod(x)/x, 2*x))

       def hessianstructure(self):
           """Returns the row and column indices for non-zero vales of the
           Hessian."""

           # NOTE: The default hessian structure is of a lower triangular matrix,
           # therefore this function is redundant. It is included as an example
           # for structure callback.

           return np.nonzero(np.tril(np.ones((4, 4))))

       def hessian(self, x, lagrange, obj_factor):
           """Returns the non-zero values of the Hessian."""

           H = obj_factor*np.array((
               (2*x[3], 0, 0, 0),
               (x[3],   0, 0, 0),
               (x[3],   0, 0, 0),
               (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

           H += lagrange[0]*np.array((
               (0, 0, 0, 0),
               (x[2]*x[3], 0, 0, 0),
               (x[1]*x[3], x[0]*x[3], 0, 0),
               (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

           H += lagrange[1]*2*np.eye(4)

           row, col = self.hessianstructure()

           return H[row, col]

       def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                        d_norm, regularization_size, alpha_du, alpha_pr,
                        ls_trials):
           """Prints information at every Ipopt iteration."""

           msg = "Objective value at iteration #{:d} is - {:g}"

           print(msg.format(iter_count, obj_value))


Now define the lower and upper bounds of :math:`x` and the constraints::

    lb = [1.0, 1.0, 1.0, 1.0]
    ub = [5.0, 5.0, 5.0, 5.0]

    cl = [25.0, 40.0]
    cu = [2.0e19, 40.0]

Define an initial guess::

    x0 = [1.0, 5.0, 5.0, 1.0]

Define the full problem using the :class:`cyipopt.Problem` class::

    nlp = cyipopt.Problem(
       n=len(x0),
       m=len(cl),
       problem_obj=HS071(),
       lb=lb,
       ub=ub,
       cl=cl,
       cu=cu,
    )

The constructor of the :class:`cyipopt.Problem` class requires:

- ``n``: the number of variables in the problem,
- ``m``: the number of constraints in the problem,
- ``lb`` and ``ub``: lower and upper bounds on the variables,
- ``cl`` and ``cu``: lower and upper bounds of the constraints.
- ``problem_obj`` is an object whose methods implement ``objective``,
  ``gradient``, ``constraints``, ``jacobian``, and ``hessian`` of the problem.

Setting optimization parameters
-------------------------------

Setting optimization parameters is done by calling the
:func:`cyipopt.Problem.add_option` method, e.g.::

    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-7)

The different options and their possible values are described in the `ipopt
documentation <https://coin-or.github.io/Ipopt/OPTIONS.html>`_.

Executing the solver
--------------------

The optimization algorithm is run by calling the :func:`cyipopt.Problem.solve`
method, which accepts the starting point for the optimization as its only
parameter::

    x, info = nlp.solve(x0)

The method returns the optimal solution and an info dictionary that contains
the status of the algorithm, the value of the constraints multipliers at the
solution, and more.

Where to go from here
=====================

Once you feel sufficiently familiar with the basics, feel free to dig into the
:ref:`reference <reference>`. For more examples, check the :file:`examples/`
subdirectory of the distribution.

.. [1] W. Hock and K. Schittkowski. Test examples for nonlinear programming
   codes. Lecture Notes in Economics and Mathematical Systems, 187, 1981.
